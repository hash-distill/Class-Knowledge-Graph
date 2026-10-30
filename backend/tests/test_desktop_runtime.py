"""Desktop boundaries: truthful file inventory, private paths, and owned lifecycle."""
import importlib.util
import builtins
import io
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sqlite3
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import threading
from urllib.request import Request, urlopen

import pytest

ROOT = Path(__file__).resolve().parents[1]


def runtime_module():
    from review import runtime
    return runtime


def entry_module():
    spec = importlib.util.spec_from_file_location('desktop_entry_test', ROOT / 'scripts/desktop_entry.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_profiles(root, profiles):
    path = root / 'configs/runtime_profiles.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({'schema_version': '1.0', 'profiles': profiles}), encoding='utf-8')
    return path


def profile(**extra):
    return {'id': 'qa', 'role': 'qa', 'label': '本地问答', 'config': 'configs/qa.json',
            'assets': [{'path': 'artifacts/models/small.gguf', 'kind': 'file'}],
            'runtimes': [{'path': 'artifacts/runtime/llama-server.exe', 'kind': 'file'}], **extra}


def test_config_does_not_imply_model_or_runtime_installed(tmp_path):
    write_profiles(tmp_path, [profile()])
    (tmp_path / 'configs/qa.json').write_text('{}')
    result = runtime_module().inspect_runtime(tmp_path)
    row = result['profiles'][0]
    assert row['config_status'] == 'present'
    assert row['assets_status'] == 'missing'
    assert row['runtime_status'] == 'missing'
    assert row['inference_status'] == 'not_verified'
    assert row['available'] is False
    assert result['model_loading'] is False


def test_inventory_requires_nonempty_files_and_does_not_load_models(tmp_path, monkeypatch):
    write_profiles(tmp_path, [profile()])
    (tmp_path / 'configs/qa.json').write_text('{}')
    for name in ('artifacts/models/small.gguf', 'artifacts/runtime/llama-server.exe'):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'asset')
    original_import = builtins.__import__
    def no_model_imports(name, *args, **kwargs):
        assert name.split('.')[0] not in {'torch', 'transformers', 'qwen_asr', 'onnxruntime', 'faster_whisper'}
        return original_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', no_model_imports)
    row = runtime_module().inspect_runtime(tmp_path)['profiles'][0]
    assert row['assets_status'] == 'installed'
    assert row['runtime_status'] == 'installed'
    assert row['inference_status'] == 'not_verified'
    assert row['available'] is False
    (tmp_path / 'artifacts/models/small.gguf').write_bytes(b'')
    assert runtime_module().inspect_runtime(tmp_path)['profiles'][0]['assets_status'] == 'missing'


def test_invalid_config_and_outside_paths_are_safe_and_redacted(tmp_path):
    secret = tmp_path.parent / 'private-account-model.gguf'
    secret.write_bytes(b'private')
    write_profiles(tmp_path, [profile(assets=[{'path': str(secret), 'kind': 'file'}])])
    (tmp_path / 'configs/qa.json').write_text('{broken')
    result = runtime_module().inspect_runtime(tmp_path)
    assert result['profiles'][0]['config_status'] == 'invalid'
    assert result['profiles'][0]['assets_status'] == 'missing'
    assert str(tmp_path) not in json.dumps(result)
    assert secret.name not in json.dumps(result)


def test_relative_symlink_cannot_disclose_external_asset(tmp_path):
    other = tmp_path.parent / 'outside-model'
    other.mkdir(exist_ok=True)
    (other / 'model.bin').write_bytes(b'secret')
    try:
        (tmp_path / 'linked').symlink_to(other, target_is_directory=True)
    except OSError:
        pytest.skip('Windows symlink privilege unavailable')
    write_profiles(tmp_path, [profile(assets=[{'path': 'linked/model.bin', 'kind': 'file'}])])
    assert runtime_module().inspect_runtime(tmp_path)['profiles'][0]['assets_status'] == 'missing'


def test_recommended_profiles_are_not_claimed_installed(tmp_path):
    result = runtime_module().inspect_runtime(tmp_path, ROOT / 'configs/runtime_profiles.json')
    assert {'asr', 'ocr', 'embeddings', 'qa', 'correction'} <= {p['role'] for p in result['profiles']}
    assert all(row['available'] is False for row in result['profiles'])
    assert all(row['assets_status'] == 'missing' for row in result['profiles'])


def test_bad_profile_shapes_and_non_path_config_values_report_missing(tmp_path):
    path = write_profiles(tmp_path, [profile(assets=[{'config_key': 'model.path'}], runtimes=[None])])
    (tmp_path / 'configs/qa.json').write_text(json.dumps({'model': {'path': 123}}))
    row = runtime_module().inspect_runtime(tmp_path)['profiles'][0]
    assert row['assets_status'] == 'missing' and row['runtime_status'] == 'missing'
    path.write_text(json.dumps({'profiles': None}))
    result = runtime_module().inspect_runtime(tmp_path)
    assert result['profiles_status'] == 'missing_or_invalid'
    assert result['profiles'] == []


def test_launch_paths_use_explicit_args_and_dedicated_data(tmp_path, monkeypatch):
    entry = entry_module()
    monkeypatch.setenv('LOCALAPPDATA', str(tmp_path / 'Local'))
    args = entry.parse_args(['--port', '0', '--registry', str(tmp_path / 'lessons.json'),
                             '--frontend-dir', str(tmp_path / '前端 dist')])
    paths = entry.resolve_paths(args)
    assert paths['registry'] == (tmp_path / 'lessons.json').resolve()
    assert paths['frontend'] == (tmp_path / '前端 dist').resolve()
    assert paths['data'].is_relative_to(tmp_path / 'Local')
    assert paths['data'] != tmp_path / 'Local'
    with pytest.raises(SystemExit):
        entry.parse_args(['--host', '0.0.0.0'])


def test_worker_mode_strips_only_marker_and_delegates(monkeypatch):
    entry = entry_module()
    import review.job_worker
    seen = []
    monkeypatch.setattr(review.job_worker, 'main', lambda: seen.append(sys.argv[1:]))
    argv = ['--db', 'jobs.sqlite3', '--registry', 'lessons.json', '--job', 'j1', '--token', 'token']
    assert entry.main(['--run-worker', *argv]) == 0
    assert seen == [argv]


def test_alignment_subworker_keeps_its_fixed_module_json_protocol(monkeypatch):
    entry = entry_module()
    output = io.StringIO()
    monkeypatch.setattr(sys, 'stdin', io.StringIO(json.dumps({'document': {'document_id': 'D1', 'pages': []}, 'bundle': {'slides': []}, 'lesson_dir': '.'})))
    monkeypatch.setattr(sys, 'stdout', output)
    assert entry.main(['-m', 'review.courseware_alignment']) == 0
    assert json.loads(output.getvalue()) == []
    with pytest.raises(SystemExit):
        entry.main(['-m', 'arbitrary.module'])


def test_only_verified_service_with_matching_launch_fingerprint_is_reused(tmp_path):
    entry = entry_module()
    assert entry.verified_service(tmp_path / 'absent.json', 'fingerprint') is None
    state = tmp_path / 'service.json'
    state.write_text(json.dumps({'url': 'https://example.com', 'token': 'x', 'fingerprint': 'fingerprint'}))
    assert entry.verified_service(state, 'fingerprint') is None


def test_identity_probe_never_follows_redirects(tmp_path):
    entry = entry_module()
    calls = []
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            calls.append(self.path)
            self.send_response(302)
            self.send_header('Location', '/unexpected-target')
            self.end_headers()
        def log_message(self, *args):
            pass
    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    state = tmp_path / 'service.json'
    state.write_text(json.dumps({'url': f'http://127.0.0.1:{server.server_port}', 'token': 'x' * 64, 'fingerprint': 'fp'}))
    try:
        assert entry.verified_service(state, 'fp') is None
        assert calls == ['/api/review/desktop/identity']
    finally:
        server.shutdown()
        server.server_close()
    state.write_text(json.dumps({'url': 'http://127.0.0.1:1', 'token': 'x', 'fingerprint': 'different'}))
    assert entry.verified_service(state, 'fingerprint') is None


@pytest.mark.parametrize('frozen,api_only', [(False,False),(False,True),(True,False)], ids=['source', 'api-only', 'frozen'])
def test_real_backend_lifecycle_and_reuse_do_not_stop_original(tmp_path, frozen, api_only):
    binary = os.environ.get('REVIEW_DESKTOP_BINARY')
    if frozen and not binary:
        pytest.skip('Set REVIEW_DESKTOP_BINARY to exercise the built sidecar')
    registry = tmp_path / 'lessons.json'
    registry.write_text('{}')
    frontend = tmp_path / 'frontend'
    if not api_only:
        frontend.mkdir()
        (frontend / 'index.html').write_text('<!doctype html><title>4C Test</title>')
    command = [binary] if frozen else [sys.executable, str(ROOT / 'scripts/desktop_entry.py')]
    args = [*command, '--port', '0',
            '--data-dir', str(tmp_path / 'state'), '--registry', str(registry),
            '--frontend-dir', str(frontend), '--watch-stdin']
    if api_only:args.append('--api-only')
    environment = {**os.environ, 'TMP': str(tmp_path), 'TEMP': str(tmp_path)}
    first = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                             text=True, encoding='utf-8', env=environment)
    second = None
    try:
        ready = first.stdout.readline().strip()
        assert ready.startswith('4C_DESKTOP_READY '), ready + first.stderr.read() if first.poll() is not None else ready
        info = json.loads(ready.partition(' ')[2])
        with urlopen(info['url'] + '/api/review/health', timeout=5) as response:
            assert json.load(response)['service'] == 'classroom-review'
        with urlopen(info['url'] + '/api/review/runtime', timeout=5) as response:
            inventory = json.load(response)
            assert inventory['inspection'] == 'local_files_only' and inventory['model_loading'] is False
            assert str(tmp_path) not in json.dumps(inventory)
        if not api_only:
            with urlopen(info['url'], timeout=5) as response:
                assert '4C Test' in response.read().decode()
                assert "default-src 'self'" in response.headers['Content-Security-Policy']
                assert "object-src 'none'" in response.headers['Content-Security-Policy']
        second = subprocess.Popen([*args, '--reuse'], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE, text=True, encoding='utf-8', env=environment)
        reused = json.loads(second.stdout.readline().partition(' ')[2])
        assert reused['url'] == info['url'] and reused['reused'] is True
        second.communicate('stop\n', timeout=8)
        assert first.poll() is None
        with urlopen(info['url'] + '/api/review/health', timeout=5) as response:
            assert response.status == 200
        if api_only:
            frontend.mkdir()
            (frontend/'index.html').write_text('<!doctype html><title>Fresh build</title>')
            full_args=[arg for arg in args if arg!='--api-only']
            second=subprocess.Popen([*full_args,'--reuse'],stdin=subprocess.PIPE,stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE,text=True,encoding='utf-8',env=environment)
            output,error=second.communicate(timeout=8)
            assert second.returncode!=0 and '4C_DESKTOP_READY' not in output
            assert 'Ctrl+C' in error and '已有服务未停止' in error
            assert first.poll() is None
        first.communicate('stop\n', timeout=8)
        assert first.returncode == 0
        assert not (tmp_path / 'state/desktop-service.json').exists()
        if api_only:
            second=subprocess.Popen([*full_args,'--reuse'],stdin=subprocess.PIPE,stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE,text=True,encoding='utf-8',env=environment)
            full=json.loads(second.stdout.readline().partition(' ')[2])
            assert full['reused'] is False
            with urlopen(full['url'],timeout=5) as response:
                assert 'Fresh build' in response.read().decode()
            second.communicate('stop\n',timeout=8)
    finally:
        for child in (second, first):
            if child is not None and child.poll() is None:
                child.kill()
                child.communicate(timeout=5)


def test_frozen_sidecar_imports_pdf_through_worker_and_alignment_processes(tmp_path):
    binary = os.environ.get('REVIEW_DESKTOP_BINARY')
    if not binary:
        pytest.skip('Set REVIEW_DESKTOP_BINARY to exercise frozen PDF import')
    import pymupdf
    lesson = tmp_path / 'lesson'
    lesson.mkdir()
    video = lesson / 'source.mp4'
    video.write_bytes(b'Synthetic source identity only; this test does not decode video.')
    bundle = {'schema_version': '3.0', 'data_source': 'lesson_review',
        'lesson': {'id': 'desktop-test', 'title': 'Synthetic desktop import', 'duration_sec': 20,
                   'status': 'complete', 'source_video': {'filename': 'source.mp4',
                       'sha256': hashlib.sha256(video.read_bytes()).hexdigest()}, 'provenance': {}},
        'transcript': [{'id': 'T0001', 'text': 'Synthetic normal reference lesson', 'start_sec': 1, 'end_sec': 5}],
        'slides': [{'id': 'S0001', 'text': 'Synthetic normal reference lesson', 'start_sec': 0, 'end_sec': 10}],
        'alignment': [], 'graph': {'nodes': [], 'edges': []}, 'snapshots': [],
        'report': {'title': 'Synthetic', 'summary': 'Fixture only', 'sections': [], 'limitations': []}}
    # Exercise the size of the real queued classroom payload without importing
    # private source content. Public queue/status queries must stay compact.
    bundle['validation_fixture_padding'] = 'x' * (8 * 1024 * 1024)
    (lesson / 'bundle.json').write_text(json.dumps(bundle), encoding='utf-8')
    registry = tmp_path / 'registry.json'
    registry.write_text(json.dumps({'desktop-test': {'directory': str(lesson), 'source_video': str(video)}}), encoding='utf-8')
    frontend = tmp_path / 'frontend'
    frontend.mkdir()
    (frontend / 'index.html').write_text('<!doctype html><title>Frozen worker test</title>')
    args = [binary, '--port', '0', '--data-dir', str(tmp_path / 'state'), '--registry', str(registry),
            '--frontend-dir', str(frontend), '--watch-stdin']
    child = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                             text=True, encoding='utf-8', env={**os.environ, 'TMP': str(tmp_path), 'TEMP': str(tmp_path)})
    try:
        ready = child.stdout.readline()
        assert ready.startswith('4C_DESKTOP_READY '), ready
        url = json.loads(ready.partition(' ')[2])['url']
        with pymupdf.open() as pdf:
            pdf.new_page().insert_text((60, 80), 'Synthetic normal reference lesson')
            content = pdf.tobytes()
        request = Request(url + '/api/review/lessons/desktop-test/courseware', data=content,
                          headers={'Content-Type': 'application/octet-stream', 'X-Filename': 'synthetic.pdf'})
        with urlopen(request, timeout=10) as response:
            assert response.status == 202
            job = json.load(response)
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            with urlopen(url + '/api/review/jobs/' + job['job_id'], timeout=10) as response:
                state = json.load(response)
            if state['status'] not in ('queued', 'running'):
                break
            time.sleep(.1)
        with sqlite3.connect(tmp_path / 'state/jobs.sqlite3') as db:
            detail = db.execute('SELECT error_detail FROM jobs WHERE job_id=?', (job['job_id'],)).fetchone()
        assert state['status'] == 'complete', (state, detail)
        assert json.loads((lesson / 'bundle.base.json').read_text(encoding='utf-8')) == bundle
        assert json.loads((lesson / 'bundle.json').read_text(encoding='utf-8'))['courseware']['pages'][0]['text'].strip() == 'Synthetic normal reference lesson'
        child.communicate('stop\n', timeout=10)
        assert child.returncode == 0
    finally:
        if child.poll() is None:
            child.kill()
            child.communicate(timeout=5)
