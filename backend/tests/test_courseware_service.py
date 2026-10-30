"""Bounded API and transactional overlay tests; fixtures are explicitly synthetic."""
import copy
import json
import sys
import threading
import time
from pathlib import Path
from fastapi.testclient import TestClient
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from review.common import atomic_json
from review.courseware_service import CoursewareService, source_fingerprint, overlay_bundle
from review.provider import ReviewError
from review.server import LessonStore, create_app
from review.retrieval import evidence_catalog, evidence_prompt
from review.mutation import LessonMutationLock


def fixture(tmp_path):
    bundle = {'schema_version': '3.0', 'data_source': 'lesson_review',
              'lesson': {'id': 'test', 'title': 'Synthetic service fixture', 'duration_sec': 20,
                         'status': 'complete', 'source_video': {'filename': 'test.mp4', 'sha256': 'a'*64}, 'provenance': {}},
              'transcript': [{'id': 'T0001', 'text': '鄭賢定理', 'source': 'asr', 'start_sec': 1, 'end_sec': 5}],
              'slides': [{'id': 'S0001', 'text': '正弦定理', 'start_sec': 0, 'end_sec': 10}],
              'alignment': [], 'graph': {'nodes': [], 'edges': []},
              'report': {'title': 'Test', 'summary': 'Fixture', 'sections': [], 'limitations': []}, 'snapshots': []}
    document = {'document_id': 'Dtest', 'filename': 'fixture.pdf', 'sha256': 'b'*64,
                'format': 'pdf', 'pages': [{'id': 'DtestP0001', 'page_number': 1, 'text': '正弦定理'}],
                'matches': [{'id': 'M0001', 'slide_id': 'S0001', 'page_id': 'DtestP0001', 'start_sec': 0,
                             'end_sec': 10, 'status': 'auto_matched', 'score': .9, 'margin': .4, 'method': 'synthetic'}],
                'lesson_fingerprint': source_fingerprint(bundle)}
    atomic_json(tmp_path/'bundle.json', bundle)
    atomic_json(tmp_path/'registry.json', {'test': {'directory': str(tmp_path), 'source_video': str(tmp_path/'test.mp4')}})
    return bundle, document, LessonStore(tmp_path/'registry.json')


def fake_derived(base, document):
    matches = [m for m in document['matches'] if m['status'] in ('auto_matched', 'confirmed')]
    return {'transcript_revisions': [], 'revised_evidence_blocks': [], 'courseware_evidence': [
        {'id': 'E0001', 'kind': 'courseware', 'text': document['pages'][0]['text'],
         'page_id': document['pages'][0]['id'], 'start_sec': m['start_sec'], 'end_sec': m['end_sec']} for m in matches]}


def mock_modules(monkeypatch, import_fn=None):
    import types
    monkeypatch.setitem(sys.modules, 'review.courseware_terms', types.SimpleNamespace(revise_transcript=fake_derived))
    if import_fn:
        monkeypatch.setitem(sys.modules, 'review.courseware_documents', types.SimpleNamespace(import_document=import_fn))
        monkeypatch.setitem(sys.modules, 'review.courseware_alignment', types.SimpleNamespace(align_document=lambda d,b,p:d['matches']))


def test_overlay_keeps_raw_and_rejects_different_source(tmp_path, monkeypatch):
    base, document, _ = fixture(tmp_path)
    snapshot = copy.deepcopy(base)
    mock_modules(monkeypatch)
    overlay = overlay_bundle(base, document)
    assert base == snapshot and overlay['transcript'] == base['transcript']
    assert overlay['courseware_evidence'][0]['start_sec'] == 0
    changed = copy.deepcopy(base); changed['transcript'][0]['text'] = 'Changed ASR'
    with pytest.raises(ReviewError, match='原始课堂证据已变化'): overlay_bundle(changed, document)


def test_completed_import_status_is_published_after_mutation_lease_release(tmp_path, monkeypatch):
    _, document, store = fixture(tmp_path)
    mock_modules(monkeypatch, lambda *args: copy.deepcopy(document))
    service = CoursewareService(store)
    states_at_release = []
    original = LessonMutationLock.release
    def release(lease):
        states_at_release.extend(job['status'] for job in service.jobs.values())
        original(lease)
    monkeypatch.setattr(LessonMutationLock, 'release', release)
    job = service.submit('test', 'test.pdf', b'%PDF-test')
    for _ in range(200):
        state = service.status(job['job_id'])
        if state['status'] != 'running' and states_at_release: break
        time.sleep(.01)
    assert state['status'] == 'complete', state
    assert states_at_release == ['running']
    assert service.remove('test') is not None


def test_courseware_refresh_respects_explicit_restricted_graph_display_policy(tmp_path, monkeypatch):
    import types
    base, document, _ = fixture(tmp_path)
    base['graph']['nodes']=[{'id':'K1','label':'正弦定理','properties':{'source_refs':['T0001']}}]
    base['graph']['provenance']={'display_policy':{'name':'model_candidates_with_observed_video_courseware_literal_support_v1'}}
    document['pages'][0]['text']='正弦定理与余弦定理'
    def derived(*args):
        result=fake_derived(base,document)
        result['transcript_revisions']=[{'edits':[{'replacement':'余弦定理','page_ids':['DtestP0001']}]}]
        return result
    monkeypatch.setitem(sys.modules,'review.courseware_terms',types.SimpleNamespace(revise_transcript=derived))
    restricted=overlay_bundle(base,document)
    assert [node['label'] for node in restricted['graph']['nodes']] == ['正弦定理']
    unrestricted=copy.deepcopy(base);unrestricted['graph'].pop('provenance')
    assert '余弦定理' in [node['label'] for node in overlay_bundle(unrestricted,document)['graph']['nodes']]


def test_import_confirm_reject_remove_and_stale_document(tmp_path, monkeypatch):
    base, document, store = fixture(tmp_path)
    mock_modules(monkeypatch, lambda *args: copy.deepcopy(document))
    service = CoursewareService(store)
    job = service.submit('test', '真实文件.pdf', b'%PDF-test')
    for _ in range(100):
        state = service.status(job['job_id'])
        if state['status'] != 'running': break
        time.sleep(.01)
    assert state['status'] == 'complete', state
    assert store.load('test')['courseware']['filename'] == '真实文件.pdf'
    with pytest.raises(ReviewError): service.confirm('test', {'document_id': 'old', 'match_id': 'M0001', 'page_id': None})
    rejected = service.confirm('test', {'document_id': 'Dtest', 'match_id': 'M0001', 'page_id': None})
    assert rejected['courseware_evidence'] == []
    confirmed = service.confirm('test', {'document_id': 'Dtest', 'match_id': 'M0001', 'page_id': 'DtestP0001'})
    assert confirmed['courseware']['matches'][0]['status'] == 'confirmed'
    assert service.remove('test') == base
    assert (tmp_path/'courseware').is_dir()  # detach preserves uploaded files


def test_failed_import_and_busy_never_publish_partial(tmp_path, monkeypatch):
    base, _, store = fixture(tmp_path)
    def fail(*args): raise ValueError('Synthetic conversion failure')
    mock_modules(monkeypatch, fail)
    service = CoursewareService(store)
    service.acquire()
    with pytest.raises(ReviewError): service.submit('test', 'test.pdf', b'x')
    service.lock.release()
    job = service.submit('test', 'test.pdf', b'x')
    for _ in range(100):
        state = service.status(job['job_id'])
        if state['status'] != 'running': break
        time.sleep(.01)
    assert state['status'] == 'failed' and store.load('test') == base
    for name in ('../test.pdf', 'a\\test.pdf', 'x.exe'):
        with pytest.raises(ReviewError): service.submit('test', name, b'x')


def test_retrieval_returns_full_source_even_when_prompt_is_bounded():
    row = {'id': 'D1E1', 'kind': 'courseware', 'text': 'x'*2000, 'start_sec': 0, 'end_sec': 5}
    prompt, used = evidence_prompt([row])
    assert len(prompt) < 1500 and used[0]['text'] == row['text']
    bundle = {'transcript': [], 'slides': [], 'courseware_evidence': [row],
              'revised_evidence_blocks': [{'id': 'RB1', 'kind': 'transcript_revision', 'text': '正弦定理', 'start_sec': 0, 'end_sec': 5}]}
    assert {r['kind'] for r in evidence_catalog(bundle)} == {'courseware', 'transcript_revision'}
    bundle['slides'] = [{'id': 'Snoise', 'text': '|\n.', 'start_sec': 0, 'end_sec': 5}]
    assert 'Snoise' not in {r['id'] for r in evidence_catalog(bundle)}


def test_stale_async_import_cannot_overwrite_updated_bundle(tmp_path, monkeypatch):
    base, document, store = fixture(tmp_path)
    entered, release = threading.Event(), threading.Event()
    def pause(*args):
        entered.set(); assert release.wait(5)
        return copy.deepcopy(document)
    mock_modules(monkeypatch, pause)
    service = CoursewareService(store)
    job = service.submit('test', 'test.pdf', b'%PDF-test')
    assert entered.wait(5)
    # An out-of-band file edit ignores our OS lock; optimistic version still detects it.
    newer = copy.deepcopy(base); newer['transcript'][0]['text'] = 'New source evidence'
    atomic_json(tmp_path/'bundle.json', newer)
    with pytest.raises(ReviewError):
        with LessonMutationLock(tmp_path): pass
    release.set()
    for _ in range(100):
        state = service.status(job['job_id'])
        if state['status'] != 'running': break
        time.sleep(.01)
    assert state['status'] == 'failed' and state['error']['code'] == 'stale_lesson'
    assert store.load('test') == newer and not (tmp_path/'bundle.base.json').exists()
    assert not list(tmp_path.rglob('automatic-match-cache.json'))
    with LessonMutationLock(tmp_path): pass  # lease released on failure


def test_images_changed_during_alignment_publish_neither_bundle_nor_cache(tmp_path, monkeypatch):
    import types
    base, document, store = fixture(tmp_path)
    base['slides'][0]['image_path'] = 'screen.jpg'
    atomic_json(tmp_path/'bundle.json', base)
    (tmp_path/'screen.jpg').write_bytes(b'old-image')
    entered, release = threading.Event(), threading.Event()
    mock_modules(monkeypatch, lambda *args: copy.deepcopy(document))
    def pause(*args):
        entered.set(); assert release.wait(5)
        return copy.deepcopy(document['matches'])
    monkeypatch.setitem(sys.modules, 'review.courseware_alignment', types.SimpleNamespace(align_document=pause))
    service = CoursewareService(store)
    job = service.submit('test', 'test.pdf', b'%PDF-test')
    assert entered.wait(5)
    (tmp_path/'screen.jpg').write_bytes(b'new-image')
    release.set()
    for _ in range(100):
        state = service.status(job['job_id'])
        if state['status'] != 'running': break
        time.sleep(.01)
    assert state['status'] == 'failed' and state['error']['code'] == 'stale_lesson_images'
    assert store.load('test') == base
    assert not list(tmp_path.rglob('automatic-match-cache.json'))


def test_reimport_reuses_only_automatic_matches_not_manual_overrides(tmp_path, monkeypatch):
    import types
    _, document, store = fixture(tmp_path)
    mock_modules(monkeypatch, lambda *args: copy.deepcopy(document))
    calls = []
    def align(*args):
        calls.append(True)
        return copy.deepcopy(document['matches'])
    monkeypatch.setitem(sys.modules, 'review.courseware_alignment', types.SimpleNamespace(align_document=align))
    service = CoursewareService(store)
    def run():
        job = service.submit('test', 'test.pdf', b'%PDF-test')
        for _ in range(200):
            state = service.status(job['job_id'])
            if state['status'] != 'running': break
            time.sleep(.01)
        assert state['status'] == 'complete', state
    run()
    service.confirm('test', {'document_id': 'Dtest', 'match_id': 'M0001', 'page_id': None})
    assert store.load('test')['courseware']['matches'][0]['status'] == 'rejected'
    service.remove('test')
    run()
    assert len(calls) == 1
    assert store.load('test')['courseware']['matches'][0]['status'] == 'auto_matched'
    assert store.load('test')['courseware']['alignment_cache_reused'] is True
    # Stable filenames may be regenerated by media processing; pixels invalidate cache.
    service.remove('test')
    base = store.load('test')
    base['slides'][0]['image_path'] = 'screen.jpg'
    (tmp_path/'screen.jpg').write_bytes(b'first-image')
    atomic_json(tmp_path/'bundle.json', base)
    run()
    assert len(calls) == 2
    service.remove('test')
    (tmp_path/'screen.jpg').write_bytes(b'updated-image-at-same-path')
    run()
    assert len(calls) == 3


def test_http_binary_upload_poll_confirm_detach_and_origin(tmp_path, monkeypatch):
    base, document, store = fixture(tmp_path)
    mock_modules(monkeypatch, lambda *args: copy.deepcopy(document))
    class Provider:
        def status(self): return {'source': 'synthetic_fixture'}
    endpoint = '/api/review/lessons/test/courseware'
    with TestClient(create_app(store=store, provider=Provider(), courseware_service=CoursewareService(store))) as client:
        denied = client.post(endpoint, content=b'x', headers={'X-Filename': 'test.pdf', 'Origin': 'https://example.com'})
        assert denied.status_code == 403
        accepted = client.post(endpoint, content=b'%PDF-fixture', headers={'X-Filename': 'test.pdf'})
        assert accepted.status_code == 202
        for _ in range(100):
            state = client.get('/api/review/jobs/'+accepted.json()['job_id']).json()
            if state['status'] != 'running': break
            time.sleep(.01)
        assert state['status'] == 'complete', state
        assert client.post(endpoint+'/matches', json={'document_id': 'Dtest', 'match_id': 'M0001', 'page_id': None}).status_code == 200
        assert client.delete(endpoint).json() == base
