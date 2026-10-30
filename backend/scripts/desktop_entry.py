"""Loopback-only desktop sidecar and source/browser launcher.

The frozen binary also handles ``--run-worker`` before importing FastAPI. Model
execution remains in separate runtime processes; this entry never loads models.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import secrets
import shutil
import socket
import sys
import threading
from urllib.parse import urlparse
from urllib.request import Request, HTTPRedirectHandler, ProxyHandler, build_opener

SOURCE_ROOT = Path(__file__).resolve().parents[1]
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description='Jingxi Insight local desktop backend; listens only on 127.0.0.1.')
    parser.add_argument('--port', type=int, default=0)
    parser.add_argument('--data-dir', type=Path)
    parser.add_argument('--registry', type=Path)
    parser.add_argument('--frontend-dir', type=Path)
    parser.add_argument('--runtime-root', type=Path)
    parser.add_argument('--reuse', action='store_true', help='Reuse only this project service with a matching private handshake and configuration fingerprint')
    parser.add_argument('--watch-stdin', action='store_true', help='Stop the service started by this launcher when stdin receives stop or closes')
    parser.add_argument('--open-browser', action='store_true')
    parser.add_argument('--api-only', action='store_true', help='Serve the API even when the frontend has not been built')
    args = parser.parse_args(argv)
    if not 0 <= args.port <= 65535:
        parser.error('The port must be between 0 and 65535.')
    return args


def resolve_paths(args):
    frozen = bool(getattr(sys, 'frozen', False))
    local = Path(os.getenv('LOCALAPPDATA') or Path.home() / '.local/share')
    data = (args.data_dir or os.getenv('REVIEW_DATA_DIR') or local / 'org.4c.workbench/state')
    resources = Path(getattr(sys, '_MEIPASS', SOURCE_ROOT))
    return {'data': Path(data).resolve(),
            'registry': Path(args.registry or os.getenv('REVIEW_LESSON_REGISTRY') or
                             (Path(data) / 'lessons.json' if frozen else SOURCE_ROOT / 'configs/lessons.json')).resolve(),
            'frontend': Path(args.frontend_dir or os.getenv('REVIEW_FRONTEND_DIR') or
                             (resources / 'frontend' if frozen else SOURCE_ROOT.parent / 'frontend/dist')).resolve(),
            'runtime': Path(args.runtime_root or os.getenv('REVIEW_RUNTIME_ROOT') or
                            (Path(data) / 'runtime' if frozen else SOURCE_ROOT)).resolve(),
            'resources': resources.resolve()}


def launch_fingerprint(paths):
    value = {key: str(paths[key]) for key in ('data', 'registry', 'frontend', 'runtime')}
    # StaticFiles is mounted only at startup. A service started before a build
    # cannot serve that build merely because index.html later appears.
    value['frontend_built'] = (paths['frontend'] / 'index.html').is_file()
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode('utf-8')).hexdigest()


def verified_service(state_path, fingerprint):
    """Never trust a port or PID alone. No network request outside numeric loopback."""
    try:
        state = json.loads(Path(state_path).read_text(encoding='utf-8'))
        url = urlparse(state['url'])
        if (url.scheme != 'http' or url.hostname != '127.0.0.1' or not url.port or
                url.path not in ('', '/') or url.username or url.password or url.query or url.fragment or
                state.get('fingerprint') != fingerprint or len(state.get('token', '')) < 32):
            return None
        request = Request(state['url'] + '/api/review/desktop/identity', headers={'X-4C-Desktop-Key': state['token']})
        class NoRedirect(HTTPRedirectHandler):
            def redirect_request(self, req, fp, code, msg, headers, newurl):
                return None
        with build_opener(ProxyHandler({}), NoRedirect()).open(request, timeout=1) as response:
            identity = json.load(response)
        if identity == {'service': '4c-desktop', 'fingerprint': fingerprint, 'token': state['token']}:
            return state
    except (OSError, ValueError, KeyError, TypeError):
        pass
    return None


def _ready(url, reused, open_browser):
    print('4C_DESKTOP_READY ' + json.dumps({'service': '4c-desktop', 'url': url, 'reused': reused}), flush=True)
    if open_browser:
        import webbrowser
        webbrowser.open(url)


def _seed_configs(paths):
    if paths['runtime'] == SOURCE_ROOT:
        return
    target = paths['runtime'] / 'configs'
    target.mkdir(parents=True, exist_ok=True)
    for name in ('runtime_profiles.json', 'local_llm.json', 'correction_llm.json', 'qwen35_llm.json', 'qwen_asr.json'):
        source = paths['resources'] / 'configs' / name
        if source.is_file() and not (target / name).exists():
            shutil.copyfile(source, target / name)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv == ['-m', 'review.courseware_alignment']:
        # Compatibility for the existing native-image isolation subprocess.
        # This is a fixed internal module, never a generic Python module runner.
        from review.courseware_alignment import _align_in_process
        payload = json.load(sys.stdin)
        json.dump(_align_in_process(payload['document'], payload['bundle'], payload['lesson_dir']),
                  sys.stdout, ensure_ascii=True, allow_nan=False)
        return 0
    if argv[:1] == ['--run-worker']:
        from review.job_worker import main as worker_main
        previous = sys.argv
        try:
            sys.argv = [sys.argv[0], *argv[1:]]
            worker_main()
        finally:
            sys.argv = previous
        return 0
    args = parse_args(argv)
    paths = resolve_paths(args)
    fingerprint = launch_fingerprint(paths)
    state_path = paths['data'] / 'desktop-service.json'
    if args.reuse and (state := verified_service(state_path, fingerprint)):
        _ready(state['url'], True, args.open_browser)
        if args.watch_stdin:
            for line in sys.stdin:
                if line.strip().lower() == 'stop':
                    break
        return 0
    if args.reuse:
        try:
            previous = json.loads(state_path.read_text(encoding='utf-8'))
        except (OSError,ValueError):
            previous = {}
        previous_fingerprint = previous.get('fingerprint') if isinstance(previous,dict) else None
        if isinstance(previous_fingerprint,str) and verified_service(state_path,previous_fingerprint):
            raise SystemExit('A service is running in this workspace, but the frontend build or startup configuration changed. Stop it with Ctrl+C in its original terminal, then restart. The existing service was left running.')
    paths['data'].mkdir(parents=True, exist_ok=True)
    if not paths['registry'].exists():
        paths['registry'].parent.mkdir(parents=True, exist_ok=True)
        paths['registry'].write_text('{}\n', encoding='utf-8')
    if not args.api_only and not (paths['frontend'] / 'index.html').is_file():
        raise SystemExit('Frontend dist/index.html is missing. Build the frontend or pass --frontend-dir.')
    _seed_configs(paths)
    os.environ.update({'REVIEW_DATA_DIR': str(paths['data']), 'REVIEW_LESSON_REGISTRY': str(paths['registry']),
                       'REVIEW_FRONTEND_DIR': str(paths['frontend']), 'REVIEW_RUNTIME_ROOT': str(paths['runtime'])})
    import uvicorn
    from fastapi import Request as FastAPIRequest
    from fastapi.routing import APIRoute
    from starlette.responses import JSONResponse
    from review.server import create_app

    token = secrets.token_hex(32)
    app = create_app()

    @app.middleware('http')
    async def desktop_headers(request, call_next):
        response = await call_next(request)
        response.headers['Content-Security-Policy'] = (
            "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data: blob:; media-src 'self' blob:; connect-src 'self'; "
            "font-src 'self' data:; object-src 'none'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'")
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['Referrer-Policy'] = 'no-referrer'
        return response

    async def identity(request: FastAPIRequest):
        if not secrets.compare_digest(request.headers.get('X-4C-Desktop-Key', ''), token):
            return JSONResponse({'error': 'forbidden'}, status_code=403)
        return {'service': '4c-desktop', 'fingerprint': fingerprint, 'token': token}

    # StaticFiles owns '/'; insert the private identity route before that mount.
    identity.__annotations__['request'] = FastAPIRequest
    app.router.routes.insert(0, APIRoute('/api/review/desktop/identity', identity, methods=['GET'], include_in_schema=False))
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    if hasattr(socket, 'SO_EXCLUSIVEADDRUSE'):
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
    sock.bind(('127.0.0.1', args.port))
    url = f'http://127.0.0.1:{sock.getsockname()[1]}'

    class DesktopServer(uvicorn.Server):
        async def startup(self, sockets=None):
            await super().startup(sockets=sockets)
            if self.should_exit:
                return
            temporary = state_path.with_name('desktop-service.' + token[:12] + '.partial')
            temporary.write_text(json.dumps({'url': url, 'token': token, 'fingerprint': fingerprint}), encoding='utf-8')
            temporary.replace(state_path)
            _ready(url, False, args.open_browser)

    server = DesktopServer(uvicorn.Config(app, host='127.0.0.1', port=args.port, workers=1,
                                          access_log=False, log_level='warning'))
    if args.watch_stdin:
        def watch():
            for line in sys.stdin:
                if line.strip().lower() == 'stop':
                    break
            server.should_exit = True
        threading.Thread(target=watch, name='desktop-owner-stdin', daemon=True).start()
    try:
        server.run(sockets=[sock])
    finally:
        sock.close()
        try:
            if json.loads(state_path.read_text(encoding='utf-8')).get('token') == token:
                state_path.unlink()
        except (OSError, ValueError):
            pass
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
