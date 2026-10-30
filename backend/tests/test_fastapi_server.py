"""HTTP compatibility checks for the FastAPI migration; all media are synthetic."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys
import threading

from fastapi.testclient import TestClient
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from review.common import atomic_json
from review.provider import ReviewError
from review import server


class Provider:
    def status(self):
        return {'source': 'test_only', 'model': 'synthetic'}

    def complete(self, *args, **kwargs):
        return {'points': [{'text': '测试内容', 'evidence_ids': ['T0001']}]}, self.status()


def store_fixture(tmp_path):
    video = tmp_path/'video.mp4'
    video.write_bytes(b'0123456789')
    atomic_json(tmp_path/'bundle.json', {
        'schema_version': '3.0', 'data_source': 'lesson_review',
        'lesson': {'id': 'test', 'title': 'Synthetic HTTP fixture', 'duration_sec': 10,
                   'status': 'complete'},
        'transcript': [{'id': 'T0001', 'start_sec': 0, 'end_sec': 5, 'text': '测试内容'}],
        'slides': [],
    })
    atomic_json(tmp_path/'registry.json', {
        'test': {'directory': str(tmp_path), 'source_video': str(video)},
    })
    return server.LessonStore(tmp_path/'registry.json')


def app_fixture(tmp_path, provider=None):
    factory = getattr(server, 'create_app', None)
    assert callable(factory), 'The review API must expose a FastAPI application factory.'
    return factory(store=store_fixture(tmp_path), provider=provider or Provider())


def test_openapi_describes_existing_chat_contract_and_health(tmp_path):
    with TestClient(app_fixture(tmp_path)) as client:
        health = client.get('/api/review/health')
        assert health.status_code == 200
        assert health.json()['framework'] == 'fastapi'
        assert health.json()['service'] == 'classroom-review'
        assert health.headers['Cache-Control'] == 'no-store'
        assert health.headers['X-Content-Type-Options'] == 'nosniff'
        schema = client.get('/openapi.json').json()
        chat = schema['paths']['/api/review/chat']['post']
        assert 'application/json' in chat['requestBody']['content']
        upload = schema['paths']['/api/review/lessons/{lesson_id}/courseware']['post']
        assert 'application/octet-stream' in upload['requestBody']['content']
        assert any(p['in'] == 'header' and p['name'] == 'X-Filename' for p in upload['parameters'])
        assert client.get('/docs').status_code == 200


@pytest.mark.parametrize('payload,code', [
    ([], 'invalid_json'),
    ({'lesson_id': 'test', 'query': ''}, 'invalid_query'),
    ({'lesson_id': 'test', 'query': '测试', 'at_sec': True}, 'invalid_time'),
    ({'lesson_id': 'test', 'query': '测试', 'at_sec': 11}, 'invalid_time'),
    ({'lesson_id': 'test', 'query': '测试', 'scope': 'other'}, 'invalid_scope'),
])
def test_chat_validation_keeps_error_envelope(tmp_path, payload, code):
    with TestClient(app_fixture(tmp_path)) as client:
        response = client.post('/api/review/chat', json=payload)
        assert response.status_code == 400
        assert response.json()['error']['code'] == code
        assert set(response.json()) == {'error'}


def test_native_chat_citations_and_model_failure_are_preserved(tmp_path):
    with TestClient(app_fixture(tmp_path)) as client:
        response = client.post('/api/review/chat', json={'lesson_id': 'test', 'query': '测试'})
        assert response.status_code == 200
        assert response.json()['citations'][0]['text'] == '测试内容'
        assert response.json()['request_id']

    class Offline(Provider):
        def complete(self, *args, **kwargs):
            raise ReviewError('model_unavailable', '模型未就绪', 503,
                              diagnostic={'private': 'must not be exposed'})

    with TestClient(app_fixture(tmp_path, Offline())) as client:
        response = client.post('/api/review/chat', json={'lesson_id': 'test', 'query': '测试'})
        assert response.status_code == 503
        assert response.json() == {'error': {'code': 'model_unavailable', 'message': '模型未就绪'}}


def test_media_get_head_ranges_and_missing_files(tmp_path):
    with TestClient(app_fixture(tmp_path)) as client:
        endpoint = '/api/review/media/test/source.mp4'
        response = client.get(endpoint, headers={'Range': 'bytes=2-5'})
        assert response.status_code == 206 and response.content == b'2345'
        assert response.headers['Content-Range'] == 'bytes 2-5/10'
        assert response.headers['Content-Length'] == '4'
        head = client.head(endpoint, headers={'Range': 'bytes=-3'})
        assert head.status_code == 206 and head.content == b''
        assert head.headers['Content-Range'] == 'bytes 7-9/10'
        assert head.headers['Content-Length'] == '3'
        invalid = client.get(endpoint, headers={'Range': 'bytes=100-'})
        assert invalid.status_code == 416 and invalid.headers['Content-Range'] == 'bytes */10'
        assert client.get('/api/review/media/test/missing.pdf').status_code == 404
        assert client.get('/api/review/media/test/private.json').status_code == 403
        assert client.get('/api/review/not-found').json()['error']['code'] == 'not_found'


def test_raw_upload_limit_origin_and_invalid_json_are_preserved(tmp_path):
    with TestClient(app_fixture(tmp_path)) as client:
        upload = '/api/review/lessons/test/courseware'
        denied = client.post(upload, content=b'x', headers={'Origin': 'https://example.com'})
        assert denied.status_code == 403 and denied.json()['error']['code'] == 'origin_rejected'
        huge = client.post(upload, content=b'x', headers={'Content-Length': str(64*1024*1024+1)})
        assert huge.status_code == 413
        malformed = client.post('/api/review/chat', content=b'{', headers={'Content-Type': 'application/json'})
        assert malformed.status_code == 400 and malformed.json()['error']['code'] == 'invalid_json'
        invalid_utf8 = client.post('/api/review/chat', content=b'\xff', headers={'Content-Type': 'application/json'})
        assert invalid_utf8.status_code == 400 and invalid_utf8.json()['error']['code'] == 'invalid_json'
        oversized = client.post('/api/review/chat', content=b'x'*65537)
        assert oversized.status_code == 413
        mismatch = client.post('/api/review/chat', content=b'{}', headers={'Content-Length': '3'})
        assert mismatch.status_code == 400 and mismatch.json()['error']['code'] == 'incomplete_upload'


def test_busy_chat_does_not_block_health_or_allow_second_generation(tmp_path):
    started, release = threading.Event(), threading.Event()

    class SlowProvider(Provider):
        def complete(self, *args, **kwargs):
            started.set()
            assert release.wait(5), 'The test did not release its synthetic provider.'
            return super().complete(*args, **kwargs)

    with TestClient(app_fixture(tmp_path, SlowProvider())) as client, ThreadPoolExecutor() as pool:
        pending = pool.submit(client.post, '/api/review/chat', json={'lesson_id': 'test', 'query': '测试'})
        try:
            assert started.wait(2)
            assert client.get('/api/review/health').status_code == 200
            busy = client.post('/api/review/chat', json={'lesson_id': 'test', 'query': '测试'})
            assert busy.status_code == 429 and busy.json()['error']['code'] == 'model_busy'
        finally:
            release.set()
        assert pending.result(timeout=3).status_code == 200
