"""Synthetic review fixtures never write human labels into registered lessons."""
from copy import deepcopy
import io
import json
import zipfile

import pytest
from PIL import Image
from fastapi.testclient import TestClient
from test_fastapi_server import Provider, store_fixture
from review.server import create_app
from review.common import atomic_json


def prepared(tmp_path):
    store = store_fixture(tmp_path)
    Image.new('RGB', (100, 60), 'blue').save(tmp_path/'frame.jpg')
    bundle = deepcopy(store.load('test'))
    bundle['lesson']['source_video'] = {'sha256': 'a'*64}
    bundle['activity_analysis'] = {'source_sha256': 'a'*64, 'frames': [{
        'id': 'F1', 'timestamp_sec': 0, 'scene_id': 1, 'width': 100, 'height': 60,
        'image_url': '/api/review/media/test/frame.jpg',
        'participants': [{'track_id': 'P1', 'bbox': [1, 2, 30, 50], 'label': 'read'},
                         {'track_id': 'P2', 'bbox': [35, 2, 60, 50], 'label': 'write'}]}]}
    atomic_json(tmp_path/'bundle.json', bundle)
    return store


BASE = '/api/review/lessons/test/vision-review'


def annotation(**changes):
    return {'id': 'P1', 'origin': 'model', 'status': 'confirmed', 'label': 'read',
            'role': 'student', 'bbox': [1, 2, 30, 50], 'reason': '', **changes}


def save(client, doc, rows):
    return client.post(BASE+'/frames/F1', json={'expected_version': doc['version'],
        'basis_fingerprint': doc['basis_fingerprint'], 'annotations': rows})


def test_review_preserves_raw_data_versions_history_and_restart(tmp_path):
    store = prepared(tmp_path); raw = (tmp_path/'bundle.json').read_bytes()
    with TestClient(create_app(store=store, provider=Provider())) as client:
        doc = client.get(BASE+'/frames/F1')
        assert doc.status_code == 200, 'Visual correction API is missing'
        doc = doc.json()
        assert doc['frame']['participants'][0]['track_id'] == 'P1'
        first = save(client, doc, [annotation(label='write')])
        assert first.status_code == 200
        assert save(client, doc, []).status_code == 409
        second = save(client, first.json(), []).json()
        assert second['version'] == 2 and second['annotations'] == []
        history = client.get(BASE+'/frames/F1/history').json()['history']
        assert history[0]['version'] == 2 and history[1]['annotations'][0]['label'] == 'write'
        restored = client.post(BASE+'/frames/F1/restore', json={'expected_version': 2,
            'basis_fingerprint': doc['basis_fingerprint'], 'restore_version': 1})
        assert restored.status_code == 200 and restored.json()['version'] == 3
    with TestClient(create_app(store=store, provider=Provider())) as client:
        assert client.get(BASE+'/frames/F1').json()['annotations'][0]['label'] == 'write'
    assert (tmp_path/'bundle.json').read_bytes() == raw


def test_changed_image_or_model_disables_old_labels_and_export(tmp_path):
    store = prepared(tmp_path)
    with TestClient(create_app(store=store, provider=Provider())) as client:
        doc = client.get(BASE+'/frames/F1').json()
        save(client, doc, [annotation()])
        Image.new('RGB', (100, 60), 'red').save(tmp_path/'frame.jpg')
        current = client.get(BASE+'/frames/F1').json()
        assert current['source_changed'] and current['annotations'] == []
        assert save(client, doc, [annotation()]).status_code == 409
        assert client.get(BASE+'/export').status_code == 409
        assert client.get(BASE+'/frames/F1/history').json()['history'][0]['annotations']


@pytest.mark.parametrize('changes', [
    {'bbox': [-1, 0, 5, 5]}, {'bbox': [0, 0, 101, 5]}, {'bbox': [5, 0, 5, 5]},
    {'bbox': [True, 0, 5, 5]}, {'id': 'fabricated'}, {'label': 'attention'},
    {'status': 'ignored', 'reason': ''}, {'role': 'identity-A'},
    {'origin': 'manual', 'id': 'P1'},
])
def test_invalid_annotations_are_rejected(tmp_path, changes):
    with TestClient(create_app(store=prepared(tmp_path), provider=Provider())) as client:
        doc = client.get(BASE+'/frames/F1').json()
        assert save(client, doc, [annotation(**changes)]).status_code == 400
        assert client.get(BASE+'/frames/F1').json()['version'] == 0


def test_zip_has_only_confirmed_student_classification_crops(tmp_path):
    with TestClient(create_app(store=prepared(tmp_path), provider=Provider())) as client:
        doc = client.get(BASE+'/frames/F1').json()
        assert client.get(BASE+'/export').status_code == 409
        rows = [annotation(), annotation(id='P2', status='ignored', reason='镜面重复'),
                annotation(id='manual-1', origin='manual', role='unknown', label='write'),
                annotation(id='manual-2', origin='manual', label='unknown')]
        assert save(client, doc, rows).status_code == 200
        response = client.get(BASE+'/export')
        assert response.status_code == 200
        with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
            manifest = json.loads(archive.read('manifest.json'))
            assert manifest['classification_sample_count'] == 1
            assert manifest['partial_annotations'] is True
            assert len([p for p in archive.namelist() if p.startswith('crops/')]) == 1
            assert len(manifest['frames'][0]['annotations']) == 4
            assert manifest['source_video_sha256'] == 'a'*64
            assert manifest['frames'][0]['review_version'] == 1


def test_duplicate_rows_cross_origin_and_cross_lesson_are_rejected(tmp_path):
    with TestClient(create_app(store=prepared(tmp_path), provider=Provider())) as client:
        doc = client.get(BASE+'/frames/F1').json()
        assert save(client, doc, [annotation(), annotation()]).status_code == 400
        assert client.get('/api/review/lessons/missing/vision-review/frames/F1').status_code == 404
        assert client.post(BASE+'/frames/F1', json={'expected_version': 0,
            'basis_fingerprint': doc['basis_fingerprint'], 'annotations': []},
            headers={'Origin': 'https://external.example'}).status_code == 403


def test_removed_frame_is_archived_without_blocking_current_sample_export(tmp_path):
    store = prepared(tmp_path)
    with TestClient(create_app(store=store, provider=Provider())) as client:
        doc = client.get(BASE+'/frames/F1').json()
        assert save(client, doc, [annotation()]).status_code == 200
        bundle = deepcopy(store.load('test'))
        bundle['activity_analysis']['frames'][0]['id'] = 'F2'
        atomic_json(tmp_path/'bundle.json', bundle)
        fresh = client.get(BASE+'/frames/F2').json()
        result = client.post(BASE+'/frames/F2', json={'expected_version': 0,
            'basis_fingerprint': fresh['basis_fingerprint'], 'annotations': [annotation()]})
        assert result.status_code == 200
        response = client.get(BASE+'/export')
        assert response.status_code == 200
        with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
            manifest = json.loads(archive.read('manifest.json'))
            assert manifest['archived_missing_frames'][0]['frame_id'] == 'F1'
            assert len(manifest['frames']) == 1 and manifest['frames'][0]['frame_id'] == 'F2'
        assert client.get(BASE+'/frames/F1/history').json()['history'][0]['version'] == 1
