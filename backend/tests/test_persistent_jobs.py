"""Persistent import jobs use synthetic lessons and real, text-only PDF bytes."""
import importlib.util
import json
import os
import sqlite3
from pathlib import Path
import sys
import time

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from review.common import atomic_json
from review.courseware_service import CoursewareService
from review.provider import ReviewError
from review.server import LessonStore


def lesson(tmp_path):
    bundle = {'schema_version': '3.0', 'data_source': 'lesson_review',
              'lesson': {'id': 'test', 'title': 'Synthetic persistent import', 'duration_sec': 20,
                         'status': 'complete', 'source_video': {'filename': 'test.mp4', 'sha256': 'a' * 64},
                         'provenance': {}},
              'transcript': [{'id': 'T0001', 'text': 'Synthetic normal reference lesson', 'start_sec': 1, 'end_sec': 5}],
              'slides': [{'id': 'S0001', 'text': 'Synthetic normal reference lesson', 'start_sec': 0, 'end_sec': 10}],
              'alignment': [], 'graph': {'nodes': [], 'edges': []}, 'snapshots': [],
              'report': {'title': 'Synthetic', 'summary': 'Fixture only', 'sections': [], 'limitations': []}}
    atomic_json(tmp_path / 'bundle.json', bundle)
    registry = tmp_path / 'registry.json'
    atomic_json(registry, {'test': {'directory': str(tmp_path), 'source_video': str(tmp_path / 'test.mp4')}})
    return bundle, LessonStore(registry), registry


def make_queue(tmp_path, registry):
    assert importlib.util.find_spec('review.jobs') is not None, 'persistent job queue is not implemented'
    from review.jobs import JobQueue
    return JobQueue(tmp_path / 'jobs.sqlite3', registry, poll_interval=0.02)


def test_queued_job_survives_recreation_and_blocks_mutations(tmp_path):
    _, store, registry = lesson(tmp_path)
    queue = make_queue(tmp_path, registry)
    service = CoursewareService(store, job_queue=queue)
    job = service.submit('test', 'synthetic.pdf', b'%PDF-synthetic')
    assert job['status'] == 'queued' and job['progress'] == 0
    other = make_queue(tmp_path, registry)
    assert other.status(job['job_id']) == queue.status(job['job_id'])
    assert other.list('test') == [queue.status(job['job_id'])]
    assert other.is_busy('test')
    for operation in (lambda: service.submit('test', 'second.pdf', b'x'),
                      lambda: service.confirm('test', {}), lambda: service.remove('test')):
        with pytest.raises(ReviewError) as error:
            operation()
        assert error.value.status == 409
    public = json.dumps(other.status(job['job_id']), ensure_ascii=False)
    assert str(tmp_path) not in public and 'expected_version' not in public and 'payload' not in public


def test_cancel_retry_uses_distinct_ids_and_keeps_provenance(tmp_path):
    _, store, registry = lesson(tmp_path)
    queue = make_queue(tmp_path, registry)
    job = CoursewareService(store, job_queue=queue).submit('test', 'synthetic.pdf', b'%PDF-synthetic')
    assert queue.cancel(job['job_id'])['status'] == 'cancelled'
    retry = queue.retry(job['job_id'])
    assert retry['job_id'] != job['job_id'] and retry['status'] == 'queued'
    assert retry['retry_of'] == job['job_id'] and retry['root_job_id'] == job['job_id'] and retry['attempt'] == 2
    assert queue.status(job['job_id'])['status'] == 'cancelled'
    with pytest.raises(ReviewError):
        queue.retry(retry['job_id'])
    with pytest.raises(ReviewError) as error:
        queue.status('missing')
    assert error.value.status == 404
    queue.cancel(retry['job_id'])
    CoursewareService(store, job_queue=queue).remove('test')


def test_atomic_claim_monotonic_progress_and_restart_recovery(tmp_path):
    base, store, registry = lesson(tmp_path)
    queue = make_queue(tmp_path, registry)
    job = CoursewareService(store, job_queue=queue).submit('test', 'synthetic.pdf', b'%PDF-synthetic')
    claimed = queue._claim()
    assert claimed['job_id'] == job['job_id'] and claimed['claim_token']
    other = make_queue(tmp_path, registry)
    assert other._claim() is None
    queue._progress(job['job_id'], claimed['claim_token'], 70)
    queue._progress(job['job_id'], claimed['claim_token'], 20)
    assert queue.status(job['job_id'])['progress'] == 70
    other.start()
    try:
        state = other.status(job['job_id'])
        assert state['status'] == 'failed' and state['error']['code'] == 'job_interrupted'
        assert store.load('test') == base
        with pytest.raises(ReviewError):
            queue._check(job['job_id'], claimed['claim_token'])
        with pytest.raises(ReviewError):
            make_queue(tmp_path, registry).start()
    finally:
        other.stop()


def pdf_bytes():
    import pymupdf
    with pymupdf.open() as pdf:
        page = pdf.new_page()
        page.insert_text((60, 80), 'Synthetic normal reference lesson')
        return pdf.tobytes()


def wait_done(queue, job_id, timeout=30):
    deadline = time.monotonic() + timeout
    states = []
    while time.monotonic() < deadline:
        state = queue.status(job_id)
        states.append(state['progress'])
        if state['status'] not in ('queued', 'running'):
            assert states == sorted(states)
            return state
        time.sleep(.02)
    pytest.fail(f'job did not finish: {state}')


def test_original_video_replacement_invalidates_queued_import(tmp_path):
    _,store,registry=lesson(tmp_path)
    video=tmp_path/'test.mp4';video.write_bytes(b'original synthetic media bytes')
    queue=make_queue(tmp_path,registry)
    job=CoursewareService(store,job_queue=queue).submit('test','test.pdf',pdf_bytes())
    video.write_bytes(b'different original media')
    queue.start()
    try:
        state=wait_done(queue,job['job_id'])
        assert state['status']=='failed' and state['error']['code']=='stale_lesson_media'
        assert 'courseware' not in store.load('test')
    finally:queue.stop()


def test_registry_media_identity_change_invalidates_import(tmp_path):
    _,store,registry=lesson(tmp_path)
    (tmp_path/'test.mp4').write_bytes(b'original')
    queue=make_queue(tmp_path,registry)
    job=CoursewareService(store,job_queue=queue).submit('test','test.pdf',pdf_bytes())
    changed=tmp_path/'other.mp4';changed.write_bytes(b'other')
    entry=json.loads(registry.read_text(encoding='utf-8'))
    entry['test']['source_video']=str(changed);atomic_json(registry,entry)
    queue.start()
    try:
        state=wait_done(queue,job['job_id'])
        assert state['status']=='failed' and state['error']['code']=='stale_lesson_media'
    finally:queue.stop()


def test_receipt_recovery_waits_for_external_lesson_mutation_lock(tmp_path):
    from review.mutation import LessonMutationLock
    _,store,registry=lesson(tmp_path)
    (tmp_path/'report.md').write_bytes(b'old report')
    queue=make_queue(tmp_path,registry)
    job=CoursewareService(store,job_queue=queue).submit('test','test.pdf',pdf_bytes())
    claim=queue._claim()
    def interrupted():
        (tmp_path/'report.md').write_bytes(b'partially published report')
        raise SystemExit(91)
    with pytest.raises(SystemExit):
        queue._publish(job['job_id'],claim['claim_token'],interrupted,lambda:None,
            publication={'directory':str(tmp_path),'expected_hash':'f'*64,'document_id':'D1'})
    lease=LessonMutationLock(tmp_path).acquire()
    try:
        queue._finish(job['job_id'],claim['claim_token'],error_code='worker_exited')
        assert (tmp_path/'report.md').read_bytes()==b'partially published report'
        assert queue.status(job['job_id'])['status']=='running'
        queue.start()
        assert (tmp_path/'report.md').read_bytes()==b'partially published report'
    finally:lease.release()
    try:
        assert wait_done(queue,job['job_id'])['status']=='failed'
        assert (tmp_path/'report.md').read_bytes()==b'old report'
    finally:queue.stop()


def test_cancel_reconciles_already_published_receipt_before_accepting(tmp_path):
    import hashlib
    base,store,registry=lesson(tmp_path)
    queue=make_queue(tmp_path,registry)
    job=CoursewareService(store,job_queue=queue).submit('test','test.pdf',pdf_bytes())
    claim=queue._claim();published={**base,'courseware':{'document_id':'D1'}}
    serialized=json.dumps(published,ensure_ascii=False,allow_nan=False,indent=2).replace('\n',os.linesep).encode()
    def operation():atomic_json(tmp_path/'bundle.json',published);return 'D1'
    def crash():raise SystemExit(91)
    with pytest.raises(SystemExit):
        queue._publish(job['job_id'],claim['claim_token'],operation,crash,
            publication={'directory':str(tmp_path),'expected_hash':hashlib.sha256(serialized).hexdigest(),'document_id':'D1'})
    with pytest.raises(ReviewError):queue.cancel(job['job_id'])
    assert queue.status(job['job_id'])['status']=='complete'
    assert queue.status(job['job_id'])['cancel_requested'] is False


def test_cancel_waits_when_crash_receipt_recovery_is_locked(tmp_path):
    import hashlib
    from review.mutation import LessonMutationLock
    base,store,registry=lesson(tmp_path);queue=make_queue(tmp_path,registry)
    job=CoursewareService(store,job_queue=queue).submit('test','test.pdf',pdf_bytes())
    claim=queue._claim();published={**base,'courseware':{'document_id':'D1'}}
    encoded=json.dumps(published,ensure_ascii=False,allow_nan=False,indent=2).replace('\n',os.linesep).encode()
    def operation():atomic_json(tmp_path/'bundle.json',published);return 'D1'
    def crash():raise SystemExit(91)
    with pytest.raises(SystemExit):
        queue._publish(job['job_id'],claim['claim_token'],operation,crash,
            publication={'directory':str(tmp_path),'expected_hash':hashlib.sha256(encoded).hexdigest(),'document_id':'D1'})
    lease=LessonMutationLock(tmp_path).acquire()
    try:
        with pytest.raises(ReviewError) as error:queue.cancel(job['job_id'])
        assert error.value.status==409
        assert queue.status(job['job_id'])['cancel_requested'] is False
    finally:lease.release()
    queue.start()
    try:assert queue.status(job['job_id'])['status']=='complete'
    finally:queue.stop()


def test_large_private_payload_can_be_listed_and_claimed_without_temp_sort_failure(tmp_path):
    _,store,registry=lesson(tmp_path);queue=make_queue(tmp_path,registry)
    job=CoursewareService(store,job_queue=queue).submit('test','test.pdf',pdf_bytes())
    with sqlite3.connect(queue.db_path) as db:
        payload=json.loads(db.execute('SELECT payload FROM jobs WHERE job_id=?',(job['job_id'],)).fetchone()[0])
        payload['large_source_fixture']='x'*8_000_000
        db.execute('UPDATE jobs SET payload=? WHERE job_id=?',(json.dumps(payload),job['job_id']))
    try:
        listed=queue.list('test')
        claimed=queue._claim()
    except sqlite3.OperationalError as exc:
        pytest.fail(f'Large real-source job metadata must not require an unavailable temp sort: {exc}')
    assert listed[0]['job_id']==job['job_id'] and 'payload' not in listed[0]
    assert claimed['job_id']==job['job_id']


def test_worker_crash_after_publication_is_reconciled_as_complete(tmp_path,monkeypatch):
    _,store,registry=lesson(tmp_path)
    guard=tmp_path/'crash_guard';guard.mkdir()
    (guard/'sitecustomize.py').write_text(
        'import os\nfrom review.jobs import JobQueue\noriginal=JobQueue._publish\n'
        'def crash(self,job,token,operation,release,**kw):\n'
        ' return original(self,job,token,operation,lambda:os._exit(91),**kw)\n'
        'JobQueue._publish=crash\n',encoding='utf-8')
    monkeypatch.setenv('PYTHONPATH',str(guard)+os.pathsep+str(Path(__file__).resolve().parents[1]))
    queue=make_queue(tmp_path,registry)
    job=CoursewareService(store,job_queue=queue).submit('test','test.pdf',pdf_bytes())
    queue.start()
    try:
        state=wait_done(queue,job['job_id'])
        assert state['status']=='complete' and state['progress']==100
        assert store.load('test')['courseware']['document_id']==state['document_id']
    finally:queue.stop()


def test_worker_crash_before_bundle_commit_restores_reports_and_allows_retry(tmp_path,monkeypatch):
    _,store,registry=lesson(tmp_path)
    (tmp_path/'report.md').write_bytes(b'original report')
    before=(tmp_path/'bundle.json').read_bytes()
    guard=tmp_path/'crash_guard';guard.mkdir()
    (guard/'sitecustomize.py').write_text(
        'import os\nfrom pathlib import Path\nfrom review import courseware_service as c\noriginal=c.atomic_json\n'
        'def crash(path,value):\n'
        ' if Path(path).name=="bundle.json":os._exit(92)\n'
        ' return original(path,value)\n'
        'c.atomic_json=crash\n',encoding='utf-8')
    monkeypatch.setenv('PYTHONPATH',str(guard)+os.pathsep+str(Path(__file__).resolve().parents[1]))
    queue=make_queue(tmp_path,registry)
    job=CoursewareService(store,job_queue=queue).submit('test','test.pdf',pdf_bytes())
    queue.start()
    try:
        state=wait_done(queue,job['job_id'])
        assert state['status']=='failed'
        assert (tmp_path/'report.md').read_bytes()==b'original report'
        assert (tmp_path/'bundle.json').read_bytes()==before
        monkeypatch.delenv('PYTHONPATH')
        retried=queue.retry(job['job_id'])
        assert wait_done(queue,retried['job_id'])['status']=='complete'
    finally:queue.stop()


def test_real_pdf_import_runs_in_child_without_model_import_and_reuses_cache(tmp_path, monkeypatch):
    base, store, registry = lesson(tmp_path)
    guard = tmp_path / 'import_guard'
    guard.mkdir()
    (guard / 'sitecustomize.py').write_text(
        'import sys\nclass Guard:\n def find_spec(self, fullname, *args):\n'
        '  if fullname.split(".")[0] in {"torch","transformers","whisper","faster_whisper"}'
        ' or fullname in {"review.model_runtime","review.model_upgrade"}:\n'
        '   raise AssertionError("model imported by courseware worker")\n'
        'sys.meta_path.insert(0,Guard())\n', encoding='utf-8')
    monkeypatch.setenv('PYTHONPATH', str(guard) + os.pathsep + os.environ.get('PYTHONPATH', ''))
    queue = make_queue(tmp_path, registry)
    source = pdf_bytes()
    queue.start()
    try:
        service = CoursewareService(store, job_queue=queue)
        job = service.submit('test', 'synthetic.pdf', source)
        state = wait_done(queue, job['job_id'])
        assert state['status'] == 'complete' and state['progress'] == 100, state
        imported = store.load('test')
        assert imported['transcript'] == base['transcript']
        assert imported['slides'] == base['slides']
        assert imported['courseware']['pages'][0]['text_source'] == 'pdf_native'
        with sqlite3.connect(queue.db_path) as db:
            worker_pid = db.execute('SELECT worker_pid FROM jobs WHERE job_id=?', (job['job_id'],)).fetchone()[0]
        assert worker_pid and worker_pid != os.getpid()
        service.remove('test')
        second = service.submit('test', 'synthetic.pdf', source)
        assert wait_done(queue, second['job_id'])['status'] == 'complete'
        assert store.load('test')['courseware']['alignment_cache_reused'] is True
    finally:
        queue.stop()


def test_queued_image_change_invalidates_original_input_snapshot(tmp_path):
    base, store, registry = lesson(tmp_path)
    base['slides'][0]['image_path'] = 'observed.png'
    (tmp_path / 'observed.png').write_bytes(b'original synthetic pixels')
    atomic_json(tmp_path / 'bundle.json', base)
    queue = make_queue(tmp_path, registry)
    job = CoursewareService(store, job_queue=queue).submit('test', 'synthetic.pdf', pdf_bytes())
    (tmp_path / 'observed.png').write_bytes(b'changed synthetic pixels')
    queue.start()
    try:
        state = wait_done(queue, job['job_id'])
        assert state['status'] == 'failed' and state['error']['code'] == 'stale_lesson_images'
        assert store.load('test') == base
        assert not (tmp_path / 'bundle.base.json').exists()
    finally:
        queue.stop()


def test_accepted_cancellation_cannot_publish_and_completion_releases_lease_first(tmp_path):
    from review.mutation import LessonMutationLock
    _, store, registry = lesson(tmp_path)
    queue = make_queue(tmp_path, registry)
    job = CoursewareService(store, job_queue=queue).submit('test', 'synthetic.pdf', b'x')
    claim = queue._claim()
    lease = LessonMutationLock(tmp_path).acquire()
    queue.cancel(job['job_id'])
    published = []
    try:
        with pytest.raises(ReviewError):
            queue._publish(job['job_id'], claim['claim_token'], lambda: published.append(True), lease.release)
    finally:
        lease.release()
    queue._finish(job['job_id'], claim['claim_token'])
    assert published == [] and queue.status(job['job_id'])['status'] == 'cancelled'
    retry = queue.retry(job['job_id'])
    claim = queue._claim()
    lease = LessonMutationLock(tmp_path).acquire()
    released = []
    def release():
        assert queue.status(retry['job_id'])['status'] == 'running'
        lease.release()
        with LessonMutationLock(tmp_path):
            pass
        released.append(True)
    queue._publish(retry['job_id'], claim['claim_token'], lambda: 'synthetic-document', release)
    assert released == [True] and queue.status(retry['job_id'])['status'] == 'complete'


def test_stop_interrupts_only_owned_child_and_releases_lesson_lease(tmp_path):
    from review.mutation import LessonMutationLock
    base, store, registry = lesson(tmp_path)
    queue = make_queue(tmp_path, registry)
    job = CoursewareService(store, job_queue=queue).submit('test', 'synthetic.pdf', pdf_bytes())
    queue.start()
    deadline = time.monotonic() + 10
    while queue.status(job['job_id'])['status'] == 'queued' and time.monotonic() < deadline:
        time.sleep(.005)
    queue.stop()
    state = queue.status(job['job_id'])
    assert state['status'] == 'failed' and state['error']['code'] == 'job_interrupted'
    assert store.load('test') == base
    with LessonMutationLock(tmp_path):
        pass
    replacement = make_queue(tmp_path, registry)
    retry = replacement.retry(job['job_id'])
    replacement.start()
    try:
        assert wait_done(replacement, retry['job_id'])['status'] == 'complete'
    finally:
        replacement.stop()


def test_publication_failure_restores_previous_reports_and_bundle(tmp_path, monkeypatch):
    import review.courseware_service as module
    base, _, _ = lesson(tmp_path)
    module.persist_bundle(tmp_path, base)
    original = {name: (tmp_path / name).read_bytes() for name in ('report.json', 'report.md', 'bundle.json')}
    updated = json.loads(json.dumps(base))
    updated['report']['summary'] = 'New derived report that must not leak on failure'
    real_atomic = module.atomic_json
    def fail_bundle(path, value):
        if Path(path).name == 'bundle.json':
            raise OSError('synthetic disk write failure')
        return real_atomic(path, value)
    monkeypatch.setattr(module, 'atomic_json', fail_bundle)
    with pytest.raises(OSError):
        module.persist_bundle(tmp_path, updated)
    assert {name: (tmp_path / name).read_bytes() for name in original} == original


@pytest.mark.parametrize('changed_input,expected_code', [('image', 'stale_lesson_images'),
                                                       ('upload', 'source_upload_changed')])
def test_input_changes_during_render_cannot_become_new_frozen_source(tmp_path, monkeypatch,
                                                                   changed_input, expected_code):
    import review.courseware_documents as documents
    from review.job_worker import run
    base, store, registry = lesson(tmp_path)
    base['slides'][0]['image_path'] = 'observed.png'
    image = tmp_path / 'observed.png'
    image.write_bytes(b'original synthetic observed image')
    atomic_json(tmp_path / 'bundle.json', base)
    queue = make_queue(tmp_path, registry)
    job = CoursewareService(store, job_queue=queue).submit('test', 'synthetic.pdf', pdf_bytes())
    claimed = queue._claim()
    actual_import = documents.import_document
    def changed_during_render(path, *args):
        document = actual_import(path, *args)
        (image if changed_input == 'image' else Path(path)).write_bytes(b'changed frozen source')
        return document
    monkeypatch.setattr(documents, 'import_document', changed_during_render)
    run(queue.db_path, registry, job['job_id'], claimed['claim_token'])
    state = queue.status(job['job_id'])
    assert state['status'] == 'failed' and state['error']['code'] == expected_code
    assert store.load('test') == base and not (tmp_path / 'bundle.base.json').exists()


def test_running_cancel_and_failed_retry_never_publish_or_leak_lease(tmp_path):
    from review.mutation import LessonMutationLock
    base, store, registry = lesson(tmp_path)
    queue = make_queue(tmp_path, registry)
    service = CoursewareService(store, job_queue=queue)
    job = service.submit('test', 'synthetic.pdf', pdf_bytes())
    queue.start()
    try:
        deadline = time.monotonic() + 10
        while queue.status(job['job_id'])['status'] == 'queued' and time.monotonic() < deadline:
            time.sleep(.005)
        assert queue.status(job['job_id'])['status'] == 'running'
        queue.cancel(job['job_id'])
        assert wait_done(queue, job['job_id'])['status'] == 'cancelled'
        assert store.load('test') == base
        with LessonMutationLock(tmp_path):
            pass
        invalid = service.submit('test', 'invalid.pdf', b'%PDF-invalid')
        assert wait_done(queue, invalid['job_id'])['status'] == 'failed'
        retried = queue.retry(invalid['job_id'])
        assert wait_done(queue, retried['job_id'])['status'] == 'failed'
        assert store.load('test') == base
        assert not (tmp_path / 'bundle.base.json').exists()
        with LessonMutationLock(tmp_path):
            pass
    finally:
        queue.stop()
