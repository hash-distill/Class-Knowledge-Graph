"""One courseware task per process. No model or speech runtime is initialized."""
import argparse
import json
from pathlib import Path

from .common import sha256
from .courseware_service import image_fingerprint, run_courseware_import
from .jobs import JobQueue
from .mutation import LessonMutationLock
from .provider import ReviewError


def run(db_path, registry_path, job_id, token):
    from .lesson_store import LessonStore
    queue = JobQueue(db_path, registry_path)
    lease = None
    failure = None
    try:
        row = queue._check(job_id, token)
        payload = json.loads(row['payload'])
        store = LessonStore(registry_path)
        directory = Path(store.entry(row['lesson_id'])['directory']).resolve()
        if directory != Path(payload['directory']).resolve():
            raise ReviewError('stale_lesson', '课堂产物目录已变化。', 409)
        if payload.get('source_video') and Path(store.entry(row['lesson_id'])['source_video']).resolve()!=Path(payload['source_video']).resolve():
            raise ReviewError('stale_lesson_media','课堂登记的原始媒体已变化，请重新导入。',409)
        lease = LessonMutationLock(directory).acquire()
        path = Path(payload['path']).resolve()
        if not path.is_relative_to(directory / 'courseware' / '_uploads'):
            raise ReviewError('source_upload_changed', '已保存的课件源文件已变化。', 409)

        def check_inputs():
            queue._check(job_id, token)
            if payload.get('source_video') and Path(store.entry(row['lesson_id'])['source_video']).resolve()!=Path(payload['source_video']).resolve():
                raise ReviewError('stale_lesson_media','课堂登记的原始媒体已变化，请重新导入。',409)
            if sha256(path) != payload['upload_sha256']:
                raise ReviewError('source_upload_changed', '已保存的课件源文件已变化。', 409)
            if sha256(directory / 'bundle.json') != payload['expected_version']:
                raise ReviewError('stale_lesson', '排队或导入期间课堂证据已变化，请重新导入。', 409)
            if image_fingerprint(directory, payload['base']['slides']) != payload['source_images']:
                raise ReviewError('stale_lesson_images', '排队或导入期间课堂图像已变化，请重新导入。', 409)
            for media,expected in payload.get('source_media',{}).items():
                file=Path(media)
                if (sha256(file) if file.is_file() else None)!=expected:
                    raise ReviewError('stale_lesson_media','排队或导入期间原始媒体已变化，请重新导入。',409)

        run_courseware_import(path, directory, payload['base'], payload['expected_version'],
                              check=check_inputs,
                              progress=lambda value: queue._progress(job_id, token, value),
                              publish=lambda operation,publication: queue._publish(job_id, token, operation, lease.release,publication=publication))
    except Exception as exc:
        failure = {'error_code': getattr(exc, 'code', 'courseware_import_failed'),
                   'error_detail': (type(exc).__name__ + ': ' + str(exc))[:4000]}
    finally:
        if lease is not None:
            lease.release()
        if failure:
            queue._finish(job_id, token, **failure)


def main():
    parser = argparse.ArgumentParser(description='Execute one privately persisted courseware import.')
    for name in ('db', 'registry', 'job', 'token'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    run(args.db, args.registry, args.job, args.token)


if __name__ == '__main__':
    main()
