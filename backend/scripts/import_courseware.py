"""Import a real standalone deck into an already registered lesson."""
import argparse
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from review.server import LessonStore
from review.courseware_service import CoursewareService, MAX_UPLOAD


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lesson-id', required=True)
    parser.add_argument('--file', required=True, type=Path)
    args = parser.parse_args()
    if not args.file.is_file() or not 0 < args.file.stat().st_size <= MAX_UPLOAD:
        parser.error('File must exist and be 1 byte–64 MiB')
    service = CoursewareService(LessonStore())
    job = service.submit(args.lesson_id, args.file.name, args.file.read_bytes())
    print('Import started:', job['job_id'], flush=True)
    while job['status'] == 'running':
        time.sleep(1)
        job = service.status(job['job_id'])
    if job['status'] != 'complete':
        raise RuntimeError(job['error']['message'])
    bundle = service.store.load(args.lesson_id)
    print('Complete:', job['document_id'],
          'pages=', len(bundle['courseware']['pages']),
          'matched=', len(bundle['courseware_evidence']),
          'revisions=', len(bundle['transcript_revisions']), flush=True)


if __name__ == '__main__':
    main()
