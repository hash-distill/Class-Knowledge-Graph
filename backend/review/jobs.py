"""Private SQLite job state and a single local subprocess controller.

Only explicitly selected summary fields leave this module. Uploaded sources,
bundle snapshots, process IDs and exception details remain in the local DB.
"""
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import base64
import hashlib
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import threading
import uuid

from .provider import ReviewError
from .mutation import LessonMutationLock

PUBLIC_FIELDS=('job_id','lesson_id','kind','filename','status','progress','created_at',
               'started_at','completed_at','attempt','retry_of','root_job_id','document_id')
PUBLIC_SELECTION=','.join((*PUBLIC_FIELDS,'cancel_requested','error_code'))


class JobCancelled(ReviewError):
    def __init__(self):
        super().__init__('job_cancelled', 'The task was cancelled. No import result was published.', 409)


def now():
    return datetime.now(timezone.utc).isoformat()


class JobQueue:
    def __init__(self, db_path, registry_path, *, poll_interval=0.1):
        self.db_path = Path(db_path).resolve()
        self.registry_path = Path(registry_path).resolve()
        self.poll_interval = poll_interval
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread = None
        self._controller_lease = None
        self._wal_anchor = None
        self._lifecycle_lock = threading.Lock()
        with self._connection() as db:
            db.execute('PRAGMA journal_mode=WAL')
            db.executescript('''
                CREATE TABLE IF NOT EXISTS jobs (
                    job_id TEXT PRIMARY KEY, lesson_id TEXT NOT NULL, kind TEXT NOT NULL,
                    filename TEXT NOT NULL, status TEXT NOT NULL, progress INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL, started_at TEXT, completed_at TEXT,
                    attempt INTEGER NOT NULL DEFAULT 1, retry_of TEXT, root_job_id TEXT NOT NULL,
                    payload TEXT NOT NULL, claim_token TEXT, worker_pid INTEGER,
                    cancel_requested INTEGER NOT NULL DEFAULT 0,
                    document_id TEXT, error_code TEXT, error_detail TEXT);
                CREATE UNIQUE INDEX IF NOT EXISTS one_pending_lesson ON jobs(lesson_id)
                    WHERE status IN ('queued', 'running');
                CREATE INDEX IF NOT EXISTS jobs_queue_order ON jobs(status,created_at,job_id);
                CREATE INDEX IF NOT EXISTS jobs_recent_order ON jobs(created_at DESC,job_id);
                CREATE TABLE IF NOT EXISTS job_publications (
                    job_id TEXT PRIMARY KEY, directory TEXT NOT NULL,
                    expected_hash TEXT NOT NULL, document_id TEXT NOT NULL,
                    previous_files TEXT NOT NULL);
            ''')

    @contextmanager
    def _connection(self):
        db = sqlite3.connect(self.db_path, timeout=30)
        db.row_factory = sqlite3.Row
        try:
            with db:
                yield db
        finally:
            db.close()

    @staticmethod
    def _summary(row):
        result = {key: row[key] for key in PUBLIC_FIELDS
                  if row[key] is not None}
        result['cancel_requested'] = bool(row['cancel_requested'])
        if row['error_code']:
            result['error'] = {'code': row['error_code'], 'message': 'The courseware task did not complete. Retry or import again.'}
        return result

    def status(self, job_id):
        with self._connection() as db:
            row = db.execute('SELECT '+PUBLIC_SELECTION+' FROM jobs WHERE job_id=?', (job_id,)).fetchone()
        if row is None:
            raise ReviewError('job_not_found', 'The task does not exist.', 404)
        return self._summary(row)

    def list(self, lesson_id=None):
        with self._connection() as db:
            rows = db.execute('SELECT '+PUBLIC_SELECTION+' FROM jobs' + (' WHERE lesson_id=?' if lesson_id else '') +
                              ' ORDER BY created_at DESC, job_id LIMIT 100', (lesson_id,) if lesson_id else ()).fetchall()
        return [self._summary(row) for row in rows]

    def is_busy(self, lesson_id):
        with self._connection() as db:
            return db.execute("SELECT 1 FROM jobs WHERE lesson_id=? AND status IN ('queued','running')",
                              (lesson_id,)).fetchone() is not None

    def enqueue_courseware(self, lesson_id, filename, payload, *, job_id=None):
        job_id = job_id or uuid.uuid4().hex
        try:
            with self._connection() as db:
                db.execute('''INSERT INTO jobs(job_id,lesson_id,kind,filename,status,created_at,root_job_id,payload)
                              VALUES(?,?,'courseware_import',?,'queued',?,?,?)''',
                           (job_id, lesson_id, filename, now(), job_id, json.dumps(payload, ensure_ascii=False)))
        except sqlite3.IntegrityError as exc:
            raise ReviewError('courseware_busy', 'This lesson already has pending courseware. Please try again shortly.', 409) from exc
        self._wake.set()
        return self.status(job_id)

    def cancel(self, job_id):
        too_late=False
        with self._connection() as db:
            db.execute('BEGIN IMMEDIATE')
            row = db.execute('SELECT * FROM jobs WHERE job_id=?', (job_id,)).fetchone()
            if row is None:
                raise ReviewError('job_not_found', 'The task does not exist.', 404)
            if row['status'] not in ('queued', 'running'):
                raise ReviewError('invalid_job_operation', 'Only pending or running tasks can be cancelled.', 409)
            if row['status']=='running':
                recovered=self._recover_publication(db,row)
                if recovered is None:
                    raise ReviewError('job_recovery_pending','The task is confirming publication. Refresh shortly before trying again.',409)
                if recovered is True:too_late=True
            if not too_late:db.execute('''UPDATE jobs SET cancel_requested=1,
                       status=CASE WHEN status='queued' THEN 'cancelled' ELSE status END,
                       completed_at=CASE WHEN status='queued' THEN ? ELSE completed_at END
                       WHERE job_id=?''', (now(), job_id))
        if too_late:
            raise ReviewError('invalid_job_operation','The task result has been published and cannot be cancelled.',409)
        self._wake.set()
        return self.status(job_id)

    def retry(self, job_id):
        new_id = uuid.uuid4().hex
        try:
            with self._connection() as db:
                db.execute('BEGIN IMMEDIATE')
                row = db.execute('SELECT * FROM jobs WHERE job_id=?', (job_id,)).fetchone()
                if row is None:
                    raise ReviewError('job_not_found', 'The task does not exist.', 404)
                if row['status'] not in ('failed', 'cancelled'):
                    raise ReviewError('invalid_job_operation', 'Only failed or cancelled tasks can be retried.', 409)
                db.execute('''INSERT INTO jobs(job_id,lesson_id,kind,filename,status,created_at,
                              attempt,retry_of,root_job_id,payload)
                              VALUES(?,?,?,?,'queued',?,?,?,?,?)''',
                           (new_id, row['lesson_id'], row['kind'], row['filename'], now(),
                            row['attempt'] + 1, job_id, row['root_job_id'], row['payload']))
        except sqlite3.IntegrityError as exc:
            raise ReviewError('courseware_busy', 'This lesson already has pending courseware. Please try again shortly.', 409) from exc
        self._wake.set()
        return self.status(new_id)

    def _claim(self):
        with self._connection() as db:
            db.execute('BEGIN IMMEDIATE')
            # Sort compact keys, not multi-megabyte source snapshots. SQLite may
            # otherwise spill the private payload into an OS temporary file.
            row = db.execute("SELECT * FROM jobs WHERE job_id=(SELECT job_id FROM jobs WHERE status='queued' ORDER BY created_at,job_id LIMIT 1)").fetchone()
            if row is None:
                return None
            token = uuid.uuid4().hex
            db.execute("UPDATE jobs SET status='running',progress=5,started_at=?,claim_token=? WHERE job_id=?",
                       (now(), token, row['job_id']))
            return dict(db.execute('SELECT * FROM jobs WHERE job_id=?', (row['job_id'],)).fetchone())

    @staticmethod
    def _running(db, job_id, token):
        row = db.execute('SELECT * FROM jobs WHERE job_id=?', (job_id,)).fetchone()
        if row is None or row['status'] != 'running' or row['claim_token'] != token:
            raise ReviewError('job_interrupted', 'The task was interrupted. The old process cannot publish results.', 409)
        if row['cancel_requested']:
            raise JobCancelled()
        return row

    def _check(self, job_id, token):
        with self._connection() as db:
            return dict(self._running(db, job_id, token))

    def _progress(self, job_id, token, value):
        with self._connection() as db:
            db.execute('BEGIN IMMEDIATE')
            self._running(db, job_id, token)
            db.execute('UPDATE jobs SET progress=MAX(progress,?) WHERE job_id=?',
                       (max(0, min(99, int(value))), job_id))

    def _finish(self, job_id, token, *, status='failed', error_code=None, error_detail=None):
        with self._connection() as db:
            db.execute('BEGIN IMMEDIATE')
            row = db.execute('SELECT * FROM jobs WHERE job_id=?', (job_id,)).fetchone()
            if row is None or row['status'] != 'running' or row['claim_token'] != token:
                return
            recovered=self._recover_publication(db,row)
            if recovered is None:
                db.execute("UPDATE jobs SET claim_token=NULL,error_code='job_recovery_pending' WHERE job_id=?",(job_id,))
                return
            if recovered:return
            if row['cancel_requested'] and error_code != 'job_interrupted':
                status, error_code = 'cancelled', None
            db.execute('''UPDATE jobs SET status=?,completed_at=?,claim_token=NULL,
                          error_code=?,error_detail=? WHERE job_id=?''',
                       (status, now(), error_code, error_detail, job_id))

    def _recover_publication(self,db,row):
        receipt=db.execute('SELECT * FROM job_publications WHERE job_id=?',(row['job_id'],)).fetchone()
        if receipt is None:return False
        directory=Path(receipt['directory']).resolve()
        expected_directory=Path(json.loads(row['payload'])['directory']).resolve()
        if directory!=expected_directory:raise ReviewError('invalid_publication','The publication record directory does not match.',409)
        try:lease=LessonMutationLock(directory).acquire()
        except ReviewError:return None
        try:return self._reconcile_files(db,row,receipt,directory)
        finally:lease.release()

    def _reconcile_files(self,db,row,receipt,directory):
        bundle=directory/'bundle.json'
        current=hashlib.sha256(bundle.read_bytes()).hexdigest() if bundle.is_file() else None
        if current==receipt['expected_hash']:
            db.execute("UPDATE jobs SET status='complete',progress=100,completed_at=?,document_id=?,claim_token=NULL,error_code=NULL,error_detail=NULL WHERE job_id=?",
                       (now(),receipt['document_id'],row['job_id']))
            complete=True
        else:
            previous=json.loads(receipt['previous_files'])
            original=previous.get('bundle.json')
            original_hash=hashlib.sha256(base64.b64decode(original)).hexdigest() if original is not None else None
            # Never roll back a different bundle published outside this job.
            if current==original_hash:
                for name,encoded in previous.items():
                    if name not in ('bundle.json','report.json','report.md'):
                        raise ReviewError('invalid_publication','Invalid publication record file.',409)
                    target=directory/name
                    if encoded is None:target.unlink(missing_ok=True)
                    else:
                        recovery=target.with_name(target.name+'.recovery')
                        recovery.write_bytes(base64.b64decode(encoded));recovery.replace(target)
            complete=False
        db.execute('DELETE FROM job_publications WHERE job_id=?',(row['job_id'],))
        return complete

    def _publish(self, job_id, token, operation, release, *, publication=None):
        if publication is not None:
            directory=Path(publication['directory']).resolve()
            previous={name:base64.b64encode((directory/name).read_bytes()).decode('ascii')
                      if (directory/name).is_file() else None for name in ('bundle.json','report.json','report.md')}
            # Commit the recovery intent before any authoritative file changes.
            with self._connection() as db:
                db.execute('BEGIN IMMEDIATE');self._running(db,job_id,token)
                db.execute('INSERT OR REPLACE INTO job_publications VALUES(?,?,?,?,?)',
                    (job_id,str(directory),publication['expected_hash'],publication['document_id'],json.dumps(previous)))
        # Cancellation and restart recovery must acquire this same SQLite write
        # transaction. An accepted cancellation therefore cannot race publication.
        with self._connection() as db:
            db.execute('BEGIN IMMEDIATE')
            self._running(db, job_id, token)
            document_id = operation()
            release()
            db.execute('''UPDATE jobs SET status='complete',progress=100,completed_at=?,
                          document_id=?,claim_token=NULL WHERE job_id=?''', (now(), document_id, job_id))
            db.execute('DELETE FROM job_publications WHERE job_id=?',(job_id,))

    def start(self):
        with self._lifecycle_lock:
            if self._thread is not None and self._thread.is_alive():
                return
            try:
                lease = LessonMutationLock(self.db_path.parent / (self.db_path.name + '.controller')).acquire()
            except ReviewError as exc:
                raise ReviewError('job_controller_busy', 'The task controller is already running.', 409) from exc
            anchor = None
            try:
                # Keep the WAL open while short-lived polling and worker
                # connections come and go. On Windows, last-connection cleanup
                # can otherwise race a dying worker's mapped -shm file.
                # Upstream: https://sqlite.org/src/info/7c103a0b7d
                anchor = sqlite3.connect(self.db_path, timeout=30, check_same_thread=False)
                anchor.execute('PRAGMA schema_version').fetchall()
                with self._connection() as db:
                    db.execute('BEGIN IMMEDIATE')
                    for row in db.execute("SELECT * FROM jobs WHERE status='running'").fetchall():
                        recovered=self._recover_publication(db,row)
                        if recovered is None:
                            db.execute("UPDATE jobs SET claim_token=NULL,error_code='job_recovery_pending' WHERE job_id=?",(row['job_id'],))
                        elif not recovered:
                            db.execute("UPDATE jobs SET status=CASE WHEN cancel_requested=1 THEN 'cancelled' ELSE 'failed' END,completed_at=?,claim_token=NULL,error_code='job_interrupted' WHERE job_id=?",(now(),row['job_id']))
                self._controller_lease = lease
                self._wal_anchor = anchor
                self._stop.clear()
                self._thread = threading.Thread(target=self._run, name='courseware-job-controller', daemon=True)
                self._thread.start()
            except Exception:
                if anchor is not None:
                    anchor.close()
                self._wal_anchor = None
                self._controller_lease = None
                self._thread = None
                lease.release()
                raise

    def stop(self):
        with self._lifecycle_lock:
            self._stop.set()
            self._wake.set()
            if self._thread is not None:
                self._thread.join(timeout=35)
                if self._thread.is_alive():
                    raise ReviewError('job_controller_busy', 'The task controller has not stopped yet.', 409)
                self._thread = None
            if self._wal_anchor is not None:
                self._wal_anchor.close()
                self._wal_anchor = None
            if self._controller_lease is not None:
                self._controller_lease.release()
                self._controller_lease = None

    def _run(self):
        while not self._stop.is_set():
            with self._connection() as db:
                db.execute('BEGIN IMMEDIATE')
                for row in db.execute("SELECT * FROM jobs WHERE status='running' AND claim_token IS NULL").fetchall():
                    recovered=self._recover_publication(db,row)
                    if recovered is False:
                        db.execute("UPDATE jobs SET status=CASE WHEN cancel_requested=1 THEN 'cancelled' ELSE 'failed' END,completed_at=?,error_code='job_interrupted' WHERE job_id=?",(now(),row['job_id']))
            claimed = self._claim()
            if claimed is None:
                self._wake.wait(self.poll_interval)
                self._wake.clear()
                continue
            job_id, token = claimed['job_id'], claimed['claim_token']
            try:
                command=([sys.executable,'--run-worker'] if getattr(sys,'frozen',False)
                         else [sys.executable,'-m','review.job_worker'])
                process = subprocess.Popen(
                    command+['--db', str(self.db_path),'--registry', str(self.registry_path), '--job', job_id, '--token', token],
                    cwd=Path(__file__).resolve().parents[1], stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
            except (OSError, subprocess.SubprocessError) as exc:
                self._finish(job_id, token, error_code='worker_start_failed', error_detail=repr(exc))
                continue
            with self._connection() as db:
                db.execute('UPDATE jobs SET worker_pid=? WHERE job_id=? AND claim_token=?',
                           (process.pid, job_id, token))
            while process.poll() is None and not self._stop.is_set():
                self._wake.wait(self.poll_interval)
                self._wake.clear()
            if process.poll() is None:
                # First revoke publication under the SQLite barrier, then stop
                # only the child represented by this controller's Popen handle.
                with self._connection() as db:
                    db.execute('UPDATE jobs SET cancel_requested=1 WHERE job_id=? AND claim_token=?', (job_id, token))
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
                self._finish(job_id, token, error_code='job_interrupted')
            else:
                process.wait()
                self._finish(job_id, token, error_code='worker_exited',
                             error_detail=f'worker exit status {process.returncode}')
