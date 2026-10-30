"""Windows WAL lifetime regression using real SQLite handles and a mapped index."""
import mmap
import os
from pathlib import Path
import sqlite3
import sys
import threading

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from review.jobs import JobQueue


@pytest.mark.skipif(os.name != 'nt', reason='Windows prevents truncating a mapped WAL index')
def test_controller_keeps_mapped_wal_readable_and_releases_handles_on_restart(tmp_path, monkeypatch):
    """Model a dead worker's lingering SHM mapping without killing a process.

    An idle controller must retain a real SQLite/DMS owner while the mapping
    remains. An orphaned mapping at fresh application startup still requires
    SQLite's upstream fix: https://sqlite.org/src/info/7c103a0b7d.
    """
    database = tmp_path / 'jobs.sqlite3'
    registry = tmp_path / 'registry.json'
    registry.write_text('{}', encoding='utf-8')
    queue = JobQueue(database, registry, poll_interval=60)
    idle = threading.Event()
    original_wait = queue._wake.wait

    def observe_idle(timeout=None):
        # Synchronize after real controller queries have closed, without sleeps
        # or replacing any persistence, locking, or lifecycle operations.
        idle.set()
        return original_wait(timeout)

    monkeypatch.setattr(queue._wake, 'wait', observe_idle)
    for cycle in range(2):
        idle.clear()
        seed = sqlite3.connect(database)
        try:
            seed.execute('SELECT job_id FROM jobs LIMIT 1').fetchall()
            with Path(str(database) + '-shm').open('r+b') as stream:
                with mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_WRITE):
                    queue.start()
                    assert idle.wait(5), 'The empty controller did not reach its idle wait.'
                    seed.close()
                    # Without the controller anchor, this sole mapped-but-
                    # unlocked SHM causes SQLITE_IOERR_TRUNCATE on affected SQLite.
                    assert queue.list() == []
        finally:
            seed.close()
            queue.stop()

        # Windows rejects this rename if the controller retained an open DB
        # handle. Starting the next cycle also requires its lease to be free.
        parked = tmp_path / f'jobs-stopped-{cycle}.sqlite3'
        database.rename(parked)
        parked.rename(database)
