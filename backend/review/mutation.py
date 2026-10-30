"""OS-backed per-lesson write lease shared by API, CLI and rebuilds."""
from pathlib import Path
import os
from .provider import ReviewError


class LessonMutationLock:
    def __init__(self, directory):
        self.path = Path(directory) / '.lesson-write.lock'
        self.handle = None

    def acquire(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle = self.path.open('a+b')
        try:
            handle.seek(0, 2)
            if handle.tell() == 0:
                handle.write(b'0'); handle.flush()
            handle.seek(0)
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            handle.close()
            raise ReviewError('lesson_busy', 'This lesson is being imported or rebuilt. Please try again shortly.', 409) from exc
        self.handle = handle
        return self

    def release(self):
        handle, self.handle = self.handle, None
        if handle is not None:
            handle.seek(0)
            try:
                if os.name == 'nt':
                    import msvcrt
                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            finally:
                handle.close()

    def __enter__(self):
        return self.acquire()

    def __exit__(self, *args):
        self.release()
