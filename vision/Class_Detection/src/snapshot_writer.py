"""Bounded, atomic JSON-array segments that fit the dashboard import limits."""
import json
from pathlib import Path


class SnapshotWriter:
    def __init__(self, path, max_bytes=16*1024*1024, max_frames=15000, max_states=200000):
        self.path = Path(path)
        self.max_bytes, self.max_frames, self.max_states = max_bytes, max_frames, max_states
        self.files = []
        self.handle = None
        self.count = self.states = self.size = 0
        self.total = 0

    def _open(self):
        number = len(self.files)+1
        self.target = self.path if number == 1 else self.path.with_name(f'{self.path.stem}_{number:03d}.json')
        self.partial = self.target.with_suffix('.json.partial')
        self.handle = self.partial.open('xb')
        self.handle.write(b'[\n')
        self.count = self.states = 0
        self.size = 2

    def append(self, snapshot):
        payload = json.dumps(snapshot, ensure_ascii=False, allow_nan=False, separators=(',', ':')).encode('utf-8')
        states = len(snapshot.get('student_states', []))
        if len(payload)+5 > self.max_bytes or states > self.max_states:
            raise ValueError('One snapshot exceeds the dashboard segment limit')
        if self.handle and self.count and (self.size+2+len(payload)+3 > self.max_bytes or
                self.count >= self.max_frames or self.states+states > self.max_states):
            self._finish_segment()
        if self.handle is None:
            self._open()
        if self.count:
            self.handle.write(b',\n'); self.size += 2
        self.handle.write(payload)
        self.size += len(payload); self.count += 1; self.states += states; self.total += 1

    def _finish_segment(self):
        if self.handle is None:
            return
        self.handle.write(b'\n]\n')
        self.handle.close()
        self.handle = None
        self.partial.replace(self.target)
        self.files.append(self.target.name)

    def finish(self):
        self._finish_segment()

    def abort(self):
        # Retain incomplete bytes for diagnostics; never rename them as successful JSON.
        if self.handle:
            self.handle.close()
            self.handle = None
