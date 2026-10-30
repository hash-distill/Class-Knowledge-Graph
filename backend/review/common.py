from __future__ import annotations
import hashlib
import json
import math
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix+'.partial')
    partial.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2), encoding='utf-8')
    partial.replace(path)


def sha256(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def valid_id(value):
    if not isinstance(value, str) or not re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9_-]{0,79}', value):
        raise ValueError('Invalid lesson ID')
    return value


def validate_segments(rows, duration, kind):
    result, used = [], set()
    for row in rows:
        item = dict(row)
        identifier = str(item.get('id', ''))
        if not re.fullmatch(r'[A-Za-z][A-Za-z0-9_-]{0,60}', identifier) or identifier in used:
            raise ValueError(f'{kind}: invalid or duplicate evidence ID')
        used.add(identifier)
        start, end = item.get('start_sec'), item.get('end_sec')
        if any(isinstance(v, bool) or not isinstance(v, (int,float)) or not math.isfinite(v) for v in [start,end]):
            raise ValueError(f'{identifier}: invalid time')
        if not (0 <= start < end <= duration+0.5):
            raise ValueError(f'{identifier}: time outside lesson duration')
        if not isinstance(item.get('text'), str):
            raise ValueError(f'{identifier}: text must be a string')
        item['kind'] = kind
        result.append(item)
    result.sort(key=lambda r:(r['start_sec'],r['end_sec']))
    return result


def union_duration(intervals):
    total = 0.; previous = None
    for start,end in sorted(intervals):
        if previous is None: previous=[start,end]
        elif start <= previous[1]: previous[1]=max(previous[1],end)
        else: total += previous[1]-previous[0]; previous=[start,end]
    return total+(previous[1]-previous[0] if previous else 0.)
