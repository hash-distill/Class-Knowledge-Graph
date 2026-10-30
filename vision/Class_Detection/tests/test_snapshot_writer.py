import json
import sys
from pathlib import Path
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.snapshot_writer import SnapshotWriter


def test_segments_bound_utf8_bytes_and_keep_complete_order(tmp_path):
    writer=SnapshotWriter(tmp_path/'snapshots.json',max_bytes=210)
    values=[{'timestamp_sec':i,'student_states':[{'label':'书写'}]} for i in range(8)]
    for value in values: writer.append(value)
    writer.finish()
    restored=[]
    assert len(writer.files)>1
    for name in writer.files:
        path=tmp_path/name
        assert path.stat().st_size<=210
        restored.extend(json.loads(path.read_text(encoding='utf-8')))
    assert restored==values
    assert not list(tmp_path.glob('*.partial'))


def test_frame_and_state_bounds_and_abort(tmp_path):
    writer=SnapshotWriter(tmp_path/'a.json',max_frames=2,max_states=2)
    for i in range(3): writer.append({'student_states':[{}]})
    writer.abort()
    assert writer.files==['a.json']
    assert len(json.loads((tmp_path/'a.json').read_text()))==2
    assert not (tmp_path/'a_002.json').exists()
    assert (tmp_path/'a_002.json.partial').exists()


def test_reject_oversized_and_non_finite_records(tmp_path):
    writer=SnapshotWriter(tmp_path/'a.json',max_bytes=50)
    with pytest.raises(ValueError): writer.append({'value':'x'*60})
    with pytest.raises(ValueError): writer.append({'value':float('nan')})
    assert not list(tmp_path.iterdir())
