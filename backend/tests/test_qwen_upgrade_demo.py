from copy import deepcopy
import wave

import pytest

from review.common import atomic_json, read_json, sha256
from scripts.build_qwen_upgrade_demo import prepare


@pytest.fixture
def sources(tmp_path):
    parent, directory = tmp_path/'old', tmp_path/'new'
    parent.mkdir()
    with wave.open(str(parent/'audio.wav'), 'wb') as wav:
        wav.setparams((1, 2, 16000, 1600, 'NONE', 'not compressed'))
        wav.writeframes(b'\x00\x00'*1600)
    source = {'sha256': 'source-video', 'audio_duration_sec': .1}
    old = {'status': 'complete', 'segments': [{'id': 'T1', 'text': '旧识别', 'start_sec': 0, 'end_sec': .1}],
           'provenance': {'source_video': source}}
    atomic_json(parent/'asr.json', old)
    atomic_json(parent/'bundle.json', {'lesson': {'id': 'old'}})
    atomic_json(parent/'slides.json', {'slides': [{'image_path': 'slides/1.jpg'}]})
    for relative in ('slides/1.jpg', 'activity/frames/1.jpg'):
        (parent/relative).parent.mkdir(parents=True, exist_ok=True)
        (parent/relative).write_bytes(b'actual-preserved-image')
    atomic_json(parent/'activity.json', {'frames': [{'image_url': '/api/review/media/old/activity/frames/1.jpg',
                                                     'participation_index': None}], 'provenance': {}})
    native = {'status': 'complete', 'time_base': 'source_audio_seconds',
              'segments': [{'id': 'T0001', 'text': '新识别。', 'start_sec': .01, 'end_sec': .09}],
              'provenance': {'source_audio': {'sha256': sha256(parent/'audio.wav')}, 'audio_duration_sec': .1,
                             'timestamp_offset_sec': 0, 'processed_range_sec': [0., .1], 'generator_exhausted': True}}
    native_path = tmp_path/'native.json'
    atomic_json(native_path, native)
    return parent, directory, native_path, native


def test_prepare_retains_native_clock_and_all_source_bytes(sources):
    parent, directory, path, native = sources
    before = {str(p.relative_to(parent)): sha256(p) for p in parent.rglob('*') if p.is_file()}
    prepare(parent, path, directory, 'new')
    raw = read_json(directory/'asr.json')
    assert [{k: row[k] for k in ('id', 'text', 'start_sec', 'end_sec')} for row in raw['segments']] == native['segments']
    assert sha256(directory/'audio.wav') == sha256(parent/'audio.wav')
    assert sha256(directory/'activity/frames/1.jpg') == sha256(parent/'activity/frames/1.jpg')
    activity = read_json(directory/'activity.json')
    assert activity['frames'][0]['participation_index'] is None
    assert activity['frames'][0]['image_url'] == '/api/review/media/new/activity/frames/1.jpg'
    assert {str(p.relative_to(parent)): sha256(p) for p in parent.rglob('*') if p.is_file()} == before
    (directory/'activity/frames/1.jpg').unlink()  # Recover an interrupted preparation.
    prepare(parent, path, directory, 'new')  # Idempotent same frozen input.
    assert sha256(directory/'activity/frames/1.jpg') == sha256(parent/'activity/frames/1.jpg')
    native['segments'][0]['text'] = '另一次识别'
    atomic_json(path, native)
    with pytest.raises(ValueError, match='another ASR snapshot'):
        prepare(parent, path, directory, 'new')


@pytest.mark.parametrize('change', [
    lambda d: d.update(status='partial'),
    lambda d: d['provenance']['source_audio'].update(sha256='other'),
    lambda d: d['provenance'].update(audio_duration_sec=.09),
    lambda d: d['provenance'].update(processed_range_sec=[.01, .1]),
    lambda d: d['provenance'].update(timestamp_offset_sec=1),
    lambda d: d['provenance'].update(generator_exhausted=False),
])
def test_prepare_refuses_wrong_source_or_incomplete_clock(sources, change):
    parent, directory, path, native = sources
    change(native)
    atomic_json(path, native)
    with pytest.raises(ValueError):
        prepare(parent, path, directory, 'new')
    assert not directory.exists()
