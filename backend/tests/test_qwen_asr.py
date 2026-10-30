import wave

import pytest

from review.qwen_asr import build_asr, inspect_wav, native_segments


def test_preserves_native_text_and_actual_predicted_boundaries():
    text = '氨基，羧基。下一句！'
    tokens = [{'text': c, 'start_time': i*.3, 'end_time': (i+1)*.3}
              for i, c in enumerate('氨基羧基下一句')]
    result = native_segments(text, tokens, 5)
    assert ''.join(x['text'] for x in result) == text
    assert result[0]['start_sec'] == tokens[0]['start_time']
    assert result[0]['end_sec'] == tokens[3]['end_time']
    assert result[1]['start_sec'] == tokens[4]['start_time']


def test_missing_timestamps_is_explicit_whole_clip():
    result = native_segments('未知时间。', [], 90)
    assert len(result) == 1
    assert result[0]['end_sec'] == 90
    assert result[0]['timestamp_method'] == 'whole_clip_no_predicted_timestamps'


@pytest.mark.parametrize('tokens', [
    [{'text': '字', 'start_time': 0, 'end_time': 11}],
    [{'text': '字', 'start_time': 0, 'end_time': 10.001}],
    [{'text': '异', 'start_time': 0, 'end_time': 1}],
    [{'text': '字', 'start_time': float('nan'), 'end_time': 1}],
])
def test_invalid_native_evidence_is_rejected(tokens):
    with pytest.raises(ValueError):
        native_segments('字', tokens, 10)


def test_audio_hash_and_original_clock(tmp_path):
    path = tmp_path / 'audio.wav'
    with wave.open(str(path), 'wb') as f:
        f.setnchannels(1); f.setsampwidth(2); f.setframerate(16000)
        f.writeframes(b'\0\0' * 32000)
    audio = inspect_wav(path)
    assert audio['duration_sec'] == 2
    with pytest.raises(ValueError, match='audio_sha256_mismatch'):
        inspect_wav(path, '0'*64)
    result = build_asr({'text': '文本', 'language': 'Chinese'}, audio,
                       {'audio_time_offset_sec': 0})
    assert result['time_base'] == 'source_audio_seconds'
    assert result['provenance']['processed_range_sec'] == [0, 2]
    assert result['provenance']['ground_truth_in_model_inputs'] is False


def test_trailing_zero_duration_native_token_merges_without_text_loss():
    items = [{'text': '甲', 'start_time': 0, 'end_time': 1},
             {'text': '乙', 'start_time': 2, 'end_time': 2}]
    result = native_segments('甲。乙', items, 2)
    assert ''.join(row['text'] for row in result) == '甲。乙'
    assert all(0 <= row['start_sec'] < row['end_sec'] <= 2 for row in result)
