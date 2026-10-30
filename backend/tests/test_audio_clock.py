"""AAC padding is trimmed to the recorded stream, never hidden by clock tolerance."""
import json
from pathlib import Path
import shutil
import subprocess
import sys
import wave
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from review.asr import decode_audio


def test_decode_audio_bounds_pcm_to_source_stream_duration(tmp_path):
    if not shutil.which('ffmpeg'):pytest.skip('FFmpeg required')
    source=tmp_path/'test-only-tone.m4a'
    subprocess.run(['ffmpeg','-v','error','-f','lavfi','-i','sine=frequency=440:sample_rate=44100:duration=0.2',
                    '-c:a','aac',str(source)],check=True)
    probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-of','json',str(source)]))
    duration=float(probe['streams'][0]['duration'])
    decoded=tmp_path/'decoded.wav'
    decode_audio(source,decoded,duration)
    with wave.open(str(decoded),'rb') as stream:
        assert stream.getframerate()==16000 and stream.getnchannels()==1
        assert abs(stream.getnframes()/16000-duration)<=1/16000


def test_invalid_decoded_duration_rejected_before_launch(tmp_path):
    for value in [0,-1,float('nan')]:
        with pytest.raises(ValueError):decode_audio(tmp_path/'missing',tmp_path/'out.wav',value)
