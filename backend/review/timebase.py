"""Validate complete, zero-offset extraction of one registered source."""
import math


def validate_media_timebase(asr,pages):
    a,p=asr['provenance'],pages['provenance']
    sa,sp=a['source_video'],p['source_video']
    def number(value):
        if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value):
            raise ValueError('Media timeline contains missing or non-finite values')
        return value
    def close(left,right,label):
        if abs(number(left)-number(right))>.01:raise ValueError('Media clock mismatch: '+label)
    for key in ['duration_sec','audio_duration_sec','video_duration_sec','size_bytes','audio_start_sec','video_start_sec']:
        close(sa.get(key),sp.get(key),key)
    for key in ['duration_sec','audio_duration_sec','video_duration_sec']:
        if number(sa[key])<=0:raise ValueError('Media duration must be positive')
    for key in ['audio_start_sec','video_start_sec']:
        close(sa[key],0,'Nonzero stream starts need explicit calibration')
    close(a.get('timestamp_offset_sec'),0,'ASR offset')
    close(p.get('timestamp_offset_sec'),0,'OCR offset')
    close(a.get('config',{}).get('audio_time_offset_sec'),0,'ASR extraction offset')
    close(p.get('config',{}).get('timestamp_offset_sec'),0,'OCR extraction offset')
    if a.get('generator_exhausted') is not True:raise ValueError('ASR generator did not finish')
    audio=sa['audio_duration_sec']; video=sa['video_duration_sec']
    for start,end,target,label in [(*a.get('processed_range_sec',[None,None]),audio,'ASR coverage'),
                                 (*p.get('observed_video_range_sec',[None,None]),video,'OCR coverage')]:
        close(start,0,label);close(end,target,label)
    close(a.get('audio_duration_sec'),audio,'Decoded audio duration')
    close(asr.get('coverage',{}).get('processed_duration_sec'),audio,'ASR processed duration')
    coverage=pages.get('coverage',{})
    close(coverage.get('video_duration_sec'),video,'OCR stream duration')
    count,expected=coverage.get('sample_count'),coverage.get('expected_sample_count')
    if not isinstance(count,int) or count<=0 or count!=expected:raise ValueError('OCR sampling is incomplete')
    interval=number(p.get('config',{}).get('sample_interval_sec'))
    if interval<=0:raise ValueError('OCR interval must be positive')
    close(coverage.get('first_sample_sec'),0,'First OCR sample')
    last=number(coverage.get('last_sample_sec'));gap=number(coverage.get('max_sampling_gap_sec'))
    if not 0<=video-last<=interval+.01 or not 0<=gap<=interval+.01:
        raise ValueError('OCR sampling does not cover full video')
    return sa['duration_sec']
