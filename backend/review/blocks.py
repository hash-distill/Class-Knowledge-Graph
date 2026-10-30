"""Readable retrieval units composed only of adjacent native ASR fragments."""


def speech_blocks(transcript,max_seconds=22,max_gap=3,max_chars=240):
    blocks=[];pending=[]
    def flush():
        if not pending:return
        blocks.append({'id':f'B{len(blocks)+1:04d}','kind':'transcript',
            'start_sec':pending[0]['start_sec'],'end_sec':pending[-1]['end_sec'],
            'text':' '.join(r['text'] for r in pending),
            'source_refs':[r['id'] for r in pending],
            'method':'adjacent_native_asr_fragments','quality':'machine_transcript_unreviewed'})
        pending.clear()
    for row in transcript:
        if pending and (row['start_sec']-pending[-1]['end_sec']>max_gap or
                row['end_sec']-pending[0]['start_sec']>max_seconds or
                sum(len(r['text']) for r in pending)+len(row['text'])>max_chars):flush()
        pending.append(row)
    flush()
    return blocks
