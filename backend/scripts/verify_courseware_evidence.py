"""Record full real-source integrity and separately labeled synthetic term checks."""
import json
import sys
from collections import Counter
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from review.common import read_json, sha256, atomic_json
from review.courseware_terms import revise_transcript
from review.blocks import speech_blocks

root = Path(__file__).resolve().parents[1]
folder = root/'artifacts/mit-optimization-real'
bundle, raw, ocr = (read_json(folder/name) for name in ('bundle.json', 'asr.json', 'slides.json'))
source = root/'artifacts/public-courseware/mit-6-0002-f2016-lec01'
provenance = read_json(source/'provenance.json')
assert sha256(source/'lecture01.mp4') == bundle['lesson']['source_video']['sha256']
assert sha256(source/'slides.pdf') == bundle['courseware']['sha256']
assert [(s['id'],s['text'],s['start_sec'],s['end_sec']) for s in raw['segments']] == [(s['id'],s['text'],s['start_sec'],s['end_sec']) for s in bundle['transcript']]
assert raw['status'] == ocr['status'] == 'complete'
assert ocr['coverage']['sample_count'] == ocr['coverage']['expected_sample_count'] == 493
assert len(bundle['courseware']['pages']) == 31
for page in bundle['courseware']['pages']:
    assert (folder/page['image_path']).is_file()
checks = []
for time, number in ((180,4),(600,11),(900,14)):
    match = next(m for m in bundle['courseware']['matches'] if m['start_sec'] <= time < m['end_sec'])
    page = next(p for p in bundle['courseware']['pages'] if p['id'] == match['page_id'])
    assert page['page_number'] == number and match['status'] == 'auto_matched'
    checks.append({'time_sec':time,'reference_page':number,'matched_page':page['page_number'],'match_id':match['id']})
result = {'scope':'actual_official_full_video_pdf_integrity_and_three_visual_spotchecks',
          'duration_sec':bundle['lesson']['duration_sec'],'video_sha256':bundle['lesson']['source_video']['sha256'],
          'document_sha256':bundle['courseware']['sha256'],'transcript_segments':len(bundle['transcript']),
          'raw_transcript_unchanged':True,'ocr_samples':493,'observed_segments':len(bundle['slides']),
          'courseware_pages':31,'match_status_counts':dict(Counter(m['status'] for m in bundle['courseware']['matches'])),
          'spot_checks':checks,'official_video_page':provenance['official_page'],
          'official_courseware_page':provenance['slides_resource_page'],
          'accuracy_evaluated':False,'official_subtitles_used_as_asr':False}
out = root/'artifacts/verification/courseware'
atomic_json(out/'real_source_validation.json',result)

cases=[]
for text, canonical, expected in [
    ('根據鄭賢定理討論相量。','正弦定理\n向量','根據正弦定理討論向量。'),
    ('韓數','函数','函数'),
    ('相量 = x','向量','相量 = x'),
    ('(相量)=x','向量','(相量)=x'),
    ('相量＝x','向量','相量＝x'),
    ('相量-x','向量','相量-x'),
]:
    transcript=[{'id':'T0001','source':'synthetic_test','text':text,'start_sec':1,'end_sec':5}]
    sample={'transcript':transcript,'slides':[{'id':'S0001','start_sec':0,'end_sec':10}],
            'evidence_blocks':speech_blocks(transcript)}
    document={'document_id':'Dsynthetic','pages':[{'id':'Psynthetic','document_id':'Dsynthetic','page_number':1,'text':canonical}],
              'matches':[{'id':'Msynthetic','document_id':'Dsynthetic','slide_id':'S0001','page_id':'Psynthetic',
                          'start_sec':0,'end_sec':10,'status':'confirmed','score':1,'method':'synthetic_fixture_match'}]}
    derived=revise_transcript(sample,document)
    output=derived['transcript_revisions'][0]['corrected_text'] if derived['transcript_revisions'] else text
    assert output==expected,(text,output,expected)
    assert transcript[0]['text']==text
    cases.append({'original':text,'page_text':canonical,'result':output,'expected':expected,'passed':True,
                  'revisions':derived['transcript_revisions']})
atomic_json(out/'synthetic_chinese_term_checks.json',{'scope':'synthetic_mechanism_only_not_classroom_accuracy','cases':cases})
print(json.dumps(result,ensure_ascii=False))
print('Synthetic Chinese term/guard checks:',len(cases),'passed')
