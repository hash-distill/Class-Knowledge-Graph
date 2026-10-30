from copy import deepcopy

import pytest

from review.common import atomic_json, sha256
from review.correction_comparison import overlay_comparison
from review.courseware_service import overlay_bundle, source_fingerprint


def source_fixture(tmp_path):
    atomic_json(tmp_path/'asr.json', {'segments':[]})
    row={'id':'T0001','text':'氨基酸','start_sec':1.,'end_sec':2.}
    bundle={'lesson':{'source_video':{'sha256':'a'*64},'provenance':{}}, 'transcript':[row],
            'slides':[], 'graph':{'nodes':[]}, 'report':{'sections':[],'limitations':[],'statistics':{}}}
    segment={'transcript_id':row['id'],'original_text':row['text'],'text':'氨基酸',
             'start_sec':1.,'end_sec':2.}
    comparison={'schema_version':'1.0','source_video_sha256':'a'*64,'source_asr_sha256':sha256(tmp_path/'asr.json'),
                'runs':[{'id':name,'segments':[deepcopy(segment)]} for name in ['qwen','cec4']]}
    return bundle,comparison


@pytest.mark.parametrize('damage,reason', [
    ('hash','comparison_asr_changed'), ('text','comparison_original_changed'),
    ('clock','comparison_clock_changed'), ('duplicate','comparison_original_changed'),
    ('missing','comparison_incomplete_coverage')])
def test_stale_or_invalid_optional_comparison_is_not_published(tmp_path,damage,reason):
    bundle,comparison=source_fixture(tmp_path)
    if damage=='hash': comparison['source_asr_sha256']='b'*64
    if damage=='text': comparison['runs'][0]['segments'][0]['original_text']='被换过的原文'
    if damage=='clock': comparison['runs'][0]['segments'][0]['start_sec']=0.
    if damage=='duplicate': comparison['runs'][0]['segments'] *= 2
    if damage=='missing': comparison['runs'][0]['segments']=[]
    atomic_json(tmp_path/'correction.comparison.json',comparison)
    result=overlay_comparison(bundle,tmp_path)
    assert 'correction_comparison' not in result
    assert result['correction_comparison_status']['reason']==reason
    assert bundle['lesson']['provenance']=={}


def test_valid_comparison_survives_courseware_overlay_without_entering_evidence(tmp_path):
    bundle,comparison=source_fixture(tmp_path)
    atomic_json(tmp_path/'correction.comparison.json',comparison)
    result=overlay_comparison(bundle,tmp_path)
    document={'lesson_fingerprint':source_fingerprint(result),'pages':[],'matches':[],
              'document_id':'fixture','filename':'fixture.pdf','sha256':'c'*64}
    imported=overlay_bundle(result,document)
    assert imported['correction_comparison']==comparison
    assert imported['transcript']==bundle['transcript']
    assert result['lesson']['provenance']['correction_comparison']['sha256']==sha256(tmp_path/'correction.comparison.json')


def test_removed_or_changed_comparison_drops_previous_overlay(tmp_path):
    bundle,comparison=source_fixture(tmp_path)
    path=tmp_path/'correction.comparison.json'
    atomic_json(path,comparison)
    overlaid=overlay_comparison(bundle,tmp_path)
    comparison['source_asr_sha256']='d'*64
    atomic_json(path,comparison)
    stale=overlay_comparison(overlaid,tmp_path)
    assert 'correction_comparison' not in stale
    assert 'correction_comparison' not in stale['lesson']['provenance']
    assert stale['correction_comparison_status']['status']=='unavailable'
    path.unlink()
    removed=overlay_comparison(overlaid,tmp_path)
    assert 'correction_comparison' not in removed
    assert 'correction_comparison_status' not in removed
