"""Publication assembly binds measured lanes without touching real artifacts."""
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from review.common import atomic_json, read_json, sha256
from scripts.publish_correction_comparison import assemble
from scripts import publish_correction_comparison as publisher


@pytest.fixture
def measured(tmp_path):
    directory, runs = tmp_path/'lesson', tmp_path/'measured'
    directory.mkdir(); runs.mkdir()
    original={'id':'T0001','start_sec':1.,'end_sec':2.,'text':'有一个缩疾'}
    atomic_json(directory/'asr.json',{'segments':[original]})
    identity={'source_asr_sha256':sha256(directory/'asr.json'),'source_video_sha256':'a'*64}
    atomic_json(directory/'bundle.json',{'lesson':{'source_video':{'sha256':'a'*64}},
        'transcript':[original]})
    report={'source':identity,'runs':[]}
    for name,alias,model_hash in [('qwen','classroom-local','b'*64),
                                  ('cec4','classroom-corrector','c'*64)]:
        edits=[{'origin':'source_vocabulary'}]
        if name=='qwen':
            edits.append({'origin':'model'})
        segment={'transcript_id':'T0001','start_sec':1.,'end_sec':2.,
            'original_text':original['text'],'text':'有一个羧基','uncertainties':[], 'edits':edits}
        result={**identity,'model':alias,'status':'complete','edit_count':len(edits),
            'segments':[segment], 'model_proposals':[], 'proposal_ledger':[],
            'input_fingerprint':{'provider':{'model':alias,'model_sha256':model_hash}}}
        controlled={'result':result,'runtime':{},'conditions':{'track':'controlled_json'}}
        path=runs/f'{name}-controlled.json'
        atomic_json(path,controlled)
        report['runs'].append({'label':name+'-controlled','sha256':sha256(path),
            'raw_model_proposal_count':1 if name=='qwen' else 0,
            'guard_rejected_proposal_count':0,
            'deterministic_comparison':{'aligned_changed_spans_additional_to_deterministic':1 if name=='qwen' else 0},
            'accepted':{'summary':{'terminology_group_presence_count':1,'terminology_group_count':1}}})
        atomic_json(runs/f'{name}-native.json',{'model':alias,
            'result':{**identity,'status':'complete','model':alias},
            'benchmark_contract':{'model_sha256':model_hash},
            'conditions':{'track':'native_text'},'runtime':{},
            'native_outputs':[{'transcript_id':'T0001','start_sec':1.,'end_sec':2.,
                'original_text':original['text'],'corrected_text':'有一个羧基','status':'complete',
                'metadata':{'model':alias,'model_sha256':model_hash}}]})
    atomic_json(runs/'controlled-report/comparison.json',report)
    return directory,runs


def test_assemble_preserves_raw_bytes_and_keeps_source_rule_attribution(measured):
    directory,runs=measured
    paths=[*directory.glob('*.json'),*runs.rglob('*.json')]
    before={path:sha256(path) for path in paths}
    result=assemble(directory,runs)
    qwen,cec4=result['runs']
    assert (qwen['accepted_edit_count'],qwen['model_accepted_edit_count'],
            qwen['source_rule_accepted_count'])==(2,1,1)
    assert (cec4['accepted_edit_count'],cec4['model_accepted_edit_count'],
            cec4['source_rule_accepted_count'])==(1,0,1)
    assert cec4['additional_changed_spans_vs_rules']==0
    assert cec4['native_segments'][0]['original_text']=='有一个缩疾'
    assert cec4['audit']['native_call_records'][0]['metadata']['model']=='classroom-corrector'
    assert {path:sha256(path) for path in paths}==before
    assert not (directory/'correction.comparison.json').exists()
    assert not (directory/'transcript.proofreading.json').exists()


def test_stale_controlled_report_cannot_label_new_run_statistics(measured):
    directory,runs=measured
    path=runs/'controlled-report/comparison.json'
    report=read_json(path)
    report['runs'][0]['sha256']='0'*64
    report['runs'][0]['raw_model_proposal_count']=999
    atomic_json(path,report)
    with pytest.raises(ValueError,match='stale relative'):
        assemble(directory,runs)


def test_swapped_native_model_cannot_be_published_under_filename_label(measured):
    directory,runs=measured
    atomic_json(runs/'qwen-native.json',read_json(runs/'cec4-native.json'))
    with pytest.raises(ValueError,match='model identit'):
        assemble(directory,runs)


@pytest.mark.parametrize('damage',['top_model','result_model','contract_hash','response_model','response_hash'])
def test_native_model_identity_must_match_controlled_run(measured,damage):
    directory,runs=measured
    path=runs/'qwen-native.json'
    native=read_json(path)
    if damage=='top_model': native['model']='classroom-corrector'
    if damage=='result_model': native['result']['model']='classroom-corrector'
    if damage=='contract_hash': native['benchmark_contract']['model_sha256']='c'*64
    if damage=='response_model': native['native_outputs'][0]['metadata']['model']='classroom-corrector'
    if damage=='response_hash': native['native_outputs'][0]['metadata']['model_sha256']='c'*64
    atomic_json(path,native)
    with pytest.raises(ValueError,match='model identit'):
        assemble(directory,runs)


@pytest.mark.parametrize('field,value,reason',[
    ('original_text','原文被改过','comparison_original_changed'),
    ('start_sec',0.,'comparison_clock_changed'),
    ('end_sec',3.,'comparison_clock_changed')])
def test_native_original_and_clock_are_validated_against_lesson(measured,field,value,reason):
    directory,runs=measured
    path=runs/'cec4-native.json'; native=read_json(path)
    native['native_outputs'][0][field]=value
    atomic_json(path,native)
    with pytest.raises(ValueError,match=reason):
        assemble(directory,runs)


def test_failed_native_request_can_preserve_original_without_model_metadata(measured):
    directory,runs=measured
    path=runs/'cec4-native.json'; native=read_json(path)
    native['result']['status']='partial'
    row=native['native_outputs'][0]
    row.update(status='failed_preserved_original',corrected_text=row['original_text'],
        metadata={},error={'code':'model_timeout','message':'请求超时，原文保留'})
    atomic_json(path,native)
    result=assemble(directory,runs)
    assert result['runs'][1]['native_status']=='partial'
    assert result['runs'][1]['native_segments'][0]['uncertainties']==['请求超时，原文保留']


def test_changed_baseline_source_is_rejected_before_any_publication(measured,monkeypatch):
    directory,runs=measured
    atomic_json(runs/'source-before.json',{'asr.json':'0'*64})
    monkeypatch.setattr(publisher,'LessonMutationLock',lambda *a:nullcontext())
    monkeypatch.setattr(publisher,'atomic_json',lambda *a:pytest.fail('Stale source was published'))
    with pytest.raises(ValueError,match='Immutable source changed'):
        publisher.publish(directory,runs)


def test_proofreading_publication_requires_current_production_fingerprint(measured,monkeypatch):
    directory,runs=measured
    atomic_json(runs/'source-before.json',{'asr.json':sha256(directory/'asr.json')})
    monkeypatch.setattr(publisher,'LessonMutationLock',lambda *a:nullcontext())
    monkeypatch.setattr(publisher,'CorrectionProvider',lambda **kw:SimpleNamespace(
        status=lambda:{'model':'different-production-profile'}))
    monkeypatch.setattr(publisher,'atomic_json',lambda *a:pytest.fail('Wrong profile was published'))
    with pytest.raises(ValueError,match='differs from production profile'):
        publisher.publish(directory,runs,publish_proofreading=True)


def test_default_publication_only_writes_optional_comparison_artifact(measured,monkeypatch):
    directory,runs=measured
    atomic_json(runs/'source-before.json',{'asr.json':sha256(directory/'asr.json')})
    writes=[]
    monkeypatch.setattr(publisher,'LessonMutationLock',lambda *a:nullcontext())
    monkeypatch.setattr(publisher,'atomic_json',lambda path,data:writes.append(path.name))
    publisher.publish(directory,runs)
    assert writes==['correction.comparison.json']
    assert not (directory/'correction.comparison.json').exists()
    assert not (directory/'transcript.proofreading.json').exists()
