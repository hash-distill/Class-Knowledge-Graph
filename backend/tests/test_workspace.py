"""Teacher records persist independently of immutable machine observations."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from review.provider import ReviewError


def fixture():
    return {'lesson': {'id': 'test', 'source_video': {'sha256': 'a'*64}, 'duration_sec': 10},
            'transcript': [{'id': 'T0001', 'source': 'asr', 'text': '原始识别', 'start_sec': 0, 'end_sec': 5},
                           {'id': 'T0002', 'source': 'asr', 'text': '后续内容', 'start_sec': 5, 'end_sec': 10}],
            'slides': [{'id': 'S0001', 'text': '真实课件', 'start_sec': 0, 'end_sec': 5}]}


def store(tmp_path):
    assert importlib.util.find_spec('review.workspace') is not None, 'Persistent teacher workspace is missing.'
    from review.workspace import WorkspaceStore
    return WorkspaceStore(tmp_path/'workspace.sqlite3')


def test_goals_survive_recreation_and_reject_stale_writes(tmp_path):
    db=store(tmp_path);bundle=fixture()
    assert db.get(bundle)['version']==0
    result=db.set_goals(bundle,'认识证据与推断',0)
    assert result['version']==1 and store(tmp_path).get(bundle)['goals']=='认识证据与推断'
    with pytest.raises(ReviewError) as error:db.set_goals(bundle,'stale',0)
    assert error.value.code=='workspace_changed' and error.value.status==409


def test_teacher_correction_overlay_and_revert_preserve_raw_evidence(tmp_path):
    db=store(tmp_path);bundle=fixture();before=deepcopy(bundle)
    edited=db.correct(bundle,'T0001','教师核对后的转写','已回听',0)
    assert edited['version']==1 and edited['corrections'][0]['original_text']=='原始识别'
    derived=db.overlay(bundle)
    assert derived['transcript']==before['transcript'] and bundle==before
    assert derived['teacher_transcript'][0]['text']=='教师核对后的转写'
    assert derived['teacher_evidence_blocks'][0]['source_refs'][0]=='T0001'
    assert derived['teacher_evidence_blocks'][0]['start_sec']==0
    result=db.revert(bundle,'T0001',1)
    assert result['corrections']==[] and result['correction_history'][0]['revoked_at']
    assert 'teacher_transcript' not in db.overlay(bundle)


def test_action_requires_real_outcome_and_valid_evidence(tmp_path):
    db=store(tmp_path);bundle=fixture()
    with pytest.raises(ReviewError):db.add_action(bundle,'复查术语',['fabricated'],0)
    w=db.add_action(bundle,'下一课回顾术语',['T0001'],0)
    action=w['actions'][0]
    with pytest.raises(ReviewError):db.update_action(bundle,action['id'],'done','',1)
    w=db.update_action(bundle,action['id'],'done','已在下一课回顾并记录学生回答',1)
    assert w['actions'][0]['outcome'] and w['actions'][0]['status']=='done'
    assert store(tmp_path).get(bundle)['actions']==w['actions']


def test_exercise_and_actual_answer_are_persisted_without_invented_score(tmp_path):
    db=store(tmp_path);bundle=fixture()
    w=db.add_assessment(bundle,'课件中出现了什么？','真实课件',['S0001'],0)
    exercise=w['assessments'][0]
    assert exercise['teacher_confirmed'] is True
    w=db.add_attempt(bundle,exercise['id'],'实际输入的回答','教师待核对',None,1)
    assert w['attempts'][0]['score'] is None and w['attempts'][0]['answer']=='实际输入的回答'
    assert store(tmp_path).get(bundle)['attempts']==w['attempts']
    with pytest.raises(ReviewError):db.add_attempt(bundle,exercise['id'],'x','',2,2)


def test_changed_source_is_explicit_and_requires_reconciliation(tmp_path):
    db=store(tmp_path);bundle=fixture()
    db.correct(bundle,'T0001','核对稿','',0)
    changed=deepcopy(bundle);changed['transcript'][0]['text']='重新生成的原始文本'
    assert db.get(changed)['source_changed'] is True
    assert 'teacher_transcript' not in db.overlay(changed)
    with pytest.raises(ReviewError) as error:db.set_goals(changed,'new',1)
    assert error.value.code=='workspace_source_changed'
    w=db.reconcile(changed,1)
    assert w['source_changed'] is False and w['corrections']==[] and w['history']
    assert db.history(bundle['lesson']['id'])[0]['snapshot']['corrections'][0]['replacement']=='核对稿'


def test_effective_evidence_is_used_by_retrieval(tmp_path):
    from review.retrieval import evidence_catalog
    db=store(tmp_path);bundle=fixture()
    db.correct(bundle,'T0001','确认后的术语','回听',0)
    rows=evidence_catalog(db.overlay(bundle))
    assert any('确认后的术语' in row['text'] for row in rows)
    assert not any('原始识别' in row['text'] for row in rows)


def test_derived_evidence_can_support_exercises_and_retains_uncertainty(tmp_path):
    from review.retrieval import evidence_prompt
    db=store(tmp_path);bundle=fixture()
    bundle['transcript_proofreading']={'segments':[
        {'id':'P0002','transcript_id':'T0002','text':'后续疑词','uncertainties':['术语待回听'],
         'uncertain_spans':[{'text':'疑词','scope':'lexical'}]}]}
    db.correct(bundle,'T0001','教师核对内容','回听',0)
    derived=db.overlay(bundle)
    block=derived['teacher_evidence_blocks'][0]
    assert '术语待回听' in block['uncertainties']
    prompt,_=evidence_prompt([block])
    assert '机器上下文' in prompt and '术语待回听' in prompt
    result=db.add_assessment(derived,'核对了什么？','教师核对内容',[block['id']],1)
    assert result['assessments'][0]['evidence_snapshot'][0]['quality']['human_verified'] is False


def test_reverted_evidence_cannot_be_reused_from_stale_derived_bundle(tmp_path):
    db=store(tmp_path);raw=fixture()
    db.correct(raw,'T0001','已撤销的核对稿','',0)
    previous=db.overlay(raw)
    db.revert(raw,'T0001',1)
    assert 'teacher_evidence_blocks' not in db.overlay(previous)
    with pytest.raises(ReviewError):
        db.add_assessment(previous,'问题','答案',['HB0001'],2)
