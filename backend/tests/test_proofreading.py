"""Generation-time proofreading must be reversible, bounded and source-linked."""
from copy import deepcopy
import json

from review.retrieval import evidence_catalog


def records():
    return ([{'id': 'T0001', 'start_sec': 0., 'end_sec': 4., 'text': '有一个氨基一个缩疾'},
             {'id': 'T0002', 'start_sec': 4., 'end_sec': 8., 'text': '知道未经吗'}],
            [{'id': 'S0001', 'start_sec': 2., 'end_sec': 3., 'text': '氨基酸 羧基'}])


class Provider:
    def __init__(self, response):
        self.response = response
        self.prompts = []

    def status(self):
        return {'source': 'test_only', 'model': 'fixture-model'}

    def complete(self, system, prompt, **kwargs):
        self.prompts.append(prompt)
        return deepcopy(self.response), {'source': 'test_only', 'model': 'fixture-model'}


def test_retrieval_prefers_proofread_blocks_over_raw_and_old_revision():
    t, s = records()
    bundle = {'transcript': t, 'slides': s,
              'revised_evidence_blocks': [{'id': 'RB1', 'text': '旧建议'}],
              'proofread_evidence_blocks': [{'id': 'PB0001', 'text': '有一个氨基一个羧基',
                  'kind': 'transcript_proofread', 'start_sec': 0., 'end_sec': 4.}]}
    rows = evidence_catalog(bundle)
    assert {r['id'] for r in rows} == {'PB0001', 'S0001'}


def test_local_edits_preserve_source_and_refuse_sentence_rewrite():
    from review.proofreading import proofread_transcript
    t, s = records()
    original = deepcopy(t)
    provider = Provider({'segments': [
        {'transcript_id': 'T0001', 'edits': [{'original': '缩疾', 'replacement': '羧基',
             'reason': '同音术语且课件出现羧基', 'evidence_ids': ['T0001', 'S0001']}], 'uncertainties': []},
        {'transcript_id': 'T0002', 'edits': [{'original': '知道未经吗', 'replacement': '谷氨酸钠就是味精',
             'reason': '模型扩写', 'evidence_ids': ['T0002']}], 'uncertainties': []}]})
    result = proofread_transcript(t, s, provider, source_asr_sha256='a'*64, source_video_sha256='b'*64)
    assert t == original
    rows = result['segments']
    assert rows[0]['text'] == '有一个氨基一个羧基'
    assert rows[0]['original_text'] == t[0]['text']
    assert rows[0]['start_sec'] == t[0]['start_sec'] and rows[0]['end_sec'] == t[0]['end_sec']
    assert rows[1]['text'] == t[1]['text'] and rows[1]['uncertainties']
    assert result['rejected_edit_count'] == 1 and result['human_verified'] is False
    assert 'T0002' in provider.prompts[0] and 'S0001' in provider.prompts[0]


def test_model_proposals_and_deterministic_overlap_are_auditable():
    from review.proofreading import proofread_transcript
    t, s = records()
    proposal = {'original': '缩疾', 'replacement': '羧基', 'reason': '模型建议',
                'evidence_ids': ['T0001', 'S0001']}
    answer = {'segments': [{'transcript_id': 'T0001', 'edits': [proposal], 'uncertainties': []}]}
    result = proofread_transcript(t, s, Provider(answer), source_asr_sha256='a'*64,
                                 source_video_sha256='b'*64)
    assert result['model_proposals'][0]['answer'] == answer
    assert len(result['model_proposals'][0]['prompt_digest']) == 64
    ledger = result['proposal_ledger']
    assert any(r['channel'] == 'source_vocabulary' and r['disposition'] == 'accepted' for r in ledger)
    model = next(r for r in ledger if r['channel'] == 'model')
    assert model['candidate'] == proposal and model['disposition'] == 'overlap_discarded'
    assert result['segments'][0]['edits'][0]['origin'] == 'source_vocabulary'
    assert result['contribution_counts']['model']['accepted'] == 0


def test_duplicate_model_segment_rows_are_retained_instead_of_overwritten():
    from review.proofreading import proofread_transcript
    t = [{'id': 'T0001', 'start_sec': 0., 'end_sec': 4., 'text': '一个缩疾谷氨酸纳'}]
    answer = {'segments': [
        {'transcript_id': 'T0001', 'edits': [{'original': '缩疾', 'replacement': '羧基',
          'reason': '模型建议', 'evidence_ids': ['T0001']}], 'uncertainties': []},
        {'transcript_id': 'T0001', 'edits': [{'original': '纳', 'replacement': '钠',
          'reason': '模型建议', 'evidence_ids': ['T0001']}], 'uncertainties': []}]}
    result = proofread_transcript(t, [], Provider(answer), source_asr_sha256='a'*64,
                                 source_video_sha256='b'*64)
    assert result['segments'][0]['text'] == '一个羧基谷氨酸钠'
    assert len([r for r in result['proposal_ledger'] if r['channel'] == 'model']) == 2


def test_unknown_refs_negation_and_wrong_phonetics_are_rejected():
    from review.proofreading import proofread_transcript
    t, s = records()
    provider = Provider({'segments': [{'transcript_id': 'T0001', 'edits': [
        {'original': '一个', 'replacement': '不是', 'reason': '改否定', 'evidence_ids': ['T0001']},
        {'original': '氨基', 'replacement': '细胞', 'reason': '换话题', 'evidence_ids': ['T0001']},
        {'original': '缩疾', 'replacement': '羧基', 'reason': '伪造引用', 'evidence_ids': ['BAD']}],
        'uncertainties': []}]})
    result = proofread_transcript(t, [], provider, source_asr_sha256='a'*64, source_video_sha256='b'*64)
    assert result['segments'][0]['text'] == t[0]['text']
    assert result['rejected_edit_count'] == 3


def test_cache_invalidates_when_raw_asr_bytes_change(tmp_path):
    from review.proofreading import load_or_proofread
    from review.common import atomic_json
    t, s = records()
    atomic_json(tmp_path/'asr.json', {'segments': t})
    provider = Provider({'segments': []})
    first = load_or_proofread(tmp_path, t, s, provider, 'b'*64)
    assert len(provider.prompts) == 1
    again = load_or_proofread(tmp_path, t, s, provider, 'b'*64)
    assert again == first and len(provider.prompts) == 1
    s[0]['image_url']='/api/review/media/fixture/slides/S0001.jpg'
    with_serving_url=load_or_proofread(tmp_path,t,s,provider,'b'*64)
    assert with_serving_url == first and len(provider.prompts) == 1
    t[0]['text'] += '啊'
    atomic_json(tmp_path/'asr.json', {'segments': t})
    changed = load_or_proofread(tmp_path, t, s, provider, 'b'*64)
    assert changed['source_asr_sha256'] != first['source_asr_sha256'] and len(provider.prompts) == 2


def test_comparison_output_keeps_published_proofreading_bytes(tmp_path):
    from review.proofreading import load_or_proofread
    from review.common import atomic_json
    t, s = records()
    atomic_json(tmp_path/'asr.json', {'segments': t})
    published = tmp_path/'transcript.proofreading.json'
    published.write_text('{"frozen":"original"}',encoding='utf-8')
    output = tmp_path/'comparison'/'cec4.json'
    result = load_or_proofread(tmp_path,t,s,Provider({'segments':[]}), 'b'*64,artifact_path=output)
    assert result['source_asr_sha256']
    assert output.is_file()
    assert published.read_text(encoding='utf-8') == '{"frozen":"original"}'


def test_failure_record_preserves_invalid_response_and_reports_partial():
    from review.proofreading import proofread_transcript
    from review.provider import ReviewError
    class FailedProvider:
        def complete(self,*args,**kwargs):
            raise ReviewError('invalid_model_json','invalid',diagnostic={'raw_response':'坏结果'})
    t, _ = records()
    result = proofread_transcript(t,[],FailedProvider(),source_asr_sha256='a'*64,source_video_sha256='b'*64)
    assert result['status'] == 'partial'
    assert result['model_proposals'][0]['metadata']['raw_response'] == '坏结果'
    assert result['model_proposals'][0]['error'] == 'invalid_model_json'



def test_blocks_have_unique_derived_ids_and_trace_raw_sources():
    from review.proofreading import proofread_blocks
    t, _ = records()
    segments = [{**r, 'id': f'P{i+1:04d}', 'transcript_id': r['id'], 'original_text': r['text'],
                 'source_refs': [r['id'], 'S0001'], 'edits': [], 'uncertainties': []}
                for i, r in enumerate(t)]
    blocks = proofread_blocks({'segments': segments})
    assert blocks[0]['id'] == 'PB0001' and blocks[0]['kind'] == 'transcript_proofread'
    assert blocks[0]['proofreading_ids'] == ['P0001', 'P0002']
    assert {'P0001', 'P0002', 'T0001', 'T0002', 'S0001'} <= set(blocks[0]['source_refs'])


def test_alternative_place_names_cannot_replace_attested_subject_terms():
    from review.proofreading import proofread_transcript
    t = [{'id': 'T0001', 'start_sec': 0., 'end_sec': 4., 'text': '氨基酸的氨基'}]
    s = [{'id': 'S0001', 'start_sec': 1., 'end_sec': 2., 'text': '认识氨基酸'}]
    alternate = {'segments': [{'id': 'T1', 'start_sec': 0., 'end_sec': 4., 'text': '安吉酸的安吉'}]}
    result = proofread_transcript(t, s, Provider({'segments': []}), source_asr_sha256='a'*64,
                                 source_video_sha256='b'*64, alternates=[alternate])
    assert result['segments'][0]['text'] == t[0]['text']


def test_same_time_alternative_restores_word_without_losing_question():
    from review.proofreading import proofread_transcript
    t = [{'id': 'T0001', 'start_sec': 0., 'end_sec': 4., 'text': '知道未经吗'}]
    alternate = {'segments': [{'id': 'T1', 'start_sec': 0., 'end_sec': 4., 'text': '知道味精吗'}]}
    result = proofread_transcript(t, [], Provider({'segments': []}), source_asr_sha256='a'*64,
                                 source_video_sha256='b'*64, alternates=[alternate])
    assert result['segments'][0]['text'] == '知道味精吗'
    assert result['segments'][0]['edits'][0]['supporting_literal_refs'] == ['U1_0001']


def test_courseware_overlay_preserves_generation_proofreading():
    from review.proofreading import proofread_transcript, overlay_proofreading
    from review.courseware_service import overlay_bundle, source_fingerprint
    t, s = records()
    result = proofread_transcript(t, s, Provider({'segments': []}), source_asr_sha256='a'*64,
                                 source_video_sha256='b'*64)
    base = {'lesson': {'source_video': {'sha256': 'b'*64}, 'provenance': {}},
            'transcript': t, 'slides': s, 'graph': {'nodes': [], 'provenance': {}},
            'report': {'sections': [], 'limitations': [], 'statistics': {}}}
    base = overlay_proofreading(base, result)
    document = {'document_id': 'D1', 'filename': 'synthetic.pdf', 'sha256': 'c'*64,
                'pages': [], 'matches': [], 'lesson_fingerprint': source_fingerprint(base)}
    first = overlay_bundle(base, document)
    second = overlay_bundle(base, document)
    assert first['transcript_proofreading'] == second['transcript_proofreading'] == result
    assert first['proofread_evidence_blocks'] == base['proofread_evidence_blocks']


def test_secondary_audio_hash_mismatch_is_rejected(tmp_path):
    import pytest
    from review.proofreading import load_or_proofread
    from review.common import atomic_json
    t, s = records()
    atomic_json(tmp_path/'asr.json', {'segments': t, 'provenance': {'audio_sha256': 'c'*64}})
    atomic_json(tmp_path/'asr.secondary.json', {'segments': [], 'provenance': {
        'source_video': {'sha256': 'b'*64}, 'audio_sha256': 'd'*64}})
    with pytest.raises(ValueError, match='audio'):
        load_or_proofread(tmp_path, t, s, Provider({'segments': []}), 'b'*64)


def test_formula_operators_and_numeric_order_cannot_be_changed():
    from review.proofreading import proofread_transcript
    for before, after in [('x+y', 'x-y'), ('11.2', '1.12'), ('十二', '二十')]:
        t = [{'id': 'T0001', 'start_sec': 0., 'end_sec': 4., 'text': '计算'+before}]
        provider = Provider({'segments': [{'transcript_id': 'T0001', 'edits': [{
            'original': before, 'replacement': after, 'reason': '模型候选', 'evidence_ids': ['T0001']}],
            'uncertainties': []}]})
        result = proofread_transcript(t, [], provider, source_asr_sha256='a'*64, source_video_sha256='b'*64)
        assert result['segments'][0]['text'] == t[0]['text']
        assert result['rejected_edit_count'] == 1


def test_malformed_model_uncertainty_and_references_degrade_without_exception():
    from review.proofreading import proofread_transcript
    t, _ = records()
    for uncertainties, refs in [(None, ['T0001']), ([], [['T0001']]), ([], [{'id': 'T0001'}])]:
        provider = Provider({'segments': [{'transcript_id': 'T0001', 'edits': [{
            'original': '缩疾', 'replacement': '羧基', 'reason': '候选', 'evidence_ids': refs}],
            'uncertainties': uncertainties}]})
        result = proofread_transcript(t, [], provider, source_asr_sha256='a'*64, source_video_sha256='b'*64)
        assert result['segments'][0]['text'] == t[0]['text']
        assert result['segments'][0]['uncertainties']


def test_explicit_model_uncertainty_cannot_feed_concept_extraction():
    from review.proofreading import proofread_transcript, graph_transcript
    t = [{'id': 'T0001', 'start_sec': 0., 'end_sec': 4., 'text': '研究一种莫名物质'}]
    provider = Provider({'segments': [{'transcript_id': 'T0001', 'edits': [],
                                      'uncertainties': ['莫名物质无法唯一恢复']}]})
    result = proofread_transcript(t, [], provider, source_asr_sha256='a'*64, source_video_sha256='b'*64)
    assert '莫名物质' not in graph_transcript(result)[0]['text']


def test_source_words_allow_small_native_asr_timestamp_disagreement():
    from review.proofreading import proofread_transcript
    t = [{'id': 'T0001', 'start_sec': 5., 'end_sec': 8., 'text': '谷氨酸纳的纳延'}]
    alternate = {'variant': 'asr.secondary.json', 'segments': [
        {'id':'R1', 'start_sec':2., 'end_sec':4., 'text':'谷氨酸钠的钠盐'}]}
    result = proofread_transcript(t, [], Provider({'segments': []}), source_asr_sha256='a'*64,
                                 source_video_sha256='b'*64, alternates=[alternate])
    assert result['segments'][0]['text'] == '谷氨酸钠的钠盐'
    assert (result['segments'][0]['start_sec'],result['segments'][0]['end_sec']) == (5.,8.)


def test_recurrent_source_supported_word_does_not_follow_secondary_nonword():
    from review.proofreading import proofread_transcript
    t = [{'id':'T0001', 'start_sec':0., 'end_sec':4., 'text':'一个缩疾'},
         {'id':'T0002', 'start_sec':12., 'end_sec':16., 'text':'只有一个缩疾了'}]
    alternate = {'variant':'asr.secondary.json','segments':[
        {'id':'R1','start_sec':0.,'end_sec':4.,'text':'一个羧基'},
        {'id':'R2','start_sec':12.,'end_sec':16.,'text':'只有一个缩基了'}]}
    result = proofread_transcript(t, [], Provider({'segments': []}), source_asr_sha256='a'*64,
                                 source_video_sha256='b'*64, alternates=[alternate])
    assert result['segments'][1]['text'] == '只有一个羧基了'
    assert result['segments'][1]['edits'][0]['basis'] == 'recurrent_same_source_lexical_context'


def test_recurrent_greek_word_can_bridge_secondary_sentence_boundary():
    from review.proofreading import proofread_transcript
    t = [{'id':'T0001', 'start_sec':0., 'end_sec':4., 'text':'叫它二发氨基酸'},
         {'id':'T0002', 'start_sec':12., 'end_sec':16., 'text':'说过二发氨基酸'}]
    alternate = {'variant':'asr.secondary.json','segments':[
        {'id':'R1','start_sec':0.,'end_sec':4.,'text':'叫它阿尔法氨基酸'},
        {'id':'R2','start_sec':12.,'end_sec':16.,'text':'尔法氨基酸'}]}
    result = proofread_transcript(t, [], Provider({'segments': []}), source_asr_sha256='a'*64,
                                 source_video_sha256='b'*64, alternates=[alternate])
    assert result['segments'][1]['text'] == '说过阿尔法氨基酸'


def test_greek_context_does_not_allow_numeric_negation_or_formula_changes():
    from review.proofreading import proofread_transcript
    for before,after in [('二发1','阿尔法2'),('不研究也可以研究','研究也可以不研究'),('Na','NA')]:
        t = [{'id':'T0001','start_sec':0.,'end_sec':4.,'text':before}]
        alt = {'segments':[{'id':'R1','start_sec':0.,'end_sec':4.,'text':after}]}
        provider = Provider({'segments':[{'transcript_id':'T0001','edits':[{
            'original':before,'replacement':after,'reason':'模型候选','evidence_ids':['T0001','U1_0001']}],
            'uncertainties':[]}]})
        result = proofread_transcript(t,[],provider,source_asr_sha256='a'*64,source_video_sha256='b'*64,alternates=[alt])
        assert result['segments'][0]['text'] == before


def test_conflicting_mixed_letter_terms_never_feed_graph_from_repeat_or_fragment():
    from review.proofreading import proofread_transcript, graph_transcript
    t = [{'id':'T0001','start_sec':0.,'end_sec':4.,'text':'叫R2瓦碳'},
         {'id':'T0002','start_sec':8.,'end_sec':12.,'text':'咱们叫R2瓦碳'},
         {'id':'T0003','start_sec':16.,'end_sec':20.,'text':'区别在R2基上'}]
    alt = {'variant':'asr.secondary.json','segments':[
        {'id':'R1','start_sec':0.,'end_sec':4.,'text':'叫阿尔法碳'},
        {'id':'R2','start_sec':8.,'end_sec':12.,'text':'咱们叫阿尔法碳'},
        {'id':'R3','start_sec':16.,'end_sec':20.,'text':'区别在r基上'}]}
    result=proofread_transcript(t,[],Provider({'segments':[]}),source_asr_sha256='a'*64,
                               source_video_sha256='b'*64,alternates=[alt])
    assert [row['text'] for row in result['segments']] == [row['text'] for row in t]
    assert all(row['text']=='' for row in graph_transcript(result))
