import sys
import json
from pathlib import Path
from fastapi.testclient import TestClient
import pytest
import requests
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from review.alignment import align,extract_concepts
from review.builder import build_lesson
from review.common import atomic_json,sha256,validate_segments
from review.provider import LLMProvider,ReviewError
from review.retrieval import answer_question,normalize,select_evidence,parse_citations,render_points
from review.server import LessonStore,create_app,byte_range
from review.timebase import validate_media_timebase
from review.blocks import speech_blocks


def records():
    transcript=[{'id':'T0001','start_sec':0.,'end_sec':8.,'text':'討論正弦定理和三角形。','source':'asr'},
                {'id':'T0002','start_sec':12.,'end_sec':18.,'text':'正弦定理的應用。','source':'asr'}]
    slides=[{'id':'S0001','start_sec':0.,'end_sec':5.,'text':'正弦定理 三角形'},
            {'id':'S0002','start_sec':5.,'end_sec':10.,'text':'证明过程'},
            {'id':'S0003','start_sec':12.,'end_sec':20.,'text':'正弦定理 三角形'}]
    return transcript,slides


@pytest.mark.parametrize('query',['总结整节课','Summarize the whole lesson','GIVE ME A LESSON OVERVIEW'])
def test_overview_samples_source_evidence_across_the_whole_lesson(query):
    rows=[{'id':f'T{i:04d}','start_sec':float(i),'end_sec':float(i+1),
           'text':'用于检验整课均匀采样的合成中文转写','source':'asr'} for i in range(36)]
    bundle={'lesson':{'title':'Synthetic source-language lesson','duration_sec':36},'transcript':rows,'slides':[]}
    selected=select_evidence(bundle,query,limit=18)
    assert any(row['id']=='T0035' for row in selected)
    assert len(selected)==18


def test_hidden_archived_lesson_is_not_in_demo_catalog(tmp_path, monkeypatch):
    registry = tmp_path/'registry.json'
    atomic_json(registry, {'chinese': {'directory': '中文', 'demo_visible': True},
                           'english': {'directory': '英文', 'demo_visible': False}})
    store = LessonStore(registry)
    def load(lesson_id):
        return {'lesson': {'id': lesson_id, 'title': '中文课' if lesson_id == 'chinese' else 'English lesson',
                           'duration_sec': 82, 'status': 'complete'}}
    monkeypatch.setattr(store, 'load', load)
    assert [row['id'] for row in store.list_lessons()] == ['chinese']
    assert store.entry('english')['directory'] == str((registry.parent/'英文').resolve())
    assert store.load('english')['lesson']['title'] == 'English lesson'


def test_visual_lesson_requires_explicit_missing_speech_and_never_fills_transcript(tmp_path, monkeypatch):
    from review import builder
    video=tmp_path/'fixture.mp4';video.write_bytes(b'synthetic fixture, not a video')
    image=tmp_path/'screen.jpg';image.write_bytes(b'synthetic image')
    source={'sha256':sha256(video)}
    atomic_json(tmp_path/'asr.json',{'status':'complete','segments':[], 'provenance':{'source_video':source}})
    atomic_json(tmp_path/'slides.json',{'status':'complete','slides':[{'id':'S0001','start_sec':0,'end_sec':10,
        'text':'测试课文证据','title':'测试','image_path':'screen.jpg'}], 'provenance':{'source_video':source}})
    monkeypatch.setattr(builder,'ROOT',tmp_path)
    monkeypatch.setattr(builder,'validate_media_timebase',lambda *args:10)
    monkeypatch.setattr(builder,'extract_concepts',lambda *args:{'nodes':[],'edges':[],'provenance':{'fixture':True}})
    with pytest.raises(ValueError,match='No speech'):
        build_lesson('fixture',video,tmp_path,'测试')
    bundle=build_lesson('fixture',video,tmp_path,'测试',provider=object(),allow_no_speech=True)
    assert bundle['transcript']==[] and bundle['evidence_blocks']==[]
    assert bundle['lesson']['modalities']['audio']=='unrecognized'
    assert any('不补写教师台词' in limit for limit in bundle['report']['limitations'])


def test_retrieval_blocks_keep_native_text_and_never_bridge_long_silence():
    rows=[{'id':'T0001','start_sec':0,'end_sec':2,'text':'原句一'},
          {'id':'T0002','start_sec':2,'end_sec':5,'text':'原句二'},
          {'id':'T0003','start_sec':12,'end_sec':15,'text':'原句三'}]
    blocks=speech_blocks(rows)
    assert len(blocks)==2
    assert blocks[0]['text']=='原句一 原句二' and blocks[0]['source_refs']==['T0001','T0002']
    assert [(b['start_sec'],b['end_sec']) for b in blocks]==[(0,5),(12,15)]
    bundle={'lesson':{'title':'测试','duration_sec':20},'transcript':rows,'slides':[],'evidence_blocks':blocks}
    response=answer_question(bundle,'原句',0,'lesson',FakeProvider({'points':[{'text':'原句说明','evidence_ids':['B0001']}]}))
    assert response['citations'][0]['text']=='原句一 原句二'


class FakeProvider:
    def __init__(self,reply=None):self.reply=reply
    def status(self):return {'source':'test_only','model':'mock'}
    def complete(self,*args,**kwargs):
        return self.reply,{'source':'local_llm','model':'test_only_mock','usage':{}}


def test_literal_verification_and_no_invented_relationships():
    t,s=records()
    provider=FakeProvider({'concepts':[{'label':'正弦定理','evidence_ids':['T0001']},
        {'label':'三角形','evidence_ids':['S0001']},{'label':'未教知识','evidence_ids':['S0001']},
        {'label':'虚构概念','evidence_ids':['MISSING']},{'label':3,'evidence_ids':[]}]})
    graph=extract_concepts(t,s,provider)
    assert [n['label'] for n in graph['nodes']]==['正弦定理','三角形']
    assert graph['nodes'][0]['properties']['source_refs']==['T0001','T0002','S0001','S0003']
    assert {e['label'] for e in graph['edges']}=={'同段出现'}
    assert graph['provenance']['rejected_candidates']==3
    assert normalize('學習三角形')==normalize('学习三角形')
    recovered=extract_concepts(t,s,FakeProvider({'concepts':[{'label':'正弦定理','evidence_ids':['MISSING']}]}))
    assert recovered['nodes'][0]['properties']['source_refs']==['T0001','T0002','S0001','S0003']


def test_concept_extraction_rejects_locative_fragment_and_clipped_unknown_prefix():
    t = [{'id':'P1', 'text':'区别在R基上。阿尔法氨基酸有一个氨基。', 'start_sec':0.,'end_sec':5.}]
    graph = extract_concepts(t, [], FakeProvider({'concepts':[
        {'label':'基上','evidence_ids':['P1']},
        {'label':'阿尔法氨基','evidence_ids':['P1']},
        {'label':'阿尔法氨基酸','evidence_ids':['P1']},
        {'label':'氨基','evidence_ids':['P1']}]}))
    assert [n['label'] for n in graph['nodes']] == ['阿尔法氨基酸','氨基']


def test_shared_clock_preserves_slide_boundary_and_disjoint_occurrences():
    t,s=records();graph={'nodes':[{'id':'K1','properties':{'source_refs':['S0001','S0003']}}]}
    rows=align(t,s,graph)
    assert [(r['start_sec'],r['end_sec'],r['slide_id']) for r in rows]==[(0,5,'S0001'),(5,8,'S0002'),(12,18,'S0003')]
    assert all(r['score'] is None for r in rows)
    assert rows[0]['concept_ids']==['K1'] and rows[1]['concept_ids']==[]


def test_out_of_source_times_duplicate_ids_and_unmatched_are_rejected_or_unknown():
    t,s=records()
    with pytest.raises(ValueError):validate_segments(t+t,20,'transcript')
    with pytest.raises(ValueError):validate_segments([{**t[0],'end_sec':float('nan')}],20,'transcript')
    unmatched=align(t,[],{'nodes':[]})
    assert all(r['slide_id'] is None and r['status']=='no_slide_evidence' for r in unmatched)
    pieces=align([{**t[0],'start_sec':0,'end_sec':20}],[{**s[0],'start_sec':5,'end_sec':10}],{'nodes':[]})
    assert [(r['start_sec'],r['end_sec'],r['slide_id']) for r in pieces]==[(0,5,None),(5,10,'S0001'),(10,20,None)]


def test_timebase_rejects_shifted_or_incomplete_extractions():
    import copy
    src={'duration_sec':20,'audio_duration_sec':20,'video_duration_sec':19.5,'size_bytes':100,
         'audio_start_sec':0,'video_start_sec':0}
    asr={'provenance':{'source_video':src,'timestamp_offset_sec':0,'config':{'audio_time_offset_sec':0},
         'generator_exhausted':True,'processed_range_sec':[0,20],'audio_duration_sec':20},'coverage':{'processed_duration_sec':20}}
    pages={'provenance':{'source_video':copy.deepcopy(src),'timestamp_offset_sec':0,'config':{'timestamp_offset_sec':0,'sample_interval_sec':5},
           'observed_video_range_sec':[0,19.5]},'coverage':{'video_duration_sec':19.5,'sample_count':5,'expected_sample_count':5,
           'first_sample_sec':0,'last_sample_sec':19.4,'max_sampling_gap_sec':5}}
    assert validate_media_timebase(asr,pages)==20
    for key,value in [('timestamp_offset_sec',1),('generator_exhausted',False),('processed_range_sec',[0,10])]:
        altered=copy.deepcopy(asr);altered['provenance'][key]=value
        with pytest.raises(ValueError):validate_media_timebase(altered,pages)
    altered=copy.deepcopy(pages);altered['provenance']['source_video']['audio_start_sec']=3
    with pytest.raises(ValueError):validate_media_timebase(asr,altered)


def test_chat_citations_are_server_evidence_not_client_supplied():
    t,s=records();bundle={'lesson':{'title':'测试','duration_sec':20},'transcript':t,'slides':s}
    good=answer_question(bundle,'本课讲什么',0,'lesson',FakeProvider({'points':[{'text':'本课讨论正弦定理。','evidence_ids':['T0001']}]}))
    assert good['citations'][0]['text']==t[0]['text']
    assert good['citations'][0]['start_sec']==0
    for bad in [{'points':[{'text':'没有引用','evidence_ids':[]}]},{'points':[{'text':'编造事实','evidence_ids':['T9999']}]}]:
        with pytest.raises(ReviewError,match='引用'):answer_question(bundle,'本课讲什么',0,'lesson',FakeProvider(bad))
    with pytest.raises(ReviewError):answer_question(bundle,'内容',float('nan'),'lesson',FakeProvider(''))
    with pytest.raises(ReviewError):answer_question(bundle,'内容',0,'other',FakeProvider(''))
    assert parse_citations('解释 [ T0001, S0001 ]',{'T0001':{},'S0001':{}})==('解释 [T0001][S0001]',['T0001','S0001'])
    with pytest.raises(ValueError):parse_citations('有效[T0001] 编造[ T9999]',{'T0001':{}})
    with pytest.raises(ValueError):render_points({'points':[{'text':'隐藏伪引用 [ T9999]','evidence_ids':['T0001']}]},{'T0001':{}})


def test_media_source_mismatch_and_partial_cannot_be_built(tmp_path):
    video=tmp_path/'lesson.mp4';video.write_bytes(b'not a real video: contract fixture only')
    atomic_json(tmp_path/'asr.json',{'status':'running'})
    atomic_json(tmp_path/'slides.json',{'status':'complete'})
    with pytest.raises(ValueError,match='incomplete'):build_lesson('test',video,tmp_path,'test')
    atomic_json(tmp_path/'asr.json',{'status':'complete','provenance':{'source_video':{'sha256':'wrong'}}})
    with pytest.raises(ValueError,match='different sources'):build_lesson('test',video,tmp_path,'test')


@pytest.mark.parametrize('header,expected',[(None,(0,99,False)),('bytes=0-1',(0,1,True)),('bytes=90-',(90,99,True)),('bytes=-10',(90,99,True))])
def test_ranges(header,expected):assert byte_range(header,100)==expected


@pytest.mark.parametrize('header',['bytes=100-','bytes=8-2','bytes=-0','bytes=0-1,3-5','nonsense'])
def test_invalid_ranges(header):
    with pytest.raises(ReviewError):byte_range(header,100)


def test_local_http_contract_range_path_errors_and_provider_failure(tmp_path):
    t,s=records();video=tmp_path/'video.mp4';video.write_bytes(b'0123456789')
    bundle={'schema_version':'3.0','data_source':'lesson_review','lesson':{'id':'test','title':'测试','duration_sec':20,'status':'complete'},'transcript':t,'slides':s}
    atomic_json(tmp_path/'bundle.json',bundle)
    registry=tmp_path/'registry.json';atomic_json(registry,{'test':{'directory':str(tmp_path),'source_video':str(video)}})
    class FailedProvider(FakeProvider):
        def complete(self,*args,**kwargs):raise ReviewError('model_unavailable','模型未就绪')
    store=LessonStore(registry)
    with pytest.raises(ReviewError):store.media('test','../private.json')
    with TestClient(create_app(store=store,provider=FailedProvider())) as client:
        assert client.get('/api/review/lessons').json()['lessons'][0]['id']=='test'
        r=client.get('/api/review/media/test/source.mp4',headers={'Range':'bytes=2-5'})
        assert r.status_code==206 and r.content==b'2345' and r.headers['Content-Range']=='bytes 2-5/10'
        assert r.headers['Cache-Control']=='no-store'
        r=client.post('/api/review/chat',json={'lesson_id':'test','query':'讲什么','at_sec':0})
        assert r.status_code==503 and r.json()['error']['code']=='model_unavailable' and 'answer' not in r.json()
        assert client.post('/api/review/chat',json=[]).status_code==400


def test_provider_never_falls_back_on_network_failure(monkeypatch,tmp_path):
    from review import model_runtime
    monkeypatch.setattr(model_runtime,'ROOT',tmp_path)
    def fail(*args,**kwargs):raise requests.ConnectionError('not a real provider')
    monkeypatch.setattr(requests,'post',fail)
    with pytest.raises(ReviewError,match='未就绪'):LLMProvider().complete('system','prompt')


def test_provider_rejects_truncated_answer_even_with_valid_citation(monkeypatch,tmp_path):
    from review import model_runtime
    monkeypatch.setattr(model_runtime,'ROOT',tmp_path)
    class Response:
        ok=True
        def json(self):return {'choices':[{'finish_reason':'length','message':{'content':'这是截断回答[T0001]'}}]}
    monkeypatch.setattr(requests,'post',lambda *a,**kw:Response())
    with pytest.raises(ReviewError,match='未正常完成'):LLMProvider().complete('system','prompt')
