"""One lesson switches correction -> QA without mutating source evidence."""
import json
import sys
from contextlib import nullcontext
from types import SimpleNamespace

from review.common import atomic_json, sha256


def test_default_build_restores_qa_before_graph_and_cache_does_not_load_corrector(tmp_path,monkeypatch):
    from review import builder
    from review.correction_provider import CorrectionProvider
    video=tmp_path/'source.mp4'; video.write_bytes(b'synthetic source')
    (tmp_path/'slide.jpg').write_bytes(b'synthetic image')
    source={'sha256':sha256(video)}
    atomic_json(tmp_path/'asr.json',{'status':'complete','provenance':{'source_video':source},
        'segments':[{'id':'T0001','start_sec':0.,'end_sec':2.,'text':'认识氨基酸'}]})
    atomic_json(tmp_path/'slides.json',{'status':'complete','provenance':{'source_video':source},
        'slides':[{'id':'S0001','start_sec':0.,'end_sec':2.,'text':'认识氨基酸','image_path':'slide.jpg'}]})
    original=(tmp_path/'asr.json').read_bytes()
    config=tmp_path/'correction.json'
    atomic_json(config,{'managed_lifecycle':True,'base_url':'http://127.0.0.1:18082/v1',
        'model_alias':'classroom-corrector'})
    events=[]
    monkeypatch.setitem(sys.modules,'review.model_runtime',SimpleNamespace(
        ensure_role=lambda role,**kw:events.append(role), model_session=nullcontext))
    def post(url,**kwargs):
        model=kwargs['json']['model']; events.append(model)
        if model == 'classroom-local':
            assert events[-2]=='qa'
            content={'concepts':[{'label':'氨基酸','evidence_ids':['S0001']}]}
        else:
            content={'segments':[]}
        return SimpleNamespace(ok=True,json=lambda:{'model':model,'choices':[
            {'finish_reason':'stop','message':{'content':json.dumps(content)}}]})
    monkeypatch.setattr('requests.post',post)
    monkeypatch.setattr(builder,'CorrectionProvider',lambda:CorrectionProvider(config_path=config))
    monkeypatch.setattr(builder,'validate_media_timebase',lambda *a:2.)
    monkeypatch.setattr(builder,'ROOT',tmp_path)
    first=builder.build_lesson('fixture',video,tmp_path,'Fixture')
    assert events==['correction','classroom-corrector','qa','classroom-local']
    assert first['transcript_proofreading']['model']=='classroom-corrector'
    assert first['graph']['provenance']['model']=='classroom-local'
    assert first['proofread_evidence_blocks'][0]['source_refs']==['P0001','T0001']
    events.clear()
    second=builder.build_lesson('fixture',video,tmp_path,'Fixture')
    assert events==[]
    assert second['transcript_proofreading']==first['transcript_proofreading']
    assert (tmp_path/'asr.json').read_bytes()==original
