"""Teaching workflow through real ASGI routes and a temporary SQLite database."""
from fastapi.testclient import TestClient
from test_fastapi_server import Provider,store_fixture
from review.server import create_app


def test_teacher_cycle_is_persisted_and_raw_bundle_is_unchanged(tmp_path):
    store=store_fixture(tmp_path);raw=(tmp_path/'bundle.json').read_bytes()
    base='/api/review/lessons/test/workspace'
    with TestClient(create_app(store=store,provider=Provider())) as client:
        response=client.get(base)
        assert response.status_code==200, 'The teaching workspace API is missing.'
        assert response.json()['version']==0
        w=client.post(base+'/goals',json={'text':'核对课堂证据','expected_version':0}).json()
        assert w['goals']=='核对课堂证据' and w['version']==1
        assert client.post(base+'/goals',json={'text':'stale','expected_version':0}).status_code==409
        w=client.post(base+'/corrections',json={'transcript_id':'T0001','replacement':'教师确认文字',
                      'reason':'已回听','expected_version':1}).json()
        assert w['version']==2
        effective=client.get('/api/review/lessons/test').json()
        assert effective['transcript'][0]['text']=='测试内容'
        assert effective['teacher_transcript'][0]['text']=='教师确认文字'
        found=client.get('/api/review/lessons/test/evidence/search',params={'q':'教师确认'}).json()['evidence']
        assert found[0]['text']=='教师确认文字'
        w=client.post(base+'/actions',json={'title':'下次课复查','evidence_ids':['T0001'],'expected_version':2}).json()
        action=w['actions'][0]['id']
        w=client.patch(base+'/actions/'+action,json={'status':'done','outcome':'已实施复查','expected_version':3}).json()
        assert w['actions'][0]['status']=='done'
        w=client.post(base+'/assessments',json={'question':'课中讲了什么？','reference_answer':'教师确认文字',
                      'evidence_ids':[found[0]['id']],'expected_version':4}).json()
        exercise=w['assessments'][0]['id']
        w=client.post(base+'/assessments/'+exercise+'/attempts',json={'answer':'实际作答','feedback':'待核对',
                      'score':None,'expected_version':5}).json()
        assert w['attempts'][0]['score'] is None
        assert client.delete(base+'/corrections/T0001',params={'expected_version':6}).status_code==200
        assert 'teacher_transcript' not in client.get('/api/review/lessons/test').json()
        assert client.get('/api/review/tasks',params={'lesson_id':'test'}).json()=={'tasks':[]}
        assert client.get('/api/review/health').json()['storage']=='sqlite'
        runtime=client.get('/api/review/runtime')
        assert runtime.status_code==200 and isinstance(runtime.json()['profiles'],list)
    with TestClient(create_app(store=store,provider=Provider())) as client:
        assert client.get(base).json()['attempts'][0]['answer']=='实际作答'
    assert (tmp_path/'bundle.json').read_bytes()==raw


def test_new_mutations_preserve_same_origin_and_request_bounds(tmp_path):
    with TestClient(create_app(store=store_fixture(tmp_path),provider=Provider())) as client:
        path='/api/review/lessons/test/workspace/actions/missing'
        denied=client.patch(path,json={'status':'done','outcome':'x','expected_version':0},
                            headers={'Origin':'https://example.com'})
        assert denied.status_code==403
        oversized=client.patch(path,content=b'x'*65537)
        assert oversized.status_code==413
