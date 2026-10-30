"""Model switching must not evict a foreign process or interrupt inference."""
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from review import model_runtime as runtime
from review.provider import ReviewError


def test_session_is_reentrant_and_excludes_other_threads(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime, 'ROOT', tmp_path)
    with runtime.model_session():
        with runtime.model_session():
            with ThreadPoolExecutor(max_workers=1) as pool:
                def competing_request():
                    with runtime.model_session():
                        pytest.fail('Concurrent request entered an owned GPU session')
                with pytest.raises(ReviewError) as error:
                    pool.submit(competing_request).result()
                assert error.value.code == 'model_busy'
    with runtime.model_session():
        pass


def test_session_releases_after_exception(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime, 'ROOT', tmp_path)
    with pytest.raises(RuntimeError):
        with runtime.model_session():
            raise RuntimeError('generation failed')
    with runtime.model_session():
        pass


def test_switch_refuses_foreign_model_before_any_stop(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime, 'ROOT', tmp_path)
    profiles = {'configs/correction_llm.json': {'qa_profile': 'configs/local_llm.json', 'model_alias': 'cec'},
                'configs/local_llm.json': {'model_alias': 'qa'}}
    monkeypatch.setattr(runtime, '_profile', lambda path: (path, profiles[path]))
    def foreign(config):
        raise ReviewError('model_port_conflict', 'Foreign port')
    monkeypatch.setattr(runtime, '_ready', foreign)
    monkeypatch.setattr(runtime, '_invoke', lambda *a, **k: pytest.fail('Stopped foreign model'))
    with pytest.raises(ReviewError) as error:
        runtime.ensure_role('correction')
    assert error.value.code == 'model_port_conflict'


def test_switch_requires_old_role_stopped_before_start(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime, 'ROOT', tmp_path)
    profiles = {'configs/correction_llm.json': {'qa_profile': 'configs/local_llm.json', 'model_alias': 'cec'},
                'configs/local_llm.json': {'model_alias': 'qa'}}
    monkeypatch.setattr(runtime, '_profile', lambda path: (path, profiles[path]))
    monkeypatch.setattr(runtime, '_ready', lambda config: True)
    calls=[]
    monkeypatch.setattr(runtime, '_invoke', lambda path, **kw: calls.append((path,kw)))
    with pytest.raises(ReviewError) as error:
        runtime.ensure_role('correction')
    assert error.value.code == 'model_ownership_unknown'
    assert calls == [('configs/local_llm.json', {'stop':True})]


def test_conflicting_target_does_not_unload_qa(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime, 'ROOT', tmp_path)
    profiles = {'configs/correction_llm.json': {'qa_profile':'configs/local_llm.json','model_alias':'cec'},
                'configs/local_llm.json': {'model_alias':'qa','base_url':'http://127.0.0.1:18081/v1'}}
    monkeypatch.setattr(runtime, '_profile', lambda path: (tmp_path/path, profiles[path]))
    def ready(config):
        if config['model_alias']=='cec':
            raise ReviewError('model_port_conflict','Foreign target')
        return True
    monkeypatch.setattr(runtime, '_ready', ready)
    monkeypatch.setattr(runtime, '_invoke', lambda *a, **k: pytest.fail('Healthy QA unloaded'))
    with pytest.raises(ReviewError) as error:
        runtime.ensure_role('correction')
    assert error.value.code=='model_port_conflict'
    recovered=runtime.ensure_role('qa')
    assert recovered['model']=='qa'
    assert recovered['other_management']=='foreign_service_unmodified'


def test_lifecycle_timeout_is_reported(monkeypatch):
    import subprocess
    # Select only the runtime's Windows branch; leave the host OS unchanged.
    monkeypatch.setattr(runtime, 'os', SimpleNamespace(name='nt'))
    monkeypatch.setattr(runtime.shutil, 'which', lambda name: 'pwsh')
    def timeout(*a, **k):
        raise subprocess.TimeoutExpired('pwsh', 180)
    monkeypatch.setattr(runtime.subprocess, 'run', timeout)
    with pytest.raises(ReviewError) as error:
        runtime._invoke('configs/correction_llm.json')
    assert error.value.code == 'model_lifecycle_timeout'


def test_non_windows_lifecycle_is_rejected_without_starting_a_shell(monkeypatch):
    monkeypatch.setattr(runtime, 'os', SimpleNamespace(name='posix'))
    monkeypatch.setattr(runtime.subprocess, 'run',
                        lambda *a, **k: pytest.fail('Started a Windows lifecycle command on POSIX'))
    with pytest.raises(ReviewError) as error:
        runtime._invoke('configs/correction_llm.json')
    assert error.value.code == 'unsupported_model_lifecycle'


def test_unrelated_http_service_is_conflict_even_without_valid_health(monkeypatch):
    def get(url, **kwargs):
        return SimpleNamespace(ok=False, json=lambda:{})
    monkeypatch.setattr(runtime.requests,'get',get)
    with pytest.raises(ReviewError) as error:
        runtime._ready({'base_url':'http://127.0.0.1:18082/v1','model_alias':'classroom-corrector'})
    assert error.value.code=='model_port_conflict'


@pytest.mark.parametrize('data',[[],{'data':None},{'data':[None]}])
def test_malformed_model_inventory_is_safe_conflict(monkeypatch,data):
    monkeypatch.setattr(runtime.requests,'get',lambda *a,**k:SimpleNamespace(ok=True,json=lambda:data))
    with pytest.raises(ReviewError) as error:
        runtime._ready({'base_url':'http://127.0.0.1:18082/v1','model_alias':'classroom-corrector'})
    assert error.value.code=='model_port_conflict'


def test_new_correction_profile_unloads_both_other_owned_roles(tmp_path,monkeypatch):
    from review.common import atomic_json
    monkeypatch.setattr(runtime,'ROOT',tmp_path)
    for name,alias,port in [('qwen35','q35',18083),('qa','qa',18081),('cec','cec',18082)]:
        atomic_json(tmp_path/f'{name}.json',{'server':{'host':'127.0.0.1'},
            'base_url':f'http://127.0.0.1:{port}/v1','model_alias':alias,
            'qa_profile':'qa.json','exclusive_profiles':['qa.json','cec.json'] if name=='qwen35' else []})
    running={'q35':False,'qa':True,'cec':False};events=[]
    monkeypatch.setattr(runtime,'_ready',lambda config:running[config['model_alias']])
    def invoke(path,stop=False):
        config=runtime._profile(path)[1]
        events.append((config['model_alias'],stop))
        running[config['model_alias']]=not stop
    monkeypatch.setattr(runtime,'_invoke',invoke)
    result=runtime.ensure_role('correction','qwen35.json')
    assert result['model']=='q35'
    assert events==[('qa',True),('cec',True),('q35',False)]
    events.clear()
    runtime.ensure_role('qa','qwen35.json')
    assert events==[('q35',True),('cec',True),('qa',False)]
    assert running=={'q35':False,'qa':True,'cec':False}


def test_foreign_third_profile_is_preflighted_before_unloading_qa(tmp_path,monkeypatch):
    from review.common import atomic_json
    monkeypatch.setattr(runtime,'ROOT',tmp_path)
    for name,alias in [('q35','q35'),('qa','qa'),('cec','foreign')]:
        atomic_json(tmp_path/f'{name}.json',{'server':{'host':'127.0.0.1'},'model_alias':alias,
            'base_url':'http://127.0.0.1:18081/v1','qa_profile':'qa.json',
            'exclusive_profiles':['cec.json'] if name=='q35' else []})
    def ready(config):
        if config['model_alias']=='foreign':
            raise ReviewError('model_port_conflict','Foreign third service')
        return config['model_alias']=='qa'
    monkeypatch.setattr(runtime,'_ready',ready)
    monkeypatch.setattr(runtime,'_invoke',lambda *a,**k:pytest.fail('QA stopped before all ports validated'))
    with pytest.raises(ReviewError) as error:
        runtime.ensure_role('correction','q35.json')
    assert error.value.code=='model_port_conflict'
