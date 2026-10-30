"""Correction requests are isolated from QA and keep model evidence for audit."""
import json
from contextlib import nullcontext

import pytest
import requests

from review.provider import ReviewError


@pytest.fixture(autouse=True)
def isolated_model_session(tmp_path,monkeypatch):
    from review import model_runtime
    monkeypatch.setattr(model_runtime,'ROOT',tmp_path)


def config(tmp_path):
    path = tmp_path/'correction.json'
    path.write_text(json.dumps({'base_url': 'http://127.0.0.1:18082/v1',
        'model_alias': 'classroom-corrector', 'timeout_seconds': 180,
        'model': {'sha256': 'a'*64}, 'request_defaults': {'temperature': 0.1,
        'chat_template_kwargs': {'enable_thinking': False}}}), encoding='utf-8')
    return path


class Response:
    ok = True
    def __init__(self, text, finish='stop'):
        self.text, self.finish = text, finish
    def json(self):
        return {'id': 'test', 'model': 'classroom-corrector', 'usage': {'total_tokens': 10},
            'choices': [{'finish_reason': self.finish, 'message': {'content': self.text}}]}


def test_dedicated_correction_config_does_not_inherit_qa_environment(tmp_path, monkeypatch):
    from review.correction_provider import CorrectionProvider
    monkeypatch.setenv('REVIEW_LLM_BASE_URL', 'http://127.0.0.1:18081/v1')
    monkeypatch.setenv('REVIEW_LLM_MODEL', 'classroom-local')
    posted = []
    def post(url, **kwargs):
        posted.append((url, kwargs)); return Response('{"segments":[]}')
    monkeypatch.setattr(requests, 'post', post)
    provider = CorrectionProvider(config_path=config(tmp_path))
    answer, metadata = provider.complete('system', 'prompt', json_mode=True)
    assert answer == {'segments': []}
    assert posted[0][0] == 'http://127.0.0.1:18082/v1/chat/completions'
    assert posted[0][1]['json']['model'] == 'classroom-corrector'
    assert metadata['raw_response'] == '{"segments":[]}'
    assert metadata['request_parameters']['temperature'] == .1
    assert provider.status()['model_sha256'] == 'a'*64


def test_native_correct_keeps_thinking_and_plain_response_separately(tmp_path, monkeypatch):
    from review.correction_provider import CorrectionProvider
    posted = []
    def post(url, **kwargs):
        posted.append(kwargs['json']); return Response('<think>候选推理</think>有一个羧基')
    monkeypatch.setattr(requests, 'post', post)
    provider = CorrectionProvider(config_path=config(tmp_path))
    text, metadata = provider.native_correct('有一个缩疾')
    assert text == '有一个羧基'
    assert metadata['raw_response'] == '<think>候选推理</think>有一个羧基'
    assert metadata['reasoning_text'] == '候选推理'
    assert 'response_format' not in posted[0]
    assert metadata['track'] == 'native_plain_text'


@pytest.mark.parametrize('text,finish,code', [('不是JSON', 'stop', 'invalid_model_json'),
    ('{"segments":[]}', 'length', 'model_incomplete')])
def test_rejected_model_response_keeps_private_diagnostic(tmp_path, monkeypatch, text, finish, code):
    from review.correction_provider import CorrectionProvider
    monkeypatch.setattr(requests, 'post', lambda *a, **k: Response(text, finish))
    with pytest.raises(ReviewError) as error:
        CorrectionProvider(config_path=config(tmp_path)).complete('system', 'prompt', json_mode=True)
    assert error.value.code == code
    assert error.value.diagnostic['raw_response'] == text


def test_managed_lifecycle_is_lazy_and_restores_qa_once(tmp_path,monkeypatch):
    import sys
    from types import SimpleNamespace
    from review.correction_provider import CorrectionProvider
    calls=[]
    monkeypatch.setitem(sys.modules,'review.model_runtime',SimpleNamespace(
        ensure_role=lambda role,**kwargs:calls.append((role,kwargs)), model_session=nullcontext))
    monkeypatch.setattr(requests,'post',lambda *a,**k:Response('{"segments":[]}'))
    provider=CorrectionProvider(config_path=config(tmp_path),managed_lifecycle=True)
    provider.status(); provider.close()
    assert calls == []
    provider.complete('system','prompt',json_mode=True)
    provider.complete('system','prompt',json_mode=True)
    provider.close(); provider.close()
    assert [r[0] for r in calls] == ['correction','qa']


def test_failed_correction_start_still_restores_qa(tmp_path,monkeypatch):
    import sys
    from types import SimpleNamespace
    from review.correction_provider import CorrectionProvider
    calls=[]
    def ensure(role,**kwargs):
        calls.append(role)
        if role=='correction':
            raise ReviewError('model_lifecycle_failed','startup failed')
    monkeypatch.setitem(sys.modules,'review.model_runtime',SimpleNamespace(ensure_role=ensure,model_session=nullcontext))
    provider=CorrectionProvider(config_path=config(tmp_path),managed_lifecycle=True)
    with pytest.raises(ReviewError):
        provider.complete('system','prompt')
    provider.close()
    assert calls == ['correction','qa']


def test_repeated_failed_switch_keeps_outer_session_until_close(tmp_path,monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from review import model_runtime
    from review.correction_provider import CorrectionProvider
    def ensure(role,**kwargs):
        if role=='correction':
            raise ReviewError('model_lifecycle_failed','startup failed')
    monkeypatch.setattr(model_runtime,'ensure_role',ensure)
    provider=CorrectionProvider(config_path=config(tmp_path),managed_lifecycle=True)
    for _ in range(3):
        with pytest.raises(ReviewError):
            provider.complete('system','prompt')
        def concurrent():
            with model_runtime.model_session():
                return 'entered'
        with ThreadPoolExecutor(max_workers=1) as pool:
            with pytest.raises(ReviewError) as error:
                pool.submit(concurrent).result()
            assert error.value.code=='model_busy'
    provider.close()
    with model_runtime.model_session():
        pass
