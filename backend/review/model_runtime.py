"""Identity-checked sequential local model switching for the 8GB demo host.

Only explicitly configured project profiles are managed. Remote services
and unrelated processes are never stopped. The OS lease serializes switches.
"""
from pathlib import Path
from contextlib import contextmanager
import os
import shutil
import subprocess
import threading
import tempfile

import requests

from .common import ROOT, read_json
from .mutation import LessonMutationLock
from .provider import ReviewError


_session = threading.local()


@contextmanager
def model_session():
    """Hold the OS lease through inference; nested calls in one thread reuse it."""
    depth = getattr(_session, 'depth', 0)
    lease = None
    if depth == 0:
        lease = LessonMutationLock(ROOT/'artifacts/runtime/model-session')
        try:
            lease.acquire()
        except ReviewError as exc:
            raise ReviewError('model_busy', '本机模型正在处理另一项任务，请稍后重试。', 409) from exc
    _session.depth = depth + 1
    try:
        yield
    finally:
        _session.depth = depth
        if lease is not None:
            lease.release()


def _profile(path):
    path = Path(path)
    path = path.resolve() if path.is_absolute() else (ROOT/path).resolve()
    if not path.is_relative_to(ROOT):
        raise ReviewError('invalid_model_profile', '模型配置必须位于本项目内。', 409)
    config = read_json(path)
    if config['server']['host'] not in ('127.0.0.1', 'localhost'):
        raise ReviewError('remote_model_management', '自动切换仅管理本机明确配置的模型。', 409)
    return path, config


def _ready(config):
    try:
        health = requests.get(config['base_url'].removesuffix('/v1')+'/health', timeout=2)
    except requests.RequestException:
        return False
    try:
        models = requests.get(config['base_url']+'/models', timeout=2)
        if not models.ok or config['model_alias'] not in [r['id'] for r in models.json().get('data', [])]:
            raise ReviewError('model_port_conflict', '模型端口由其它服务占用，未停止该服务。', 409)
        return health.ok and health.json().get('status') == 'ok'
    except (requests.RequestException, ValueError, KeyError, TypeError, AttributeError):
        raise ReviewError('model_port_conflict', '模型端口未能确认所属模型，未停止该服务。', 409)


def _invoke(path, stop=False):
    if os.name != 'nt':
        raise ReviewError('unsupported_model_lifecycle', '此配置的自动切换使用Windows运行时，请配置独立模型服务。', 503)
    shell = shutil.which('pwsh') or shutil.which('powershell')
    if not shell:
        raise ReviewError('model_shell_unavailable', '本机PowerShell不可用，模型未启动。', 503)
    command = [shell, '-NoLogo', '-NoProfile', '-File', str(ROOT/'scripts/start_local_llm.ps1'),
               '-ConfigPath', str(path)]
    if stop:
        command.append('-Stop')
    try:
        # Detached Windows children can inherit pipe handles after pwsh exits;
        # real files let wait() finish without waiting for llama-server EOF.
        with tempfile.TemporaryFile(mode='w+t', encoding='utf-8', errors='replace') as out, tempfile.TemporaryFile(mode='w+t', encoding='utf-8', errors='replace') as err:
            result = subprocess.run(command, stdin=subprocess.DEVNULL, stdout=out, stderr=err, timeout=180)
            out.seek(0)
            output = out.read().strip()
    except subprocess.TimeoutExpired as exc:
        raise ReviewError('model_lifecycle_timeout', '本机模型切换超时，请查看对应模型运行日志。', 504) from exc
    if result.returncode:
        raise ReviewError('model_lifecycle_failed', '本机模型切换失败，请查看对应模型运行日志。', 503)
    return output


def ensure_role(role, profile_path=None):
    if role not in ('correction', 'qa'):
        raise ValueError('Unsupported model role')
    correction_path, correction = _profile(profile_path or 'configs/correction_llm.json')
    qa_path, qa = _profile(correction.get('qa_profile', 'configs/local_llm.json'))
    target_path, target = (correction_path, correction) if role == 'correction' else (qa_path, qa)
    configured=[(correction_path,correction),(qa_path,qa)]
    for extra in correction.get('exclusive_profiles',[]):
        candidate=_profile(extra)
        if not any(path==candidate[0] for path,_ in configured):
            configured.append(candidate)
    others=[(path,config) for path,config in configured if path!=target_path]
    with model_session():
        # Preflight both ports before unloading a healthy role. A conflicting
        # target must never evict QA and then also prevent its recovery.
        target_ready = _ready(target)
        try:
            for _,other in others:
                _ready(other)
        except ReviewError:
            if target_ready:
                return {'role': role, 'model': target['model_alias'], 'base_url': target['base_url'],
                        'profile': str(target_path.relative_to(ROOT)), 'sequential': True,
                        'other_management': 'foreign_service_unmodified'}
            raise
        for other_path,other in others:
            _invoke(other_path, stop=True)
            if _ready(other):
                raise ReviewError('model_ownership_unknown', '另一模型服务未由本项目停止，暂不同时加载模型。', 409)
        if not _ready(target):
            _invoke(target_path)
        if not _ready(target):
            raise ReviewError('model_not_ready', '目标模型尚未就绪。', 503)
    return {'role': role, 'model': target['model_alias'], 'base_url': target['base_url'],
            'profile': str(target_path.relative_to(ROOT)), 'sequential': True}
