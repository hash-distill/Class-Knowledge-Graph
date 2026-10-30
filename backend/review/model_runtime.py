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
            raise ReviewError('model_busy', 'The local model is processing another task. Please try again shortly.', 409) from exc
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
        raise ReviewError('invalid_model_profile', 'The model configuration must be inside this project.', 409)
    config = read_json(path)
    if config['server']['host'] not in ('127.0.0.1', 'localhost'):
        raise ReviewError('remote_model_management', 'Automatic switching manages only explicitly configured local models.', 409)
    return path, config


def _ready(config):
    try:
        health = requests.get(config['base_url'].removesuffix('/v1')+'/health', timeout=2)
    except requests.RequestException:
        return False
    try:
        models = requests.get(config['base_url']+'/models', timeout=2)
        if not models.ok or config['model_alias'] not in [r['id'] for r in models.json().get('data', [])]:
            raise ReviewError('model_port_conflict', 'Another service is using the model port. That service was left running.', 409)
        return health.ok and health.json().get('status') == 'ok'
    except (requests.RequestException, ValueError, KeyError, TypeError, AttributeError):
        raise ReviewError('model_port_conflict', 'The model using this port could not be verified. The service was left running.', 409)


def _invoke(path, stop=False):
    if os.name != 'nt':
        raise ReviewError('unsupported_model_lifecycle', 'Automatic switching for this configuration requires the Windows runtime. Configure a separate model service.', 503)
    shell = shutil.which('pwsh') or shutil.which('powershell')
    if not shell:
        raise ReviewError('model_shell_unavailable', 'Local PowerShell is unavailable. The model was not started.', 503)
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
        raise ReviewError('model_lifecycle_timeout', 'Local model switching timed out. Check the model runtime logs.', 504) from exc
    if result.returncode:
        raise ReviewError('model_lifecycle_failed', 'Local model switching failed. Check the model runtime logs.', 503)
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
                raise ReviewError('model_ownership_unknown', 'The other model service was not stopped by this project. Concurrent model loading is deferred.', 409)
        if not _ready(target):
            _invoke(target_path)
        if not _ready(target):
            raise ReviewError('model_not_ready', 'The target model is not ready.', 503)
    return {'role': role, 'model': target['model_alias'], 'base_url': target['base_url'],
            'profile': str(target_path.relative_to(ROOT)), 'sequential': True}
