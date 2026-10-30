"""Read-only local asset inventory. Never imports or initializes model runtimes."""
from __future__ import annotations

import json
import os
from pathlib import Path, PureWindowsPath

from .common import ROOT


def _inside(root, value):
    """Accept only project-relative, non-escaping paths; return no host path in JSON."""
    if not isinstance(value, str) or not value or PureWindowsPath(value).is_absolute():
        return None
    relative = Path(value)
    if relative.is_absolute() or '..' in relative.parts or ':' in value:
        return None
    candidate = (root / relative).resolve()
    return candidate if candidate.is_relative_to(root) else None


def _read(path):
    try:
        value = json.loads(path.read_text(encoding='utf-8-sig'))
        return value if isinstance(value, dict) else None
    except (OSError, ValueError, TypeError):
        return None


def _item(root, spec, config):
    spec = spec if isinstance(spec, dict) else {}
    value = spec.get('path')
    if spec.get('config_key'):
        value = config
        keys = spec['config_key'].split('.') if isinstance(spec['config_key'], str) else []
        for key in keys:
            value = value.get(key) if isinstance(value, dict) else None
        if isinstance(value, str) and isinstance(spec.get('append'), str):
            value = str(Path(value) / spec['append'])
    path = _inside(root, value)
    present = False
    try:
        if path:
            present = (path.is_dir() and any(path.iterdir())) if spec.get('kind') == 'directory' else path.is_file() and path.stat().st_size > 0
    except OSError:
        pass
    return {'path': path.relative_to(root).as_posix() if path else None,
            'status': 'present' if present else 'missing', 'integrity': 'not_verified'}


def inspect_runtime(project_root=None, profiles_path=None):
    """Return a redacted, no-inference inventory suitable for GET /api/review/runtime.

    Files are checked under ``REVIEW_RUNTIME_ROOT`` (or the source project). A
    packaged desktop app seeds its public configs there, while models remain an
    independently installed component. No network request or GPU discovery runs.
    ``available`` is intentionally false until a separate inference health check
    is implemented: file presence does not prove inference correctness.
    """
    root = Path(project_root or os.getenv('REVIEW_RUNTIME_ROOT') or ROOT).resolve()
    manifest = Path(profiles_path or os.getenv('REVIEW_RUNTIME_PROFILES') or root / 'configs/runtime_profiles.json')
    document = _read(manifest)
    profiles = (document or {}).get('profiles')
    valid_manifest = isinstance(profiles, list)
    result = {'schema_version': '1.0', 'inspection': 'local_files_only',
              'model_loading': False, 'network_access': False, 'gpu_status': 'not_checked',
              'profiles_status': 'present' if valid_manifest else 'missing_or_invalid', 'profiles': []}
    for profile in profiles if valid_manifest else []:
        if not isinstance(profile, dict):
            continue
        config_name = profile.get('config')
        config_path = _inside(root, config_name)
        config = _read(config_path) if config_path and config_path.is_file() else None
        config_status = ('present' if config is not None else 'invalid') if config_path and config_path.is_file() else ('missing' if config_name else 'not_configured')
        assets = [_item(root, item, config or {}) for item in (profile.get('assets') or [])]
        runtimes = [_item(root, item, config or {}) for item in (profile.get('runtimes') or [])]
        asset_status = 'installed' if assets and all(x['status'] == 'present' for x in assets) else 'missing'
        runtime_status = 'installed' if runtimes and any(x['status'] == 'present' for x in runtimes) else 'missing'
        status = ('missing_assets' if asset_status == 'missing' else
                  'missing_runtime' if runtime_status == 'missing' else
                  'not_configured' if config_status not in ('present', 'not_configured') else 'installed_unverified')
        result['profiles'].append({'id': profile.get('id'), 'role': profile.get('role'),
            'label': profile.get('label'), 'recommended': bool(profile.get('recommended')),
            'model_id': profile.get('model_id'), 'license': profile.get('license', 'see_model_license'),
            'config': config_path.relative_to(root).as_posix() if config_path else None,
            'config_status': config_status, 'assets': assets, 'runtimes': runtimes,
            'assets_status': asset_status, 'runtime_status': runtime_status, 'status': status,
            'service_status': 'not_checked', 'inference_status': 'not_verified', 'available': False})
    return result
