"""A clean checkout must not depend on the original machine's paths or private data."""
import importlib.util
import json
from pathlib import Path
import sys
import subprocess
import zipfile

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from review.common import atomic_json
from review.lesson_store import LessonStore
from review.provider import ReviewError


def module(name):
    file = ROOT/'scripts'/f'{name}.py'
    assert file.is_file(), f'Missing public workspace tool {name}'
    spec = importlib.util.spec_from_file_location(name, file)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


def test_relative_registry_paths_are_bound_to_registry_not_shell_cwd(tmp_path, monkeypatch):
    root = tmp_path/'project'; (root/'configs').mkdir(parents=True)
    (root/'media').mkdir(); (root/'media/video.mp4').write_bytes(b'synthetic')
    atomic_json(root/'lessons/test/bundle.json', {'schema_version':'3.0','data_source':'lesson_review',
        'lesson':{'id':'test','status':'complete'}})
    registry = root/'configs/lessons.json'
    atomic_json(registry, {'test':{'directory':'../lessons/test','source_video':'../media/video.mp4'}})
    store = LessonStore(registry)
    other = tmp_path/'other'; other.mkdir(); monkeypatch.chdir(other)
    assert store.load('test')['lesson']['id']=='test'
    assert store.media('test','source.mp4')==root/'media/video.mp4'
    assert json.loads(registry.read_text())['test']['directory']=='../lessons/test'


@pytest.mark.parametrize('value', [[], {'test':None}, {'test':{'directory':None,'source_video':'video.mp4'}}])
def test_invalid_registry_has_an_actionable_error(tmp_path, value):
    atomic_json(tmp_path/'lessons.json', value)
    with pytest.raises(ReviewError) as error: LessonStore(tmp_path/'lessons.json').entry('test')
    assert error.value.code=='invalid_lesson_registry'


def test_workspace_setup_preserves_existing_registry_and_check_is_read_only(tmp_path):
    tools = module('workspace_setup')
    before = set(tmp_path.rglob('*'))
    result = tools.inspect_workspace(tmp_path)
    assert not result['registry_exists'] and set(tmp_path.rglob('*'))==before
    initialized = tools.initialize_workspace(tmp_path)
    registry = Path(initialized['registry'])
    assert json.loads(registry.read_text())=={}
    registry.write_text('{"kept":{"directory":"../data","source_video":"../video.mp4"}}',encoding='utf-8')
    original = registry.read_bytes()
    tools.initialize_workspace(tmp_path)
    assert registry.read_bytes()==original


def test_source_command_is_independent_of_working_directory(tmp_path):
    tools=module('workspace_setup')
    args=tools.backend_command(tmp_path, 'python', 8000, tmp_path/'registry.json', tmp_path/'state')
    assert str(tmp_path/'backend/scripts/desktop_entry.py') in args
    assert args[args.index('--registry')+1]==str(tmp_path/'registry.json')
    assert '--open-browser' not in args and '--reuse' in args


def test_legacy_registry_without_video_still_loads_bundle(tmp_path):
    atomic_json(tmp_path/'lesson/bundle.json',{'schema_version':'3.0','data_source':'lesson_review',
        'lesson':{'id':'legacy','status':'complete'}})
    atomic_json(tmp_path/'registry.json',{'legacy':{'directory':'lesson'}})
    store=LessonStore(tmp_path/'registry.json')
    assert store.load('legacy')['lesson']['id']=='legacy'
    with pytest.raises(ReviewError) as error:store.media('legacy','source.mp4')
    assert error.value.code=='media_not_found'


def fixture_source(tmp_path):
    files={'README.md':'# Jingxi Insight','backend/review/main.py':'print("source")',
        'vision/Class_Detection/models/graph.py':'class Graph: pass',
        'backend/configs/lessons.example.json':'{}','frontend/src/App.vue':'<template>source</template>',
        'frontend/public/brand.svg':'<svg/>','desktop/src-tauri/icons/icon.png':'owned icon'}
    private={'backend/configs/lessons.json':'private registry','backend/artifacts/lesson/transcript.json':'private transcript',
        'backend/.venv-media/private.py':'environment','frontend/.git/config':'private Git settings',
        'frontend/output/playwright/export.json':'classroom export','frontend/public/classroom.json':'private snapshots',
        'frontend/design-preview/snapshot-export.json':'historical classroom export',
        'desktop/workspace/workspace.sqlite3':'private state','vision/Class_Detection/artifacts/weights.pt':'weights',
        'backend/.env':'credentials'}
    for name, content in {**files,**private}.items():
        path=tmp_path/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(content,encoding='utf-8')
    return files, private


def test_source_release_has_code_and_brand_assets_without_private_files(tmp_path):
    tools=module('prepare_release');source=tmp_path/'project';source.mkdir()
    expected,private=fixture_source(source)
    report=tools.prepare_release(source,tmp_path/'release')
    assert report['file_count']==len(expected)
    with zipfile.ZipFile(report['archive']) as archive:
        assert set(archive.namelist())==set(expected)
        assert not set(private).intersection(archive.namelist())
    with pytest.raises(FileExistsError):tools.prepare_release(source,tmp_path/'release')


def test_source_release_rejects_realistic_credentials_without_printing_them(tmp_path):
    tools=module('prepare_release');fixture_source(tmp_path)
    secret='sk-'+'Q'*36
    (tmp_path/'backend/review/main.py').write_text(f'KEY="{secret}"',encoding='utf-8')
    with pytest.raises(ValueError) as error:tools.prepare_release(tmp_path,tmp_path/'out')
    assert secret not in str(error.value) and 'main.py' in str(error.value)
    assert not (tmp_path/'out').exists()


def test_secondary_asr_requires_an_explicit_remote_host(tmp_path):
    import os
    environment={key:value for key,value in os.environ.items() if key!='REVIEW_ASR_SSH_HOST'}
    script=ROOT/'backend/scripts/generate_chinese_reference.py'
    result=subprocess.run([sys.executable,str(script),'--media-dir',str(tmp_path)],capture_output=True,
                          text=True,encoding='utf-8',env=environment)
    assert result.returncode==2 and 'REVIEW_ASR_SSH_HOST' in result.stderr
