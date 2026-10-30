"""First-run workspace tools; only user-local state is created, never sample observations."""
import argparse
import json
from pathlib import Path
import shutil
import sys

PROJECT_ROOT=Path(__file__).resolve().parents[1]


def paths(root,registry=None,data_dir=None):
    root=Path(root).resolve()
    return {'root':root,'registry':Path(registry).expanduser().resolve() if registry else root/'backend/configs/lessons.json',
            'data':Path(data_dir).expanduser().resolve() if data_dir else root/'backend/artifacts/runtime',
            'frontend':root/'frontend/dist'}


def inspect_workspace(root,registry=None,data_dir=None):
    current=paths(root,registry,data_dir)
    return {**{key:str(value) for key,value in current.items()},'python':sys.executable,
            'python_supported':sys.version_info>=(3,11),'registry_exists':current['registry'].is_file(),
            'frontend_built':(current['frontend']/'index.html').is_file(),
            'tools':{name:bool(shutil.which(name)) for name in ('node','npm','ffmpeg','tesseract')}}


def initialize_workspace(root,registry=None,data_dir=None):
    current=paths(root,registry,data_dir)
    target=current['registry']
    target.parent.mkdir(parents=True,exist_ok=True)
    try:
        with target.open('x',encoding='utf-8') as handle:handle.write('{}\n')
    except FileExistsError:pass
    document=json.loads(target.read_text(encoding='utf-8-sig'))
    if not isinstance(document,dict):raise ValueError('The lesson registry must be a JSON object. The original file was preserved.')
    current['data'].mkdir(parents=True,exist_ok=True)
    return inspect_workspace(root,registry,data_dir)


def backend_command(root,python,port,registry,data_dir,open_browser=False,api_only=False):
    current=paths(root,registry,data_dir)
    command=[str(python),str(current['root']/'backend/scripts/desktop_entry.py'),'--port',str(port),
             '--registry',str(current['registry']),'--data-dir',str(current['data']),
             '--frontend-dir',str(current['frontend']),'--runtime-root',str(current['root']/'backend'),'--reuse']
    if open_browser:command.append('--open-browser')
    if api_only:command.append('--api-only')
    return command


def main():
    parser=argparse.ArgumentParser(description='Inspect or initialize a Jingxi Insight workspace without overwriting an existing lesson registry.')
    parser.add_argument('--root',type=Path,default=PROJECT_ROOT)
    parser.add_argument('--registry',type=Path)
    parser.add_argument('--data-dir',type=Path)
    parser.add_argument('--check',action='store_true',help='Inspect only; do not create files.')
    args=parser.parse_args()
    operation=inspect_workspace if args.check else initialize_workspace
    try:report=operation(args.root,args.registry,args.data_dir)
    except (OSError,ValueError) as exc:parser.exit(1,f'Workspace initialization failed: {exc}\n')
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
