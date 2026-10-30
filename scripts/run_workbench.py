"""Portable source entrypoint for the real loopback-only application."""
import argparse
from pathlib import Path
import subprocess
import sys
from workspace_setup import PROJECT_ROOT,backend_command,initialize_workspace,inspect_workspace


def main():
    parser=argparse.ArgumentParser(description='Start Jingxi Insight from source on the loopback interface only.')
    parser.add_argument('--port',type=int,default=8000)
    parser.add_argument('--python',type=Path)
    parser.add_argument('--registry',type=Path)
    parser.add_argument('--data-dir',type=Path)
    parser.add_argument('--open-browser',action='store_true')
    parser.add_argument('--api-only',action='store_true',help='Start the API before the frontend is built, for use with the Vite development server.')
    parser.add_argument('--check',action='store_true')
    args=parser.parse_args()
    if not 0<=args.port<=65535:parser.error('Port must be between 0 and 65535.')
    if args.check:
        import json
        print(json.dumps(inspect_workspace(PROJECT_ROOT,args.registry,args.data_dir),ensure_ascii=False,indent=2));return 0
    try:report=initialize_workspace(PROJECT_ROOT,args.registry,args.data_dir)
    except (OSError,ValueError) as exc:parser.exit(1,f'Workspace initialization failed: {exc}\n')
    if not report['frontend_built'] and not args.api_only:
        parser.exit(1,'The frontend build is missing. Run npm ci and npm run build in frontend, or use --api-only.\n')
    candidates=[PROJECT_ROOT/'backend/.venv-media/Scripts/python.exe',PROJECT_ROOT/'backend/.venv-media/bin/python']
    python=(args.python or next((path for path in candidates if path.is_file()),Path(sys.executable))).expanduser().resolve()
    try:
        probe=subprocess.run([str(python),'-c','import fastapi,uvicorn,PIL,requests'],capture_output=True,timeout=15)
    except (OSError,subprocess.TimeoutExpired):parser.exit(1,'Unable to run backend Python. Use --python to select an interpreter with the application dependencies installed.\n')
    if probe.returncode:parser.exit(1,'Backend dependencies are incomplete. Install backend/requirements-app.txt.\n')
    command=backend_command(PROJECT_ROOT,python,args.port,report['registry'],report['data'],args.open_browser,args.api_only)
    try:return subprocess.call(command,cwd=PROJECT_ROOT/'backend')
    except KeyboardInterrupt:return 130


if __name__=='__main__':raise SystemExit(main())
