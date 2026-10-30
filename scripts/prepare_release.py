"""Build an audited source snapshot without nested Git repositories or classroom data."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import zipfile

ROOT=Path(__file__).resolve().parents[1]
ROOT_FILES={'README.md','.gitignore','.gitattributes','CONTRIBUTING.md','LICENSE','LICENSE.md','NOTICE'}
TREES=('scripts','.github','docs','backend','frontend','desktop','vision')
DROP={'.git','node_modules','artifacts','target','build','dist','dist-ssr','workspace','release','.toolchains',
      '__pycache__','.pytest_cache','.cache','.idea','.vscode','output','test-results','playwright-report',
      '.vendor','runs','checkpoints','wandb','tensorboard','test_images'}
TEXT={'.py','.pyi','.ts','.vue','.js','.mjs','.cjs','.html','.css','.scss','.json','.yaml','.yml','.toml',
      '.md','.txt','.ps1','.lock','.ini','.nsh','.rs','.svg'}
TOKEN=re.compile(r'(?:sk-[A-Za-z0-9_-]{20,}|gh[pousr]_[A-Za-z0-9_]{25,}|hf_[A-Za-z0-9]{25,}|AKIA[A-Z0-9]{16}|-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----)')


def source_files(root):
    root=Path(root).resolve();found=set()
    for name in ROOT_FILES:
        candidate=root/name
        if candidate.is_file() and not candidate.is_symlink() and candidate.resolve().is_relative_to(root):found.add(candidate)
    for tree in TREES:
        base=root/tree
        if not base.is_dir() or base.is_symlink() or not base.resolve().is_relative_to(root):continue
        for directory,dirs,files in os.walk(base,followlinks=False):
            dirs[:]=sorted(d for d in dirs if d not in DROP and not d.startswith(('.venv','pip-','pymp-','tmp','pytest-of-'))
                          and not (Path(directory)/d).is_symlink())
            for name in sorted(files):
                path=Path(directory)/name;relative=path.relative_to(root).as_posix()
                if relative.startswith('desktop/src-tauri/resources/frontend/'):continue
                if relative.startswith('frontend/design-preview/') and path.suffix.lower()=='.json':continue
                if path.is_symlink() or not path.resolve().is_relative_to(root):continue
                if name.startswith('.env') or name in {'lessons.json','jieba.cache'} or '.sqlite' in name or name.endswith('_snapshots.json'):continue
                if relative.startswith('frontend/public/') and relative not in {
                        'frontend/public/brand.svg','frontend/public/favicon.ico'}:continue
                image=path.suffix.lower() in {'.png','.ico','.jpg','.jpeg','.webp','.icns'} and (
                    relative.startswith(('desktop/src-tauri/icons/','desktop/assets/','frontend/src/assets/')) or
                    relative=='frontend/public/favicon.ico')
                if path.suffix.lower() in TEXT or image or name in {'.gitignore','.gitattributes'} or name.upper().startswith(('LICENSE','NOTICE')):
                    found.add(path)
    return sorted(found,key=lambda p:p.relative_to(root).as_posix())


def audit_sources(root):
    root=Path(root).resolve();files=source_files(root);findings=[]
    for path in files:
        if path.suffix.lower() not in TEXT:continue
        try:content=path.read_text(encoding='utf-8-sig')
        except UnicodeError:continue
        for number,line in enumerate(content.splitlines(),1):
            if TOKEN.search(line):findings.append(f'{path.relative_to(root).as_posix()}:{number}')
    if findings:raise ValueError('Possible credentials detected; no release was generated. Check: '+', '.join(findings))
    return files


def prepare_release(root,output):
    root=Path(root).resolve();output=Path(output).resolve()
    if output.exists():raise FileExistsError('The output directory already exists. Choose a new directory; existing files were preserved.')
    files=audit_sources(root)
    if not files:raise ValueError('No publishable source files were found.')
    if root==output or root.is_relative_to(output):raise ValueError('The release directory must not contain the project root.')
    output.mkdir(parents=True)
    source=output/'source';manifest=[]
    archive=output/'jingxi-insight-source.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as package:
        for path in files:
            relative=path.relative_to(root);target=source/relative
            target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
            package.write(target,relative.as_posix())
            manifest.append({'path':relative.as_posix(),'size':target.stat().st_size,
                             'sha256':hashlib.sha256(target.read_bytes()).hexdigest()})
    report={'file_count':len(manifest),'source_directory':str(source),'archive':str(archive),'files':manifest,
            'excluded':['nested Git metadata','environments and caches','private lesson registry','classroom media and exports',
                        'model weights','local databases','build outputs']}
    (output/'manifest.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=ROOT)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--check',action='store_true')
    args=parser.parse_args()
    try:
        if args.check:
            files=audit_sources(args.root)
            print(json.dumps({'status':'passed','file_count':len(files),'bytes':sum(p.stat().st_size for p in files)},ensure_ascii=False));return
        if not args.output:parser.error('Specify a new --output directory, or use --check for a read-only audit.')
        report=prepare_release(args.root,args.output)
        print(json.dumps({k:v for k,v in report.items() if k!='files'},ensure_ascii=False,indent=2))
    except (OSError,ValueError) as exc:parser.exit(1,f'{exc}\n')


if __name__=='__main__':main()
