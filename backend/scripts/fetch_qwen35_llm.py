"""Download and verify the pinned Qwen3.5 text-correction GGUF."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import urllib.request

from fetch_local_llm import ROOT, download, verify


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-only',action='store_true')
    args=parser.parse_args()
    config=json.loads((ROOT/'configs/qwen35_llm.json').read_text(encoding='utf-8'))
    asset=config['model']; path=ROOT/asset['path']
    result=verify(path,asset) if args.verify_only else download(asset,path)
    result=verify(path,asset)
    if not args.verify_only:
        for repo,revision,name in [(asset['repository'],asset['revision'],'QUANT_MODEL_CARD.md'),
                (asset['upstream_repository'],asset['upstream_revision'],'UPSTREAM_MODEL_CARD.md')]:
            with urllib.request.urlopen(f'https://huggingface.co/{repo}/raw/{revision}/README.md',timeout=40) as response:
                (path.parent/name).write_bytes(response.read())
    manifest={'model':result,'repository':asset['repository'],'revision':asset['revision'],
              'upstream':asset['upstream_repository'],'upstream_revision':asset['upstream_revision'],
              'license':asset['license'],'runtime':config['runtime']['revision'],
              'vision_projector_loaded':False,'verified_at':datetime.now(timezone.utc).isoformat()}
    out=ROOT/'artifacts/verification/qwen-upgrade-20261028';out.mkdir(parents=True,exist_ok=True)
    (out/'qwen35-asset-manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(manifest,ensure_ascii=False))


if __name__=='__main__':
    main()
