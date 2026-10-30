"""Fetch and verify the pinned CEC4 GGUF, reusing the established public fetcher."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import urllib.request

from fetch_local_llm import ROOT, download, verify


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    config = json.loads((ROOT/'configs/correction_llm.json').read_text(encoding='utf-8'))
    asset = config['model']
    path = ROOT/asset['path']
    result = verify(path, asset) if args.verify_only else download(asset, path)
    result = verify(path, asset)
    if not args.verify_only:
        directory = path.parent
        for repo, name in [(asset['repository'], 'QUANT_MODEL_CARD.md'),
                           (asset['upstream_repository'], 'UPSTREAM_MODEL_CARD.md')]:
            revision = asset['revision'] if repo == asset['repository'] else 'main'
            with urllib.request.urlopen(f'https://huggingface.co/{repo}/raw/{revision}/README.md', timeout=30) as response:
                (directory/name).write_bytes(response.read())
    manifest = {'model': result, 'repository': asset['repository'], 'revision': asset['revision'],
                'license': asset['license'], 'source_model': asset['upstream_repository'],
                'quantization': asset['quantization'], 'runtime_reused': config['runtime']['cuda'],
                'verified_at': datetime.now(timezone.utc).isoformat()}
    output = ROOT/'artifacts/verification/llm-correction'; output.mkdir(parents=True, exist_ok=True)
    (output/'asset_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(manifest, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
