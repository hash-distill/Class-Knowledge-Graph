"""Download immutable official snapshots, checking every HF LFS SHA-256."""
import argparse
import hashlib
import json
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from review.qwen_asr import atomic_json, sha256_file
from fetch_local_llm import download


def read_url(url):
    for attempt in range(4):
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                return response.read()
        except (OSError, TimeoutError):
            if attempt == 3:
                raise
            time.sleep(2)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--cache', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--kind', choices=['model', 'aligner'],
                   help='Optionally fetch one independent snapshot only.')
    a = p.parse_args()
    config = json.loads(a.config.read_text(encoding='utf-8-sig'))
    report = {}
    for kind in ([a.kind] if a.kind else ['model', 'aligner']):
        repo, revision = config[kind + '_id'], config[kind + '_revision']
        url = f'https://huggingface.co/api/models/{repo}/revision/{revision}?blobs=true'
        metadata = json.loads(read_url(url))
        if metadata['sha'] != revision:
            raise ValueError('official_revision_mismatch')
        folder = a.cache / 'hub' / ('models--' + repo.replace('/', '--')) / 'snapshots' / revision
        folder.mkdir(parents=True, exist_ok=True)
        atomic_json(a.output / f'{kind}-official-file-metadata.json', metadata)
        hashes = {}
        for item in metadata['siblings']:
            name = item['rfilename']
            if name == '.gitattributes':
                continue
            dest = folder / name
            url = f'https://huggingface.co/{repo}/resolve/{revision}/{name}'
            if item.get('lfs'):
                asset = {'url': url, 'size_bytes': item['lfs']['size'],
                         'sha256': item['lfs']['sha256']}
                download(asset, dest.resolve())
            else:
                if not dest.is_file():
                    data = read_url(url)
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    dest.write_bytes(data)
                data = dest.read_bytes()
                git_sha = hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()
                if len(data) != item['size'] or git_sha != item['blobId']:
                    raise ValueError('official_git_blob_mismatch: ' + name)
            hashes[name] = sha256_file(dest)
        report[kind] = {'id': repo, 'revision': revision, 'path': str(folder.resolve()),
                        'files_sha256': hashes, 'official_identity_verified': True}
        atomic_json(a.output / 'download_manifest.json', report)


if __name__ == '__main__':
    main()
