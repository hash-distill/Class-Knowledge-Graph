"""Fetch pinned public GGUF/runtime files, verify SHA-256, and safely extract.

No third-party Python packages, account, API key, or paid endpoint required.
Partial downloads resume only when the server honors HTTP Range.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path
import struct
import time
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(path: Path, asset: dict) -> dict:
    if path.stat().st_size != asset["size_bytes"]:
        raise ValueError(f"Size mismatch: {path.name}")
    actual = sha256(path)
    if actual != asset["sha256"]:
        raise ValueError(f"SHA-256 mismatch: {path.name}")
    if path.suffix == ".gguf":
        with path.open("rb") as source:
            magic, version = struct.unpack("<4sI", source.read(8))
        if magic != b"GGUF" or version not in (2, 3):
            raise ValueError("Not a supported GGUF file")
    return {"path": str(path.relative_to(ROOT)), "size_bytes": path.stat().st_size,
            "sha256": actual, "verified_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}


def download(asset: dict, path: Path) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return verify(path, asset)
    partial = path.with_suffix(path.suffix + ".part")
    for attempt in range(4):
        offset = partial.stat().st_size if partial.exists() else 0
        if offset == asset["size_bytes"]:
            break
        headers = {"User-Agent": "classroom-review-public-model-fetch/1.0"}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        try:
            request = urllib.request.Request(asset["url"], headers=headers)
            with urllib.request.urlopen(request, timeout=60) as response:
                append = offset > 0 and response.status == 206
                if append and not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
                    raise ValueError("Unexpected Content-Range")
                written = offset if append else 0
                with partial.open("ab" if append else "wb") as target:
                    report = time.monotonic()
                    while chunk := response.read(4 * 1024 * 1024):
                        target.write(chunk)
                        written += len(chunk)
                        if written > asset["size_bytes"]:
                            raise ValueError("Downloaded response exceeds pinned file size")
                        if time.monotonic() - report >= 15:
                            print(f"{path.name}: {written / asset['size_bytes']:.1%}", flush=True)
                            report = time.monotonic()
            break
        except (OSError, TimeoutError) as error:
            if attempt == 3:
                raise
            print(f"Retry {attempt + 1}: {path.name}: {type(error).__name__}", flush=True)
            time.sleep(2)
    result = verify(partial, asset)
    partial.replace(path)
    result["path"] = str(path.relative_to(ROOT))
    # The final suffix, unlike .part, triggers the GGUF magic check.
    if path.suffix == ".gguf":
        with path.open("rb") as source:
            if source.read(4) != b"GGUF":
                raise ValueError("Not a GGUF file")
    print(f"Verified: {path.name}", flush=True)
    return result


def safe_extract(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    with zipfile.ZipFile(archive) as bundle:
        for item in bundle.infolist():
            final = (root / item.filename).resolve()
            if not final.is_relative_to(root):
                raise ValueError("Archive member escapes destination")
            if (item.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError("Archive symlinks are not accepted")
        bundle.extractall(root)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", choices=("cpu", "cuda"), default=None)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    config = json.loads((ROOT / "configs/local_llm.json").read_text(encoding="utf-8-sig"))
    args.runtime = args.runtime or config["runtime"]["default_backend"]
    runtime = config["runtime"][args.runtime]
    assets = [(config["model"], ROOT / config["model"]["path"])]
    assets += [(item, ROOT / "artifacts/runtime/downloads" / item["filename"]) for item in runtime["archives"]]
    action = verify if args.verify_only else download
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(action, path, asset) if args.verify_only else pool.submit(action, asset, path)
                   for asset, path in assets]
        results = [future.result() for future in futures]
    if not args.verify_only:
        for asset, archive in assets[1:]:
            safe_extract(archive, ROOT / runtime["directory"])
        source_dir = ROOT / "artifacts/models/Qwen3-4B-GGUF"
        urls = {
            "MODEL_CARD.md": f"https://huggingface.co/{config['model']['repository']}/raw/{config['model']['revision']}/README.md",
            "LICENSE": f"https://huggingface.co/{config['model']['repository']}/raw/{config['model']['revision']}/LICENSE",
        }
        for name, url in urls.items():
            with urllib.request.urlopen(url, timeout=45) as response:
                (source_dir / name).write_bytes(response.read())
        with urllib.request.urlopen(f"https://raw.githubusercontent.com/ggml-org/llama.cpp/{config['runtime']['revision']}/LICENSE", timeout=45) as response:
            (ROOT / runtime["directory"] / "LICENSE").write_bytes(response.read())
    manifest = {"model_repository": config["model"]["repository"], "model_revision": config["model"]["revision"],
                "runtime_release": config["runtime"]["release"], "backend": args.runtime, "assets": results}
    out = ROOT / "artifacts/verification/llm"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"asset_manifest_{args.runtime}.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
