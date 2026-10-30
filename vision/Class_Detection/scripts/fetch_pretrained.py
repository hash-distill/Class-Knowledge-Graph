#!/usr/bin/env python3
"""Acquire pinned inference checkpoints using only the Python standard library.

No training, deserialization, or inference occurs here. Existing files must pass
both length and SHA-256 checks. An invalid existing file is replaced only with
--repair and only after its replacement has been downloaded and verified.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import tempfile
import urllib.error
import urllib.parse
import urllib.request


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = PROJECT_ROOT / "configs" / "pretrained_models.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "models"
CHUNK_SIZE = 1024 * 1024


class IntegrityError(ValueError):
    """Bytes do not match the pinned checkpoint."""


def validate_spec(model: dict) -> None:
    filename = model.get("filename", "")
    if not isinstance(filename, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*\.pt", filename):
        raise ValueError("Model filename must be a plain .pt filename without directories")
    size = model.get("size_bytes")
    if type(size) is not int or size <= 0:
        raise ValueError("Model size_bytes must be a positive integer")
    digest = model.get("sha256", "")
    if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ValueError("Model sha256 must be a pinned lowercase SHA-256 digest")
    url = model.get("url", "")
    if not isinstance(url, str):
        raise ValueError("Model url must be HTTPS")
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme != "https" or not parsed.netloc or parsed.username or parsed.password:
        raise ValueError("Model url must be HTTPS without embedded credentials")


def load_manifest(path: Path = DEFAULT_MANIFEST) -> dict[str, dict]:
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1 or not isinstance(manifest.get("models"), dict):
        raise ValueError("Unsupported pretrained manifest: expected schema_version 1 and models object")
    models = manifest["models"]
    if not models:
        raise ValueError("Pretrained manifest contains no models")
    filenames = set()
    for model in models.values():
        validate_spec(model)
        if model["filename"].casefold() in filenames:
            raise ValueError("Pretrained manifest contains duplicate destination filenames")
        filenames.add(model["filename"].casefold())
    return models


def verify_file(path: Path, model: dict) -> None:
    """Rehash a local file; file existence alone never establishes integrity."""
    validate_spec(model)
    path = Path(path)
    size = path.stat().st_size
    if size != model["size_bytes"]:
        raise IntegrityError(f"{path.name}: size mismatch: expected {model['size_bytes']}, got {size}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    if digest.hexdigest() != model["sha256"]:
        raise IntegrityError(f"{path.name}: SHA-256 mismatch: expected {model['sha256']}, got {digest.hexdigest()}")


def fetch_model(model: dict, output_dir: Path = DEFAULT_OUTPUT_DIR, *, repair: bool = False,
                timeout: float = 60, opener=None) -> Path:
    """Stream to a unique sibling temp file, verify, then atomically publish it."""
    validate_spec(model)
    if timeout <= 0:
        raise ValueError("timeout must be positive")
    output_dir = Path(output_dir).resolve()
    destination = output_dir / model["filename"]
    if destination.exists():
        try:
            verify_file(destination, model)
            return destination
        except IntegrityError as exc:
            if not repair:
                raise IntegrityError(f"{exc}. Use --repair to download a verified replacement.") from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(
        model["url"], headers={"User-Agent": "Class-Detection-Pretrained/1.0", "Accept-Encoding": "identity"}
    )
    opener = urllib.request.urlopen if opener is None else opener
    temp_path = None
    try:
        with opener(request, timeout=timeout) as response:
            declared_size = response.headers.get("Content-Length")
            if declared_size is not None and int(declared_size) != model["size_bytes"]:
                raise IntegrityError(f"{destination.name}: Content-Length does not match pinned size")
            with tempfile.NamedTemporaryFile(mode="wb", prefix=f".{destination.name}.", suffix=".part",
                                             dir=output_dir, delete=False) as handle:
                temp_path = Path(handle.name)
                digest = hashlib.sha256()
                size = 0
                while True:
                    chunk = response.read(CHUNK_SIZE)
                    if not chunk:
                        break
                    size += len(chunk)
                    if size > model["size_bytes"]:
                        raise IntegrityError(f"{destination.name}: download exceeds pinned size")
                    digest.update(chunk)
                    handle.write(chunk)
                if size != model["size_bytes"]:
                    raise IntegrityError(f"{destination.name}: size mismatch: expected {model['size_bytes']}, got {size}")
                if digest.hexdigest() != model["sha256"]:
                    raise IntegrityError(f"{destination.name}: SHA-256 mismatch: expected {model['sha256']}, got {digest.hexdigest()}")
                handle.flush()
                os.fsync(handle.fileno())
        os.replace(temp_path, destination)
        return destination
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model", action="append", help="Manifest key; repeat to select models (default: all)")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--verify-only", action="store_true", help="Check existing files without network access")
    modes.add_argument("--repair", action="store_true", help="Replace invalid existing files after successful verification")
    parser.add_argument("--timeout", type=float, default=60, help="Network socket timeout in seconds")
    args = parser.parse_args(argv)
    try:
        models = load_manifest(args.manifest)
        selected = list(dict.fromkeys(args.model or models))
        unknown = set(selected) - models.keys()
        if unknown:
            raise ValueError(f"Unknown model key(s): {', '.join(sorted(unknown))}")
        for key in selected:
            model = models[key]
            path = args.output_dir.resolve() / model["filename"]
            print(f"{'Verifying' if args.verify_only else 'Acquiring'} {key}: {path}", file=sys.stderr, flush=True)
            if args.verify_only:
                verify_file(path, model)
            else:
                path = fetch_model(model, args.output_dir, repair=args.repair, timeout=args.timeout)
            print(json.dumps({"model": key, "path": str(path), "size_bytes": model["size_bytes"],
                              "sha256": model["sha256"], "status": "verified"}), flush=True)
        return 0
    except (ValueError, OSError, urllib.error.URLError) as exc:
        print(f"Pretrained acquisition failed: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Pretrained acquisition interrupted; no partial checkpoint was published.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
