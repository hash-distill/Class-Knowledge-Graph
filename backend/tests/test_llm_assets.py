"""Offline integrity/security checks; these do not claim model quality."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("fetch_local_llm", ROOT / "scripts/fetch_local_llm.py")
FETCH = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(FETCH)


class LocalModelAssetsTests(unittest.TestCase):
    def test_pinned_identity_and_local_only_defaults(self):
        config = json.loads((ROOT / "configs/local_llm.json").read_text(encoding="utf-8-sig"))
        self.assertEqual(config["server"]["host"], "127.0.0.1")
        self.assertEqual(config["request_defaults"]["model"], config["model_alias"])
        self.assertFalse(config["request_defaults"]["chat_template_kwargs"]["enable_thinking"])
        self.assertRegex(config["model"]["revision"], r"^[0-9a-f]{40}$")
        self.assertIn(config["model"]["revision"], config["model"]["url"])
        self.assertFalse(config["model"]["trained_here"])
        for backend in ("cpu", "cuda"):
            for asset in config["runtime"][backend]["archives"]:
                self.assertRegex(asset["sha256"], r"^[0-9a-f]{64}$")
                self.assertIn('/' + config["runtime"]["release"] + '/', asset["url"])

    def test_same_size_corruption_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "model.gguf"
            path.write_bytes(b"GGUF" + b"\x03\x00\x00\x00")
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                FETCH.verify(path, {"size_bytes": 8, "sha256": hashlib.sha256(b"XXXXXXXX").hexdigest()})

    def test_unsafe_zip_path_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = root / "unsafe.zip"
            with zipfile.ZipFile(archive, "w") as bundle:
                bundle.writestr("../escape.txt", "unsafe")
            with self.assertRaisesRegex(ValueError, "escapes"):
                FETCH.safe_extract(archive, root / "output")
            self.assertFalse((root / "escape.txt").exists())


if __name__ == "__main__":
    unittest.main()
