"""Offline integrity tests: no pretrained weights or network are required."""
from __future__ import annotations

import hashlib
from contextlib import redirect_stderr
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "fetch_pretrained.py"


class FakeResponse(io.BytesIO):
    def __init__(self, content: bytes, content_length: int | None = None):
        super().__init__(content)
        self.headers = {} if content_length is None else {"Content-Length": str(content_length)}


class InterruptedResponse(FakeResponse):
    def read(self, size=-1):
        if self.tell():
            raise OSError("connection interrupted")
        return super().read(4)


class DownloadTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(SCRIPT.is_file(), "The verified pretrained downloader must exist")
        spec = importlib.util.spec_from_file_location("fetch_pretrained_test", SCRIPT)
        self.downloader = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.downloader)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.output = Path(self.temp.name)
        self.content = b"test-only pretrained weight bytes"
        self.model = {
            "filename": "sample.pt",
            "url": "https://example.invalid/pinned/sample.pt",
            "size_bytes": len(self.content),
            "sha256": hashlib.sha256(self.content).hexdigest(),
        }
        self.network = patch("urllib.request.urlopen", side_effect=AssertionError("Unexpected network"))
        self.network.start()
        self.addCleanup(self.network.stop)

    def fetch(self, content=None, **kwargs):
        content = self.content if content is None else content
        return self.downloader.fetch_model(
            self.model, self.output, opener=lambda *a, **k: FakeResponse(content), **kwargs
        )

    def assert_no_partial(self):
        self.assertEqual(list(self.output.glob("*.part")), [])

    def test_streamed_file_is_verified_and_published(self):
        path = self.fetch()
        self.assertEqual(path, self.output / "sample.pt")
        self.assertEqual(path.read_bytes(), self.content)
        self.assert_no_partial()

    def test_valid_existing_file_is_rehashed_without_network(self):
        destination = self.output / "sample.pt"
        destination.write_bytes(self.content)
        self.assertEqual(self.downloader.fetch_model(self.model, self.output), destination)

    def test_corrupt_existing_file_is_rejected_without_repair(self):
        destination = self.output / "sample.pt"
        destination.write_bytes(b"x" * len(self.content))
        with self.assertRaisesRegex(self.downloader.IntegrityError, "--repair"):
            self.downloader.fetch_model(self.model, self.output)
        self.assertEqual(destination.read_bytes(), b"x" * len(self.content))

    def test_explicit_repair_replaces_only_with_verified_bytes(self):
        destination = self.output / "sample.pt"
        destination.write_bytes(b"bad")
        self.fetch(repair=True)
        self.assertEqual(destination.read_bytes(), self.content)
        self.assert_no_partial()

    def test_failed_repair_preserves_original(self):
        destination = self.output / "sample.pt"
        destination.write_bytes(b"original")
        with self.assertRaises(self.downloader.IntegrityError):
            self.fetch(b"z" * len(self.content), repair=True)
        self.assertEqual(destination.read_bytes(), b"original")
        self.assert_no_partial()

    def test_wrong_hash_never_publishes_file(self):
        with self.assertRaisesRegex(self.downloader.IntegrityError, "SHA-256"):
            self.fetch(b"z" * len(self.content))
        self.assertFalse((self.output / "sample.pt").exists())
        self.assert_no_partial()

    def test_short_download_never_publishes_file(self):
        with self.assertRaisesRegex(self.downloader.IntegrityError, "size"):
            self.fetch(b"short")
        self.assert_no_partial()
        self.assertFalse((self.output / "sample.pt").exists())

    def test_oversized_download_never_publishes_file(self):
        with self.assertRaisesRegex(self.downloader.IntegrityError, "size"):
            self.fetch(self.content + b"extra")
        self.assert_no_partial()
        self.assertFalse((self.output / "sample.pt").exists())

    def test_wrong_content_length_is_rejected(self):
        with self.assertRaisesRegex(self.downloader.IntegrityError, "Content-Length"):
            self.downloader.fetch_model(self.model, self.output, opener=lambda *a, **k: FakeResponse(self.content, 2))
        self.assert_no_partial()

    def test_network_interruption_removes_partial_file(self):
        with self.assertRaisesRegex(OSError, "interrupted"):
            self.downloader.fetch_model(self.model, self.output, opener=lambda *a, **k: InterruptedResponse(self.content))
        self.assertFalse((self.output / "sample.pt").exists())
        self.assert_no_partial()

    def test_unsafe_model_filename_is_rejected(self):
        for name in ("../escape.pt", "sub/file.pt", "sub\\file.pt", "C:escape.pt", ""):
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.downloader.fetch_model({**self.model, "filename": name}, self.output)

    def test_unpinned_integrity_metadata_is_rejected(self):
        for update in ({"sha256": ""}, {"sha256": "not-a-hash"}, {"size_bytes": 0}, {"url": "http://example.invalid/a.pt"}):
            with self.subTest(update=update), self.assertRaises(ValueError):
                self.downloader.fetch_model({**self.model, **update}, self.output)

    def test_verify_only_does_not_download_missing_file(self):
        manifest = self.output / "manifest.json"
        manifest.write_text(json.dumps({"schema_version": 1, "models": {"sample": self.model}}), encoding="utf-8")
        with redirect_stderr(io.StringIO()) as stderr:
            status = self.downloader.main(["--manifest", str(manifest), "--output-dir", str(self.output), "--verify-only"])
        self.assertEqual(status, 1)
        self.assertIn("Pretrained acquisition failed", stderr.getvalue())
        self.assertFalse((self.output / "sample.pt").exists())


if __name__ == "__main__":
    unittest.main()
