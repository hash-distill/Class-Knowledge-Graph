"""Checks for truthful media timing, contiguous pages, and verified resume."""
import base64
import json
import tempfile
import unittest
from pathlib import Path

from review.asr import atomic_json, valid_completed
from review.media import group_samples, sample_times, text_similarity


def sample(index, timestamp, text, shade=180, status="text_observed"):
    return {"id": f"F{index:04d}", "time_sec": timestamp, "text": text,
            "screen_status": status, "mean_confidence": 80.0,
            "visual_signature": base64.b64encode(bytes([shade]*12)).decode("ascii")}


class MediaContractTests(unittest.TestCase):
    def test_full_sample_schedule_includes_tail_without_audio_extension(self):
        times = sample_times(1144.433313, 5)
        self.assertEqual(len(times), 230)
        self.assertEqual(times[0], 0)
        self.assertAlmostEqual(times[-1], 1144.333313)
        self.assertLess(max(b-a for a, b in zip(times, times[1:])), 5.00001)

    def test_short_video_and_invalid_interval(self):
        self.assertEqual(sample_times(0.05, 5), [0])
        with self.assertRaises(ValueError):
            sample_times(10, 0)

    def test_recurring_content_stays_separate_in_time(self):
        a = "第一部分三角函数与边角关系实际观测文字"
        b = "第二部分空间曲面和立体形状完全不同内容"
        slides = group_samples([sample(1, 0, a), sample(2, 5, a),
                                sample(3, 10, b, 120), sample(4, 15, a)], 20, 5)
        self.assertEqual(len(slides), 3)
        self.assertEqual([x["id"] for x in slides], ["S0001", "S0002", "S0003"])
        self.assertEqual([(x["start_sec"], x["end_sec"]) for x in slides], [(0, 10), (10, 15), (15, 20)])

    def test_dark_interval_is_explicit_not_fake_lesson_text(self):
        slides = group_samples([sample(1, 0, "", 0, "dark"), sample(2, 5, "", 0, "dark"),
                                sample(3, 10, "真实课件原文")], 12, 5)
        self.assertEqual(len(slides), 2)
        self.assertEqual(slides[0]["text"], "")
        self.assertEqual(slides[0]["quality"]["screen_status"], "dark")
        self.assertEqual(slides[0]["end_sec"], slides[1]["start_sec"])

    def test_whitespace_does_not_create_false_text_change(self):
        self.assertEqual(text_similarity("正 弦 定 理", "正弦定理"), 1)

    def test_ocr_noise_without_visual_change_does_not_split_page(self):
        slides = group_samples([sample(1, 0, "识别不稳定的公式正文 ABCD"),
                                sample(2, 5, "完全错误的另一串字母 EFGH")], 10, 5)
        self.assertEqual(len(slides), 1)

    def test_resume_rejects_partial_wrong_source_and_wrong_config(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "asr.json"
            source, config = {"sha256": "source-a"}, {"beam_size": 3}
            atomic_json(path, {"status": "running", "provenance": {"source_video": source, "config": config}})
            self.assertFalse(valid_completed(path, source, config))
            value = json.loads(path.read_text())
            value["status"] = "complete"
            atomic_json(path, value)
            self.assertTrue(valid_completed(path, source, config))
            self.assertFalse(valid_completed(path, {"sha256": "source-b"}, config))
            self.assertFalse(valid_completed(path, source, {"beam_size": 1}))


if __name__ == "__main__":
    unittest.main()
