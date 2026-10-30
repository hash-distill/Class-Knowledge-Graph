"""Observed screen OCR and contiguous change segments from the same lesson video."""
from __future__ import annotations

import base64
import importlib.metadata
import json
import math
import os
import re
import shutil
from difflib import SequenceMatcher
from pathlib import Path

from .asr import atomic_json, inspect_video, sha256_file, utc_now, valid_completed


def normalized_text(text):
    return re.sub(r"[^\w\u4e00-\u9fff]", "", text).casefold()


def text_similarity(a, b):
    a, b = normalized_text(a), normalized_text(b)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b, autojunk=False).ratio()


def sample_times(duration, interval):
    if duration <= 0 or interval <= 0:
        raise ValueError("Duration and sample interval must be positive")
    times = [round(i * interval, 6) for i in range(math.ceil(duration / interval))]
    # Explicitly observe the final decodable part, rather than assume last page persists.
    tail = round(max(0, duration - 0.10), 6)
    if tail - times[-1] > 0.1:
        times.append(tail)
    return times


def page_changed(previous, current):
    """Only compare temporally contiguous groups, never deduplicate whole lesson."""
    if previous["screen_status"] == "dark" or current["screen_status"] == "dark":
        return previous["screen_status"] != current["screen_status"]
    similarity = text_similarity(previous["text"], current["text"])
    chinese_a = "".join(re.findall(r"[\u4e00-\u9fff]", previous["text"]))
    chinese_b = "".join(re.findall(r"[\u4e00-\u9fff]", current["text"]))
    if min(len(chinese_a), len(chinese_b)) >= 8:
        similarity = max(similarity, text_similarity(chinese_a, chinese_b))
    a = base64.b64decode(previous["visual_signature"])
    b = base64.b64decode(current["visual_signature"])
    difference = sum(abs(x-y) for x, y in zip(a, b)) / (255 * max(1, len(a)))
    comparable_text = min(len(normalized_text(previous["text"])), len(normalized_text(current["text"]))) >= 8
    if comparable_text and similarity < 0.45 and difference > 0.02:
        return True
    if difference > 0.19:
        return True
    if comparable_text and similarity < 0.80 and difference > 0.04:
        return True
    if not comparable_text and difference > 0.075:
        return True
    return False


def representative_score(sample):
    # Prefer an unobscured, text-rich frame. Confidence is OCR engine output, not accuracy.
    return math.sqrt(len(normalized_text(sample["text"]))) * max(1, sample["mean_confidence"])


def group_samples(samples, duration, interval):
    if not samples:
        return []
    groups = [[samples[0]]]
    representative = samples[0]
    for sample in samples[1:]:
        if page_changed(representative, sample):
            groups.append([sample])
            representative = sample
        else:
            groups[-1].append(sample)
            if representative_score(sample) > representative_score(representative):
                representative = sample
    slides = []
    for index, group in enumerate(groups):
        chosen = max(group, key=representative_score)
        slide_id = f"S{index+1:04d}"
        lines = [line.strip() for line in chosen["text"].splitlines() if line.strip()]
        title_candidates = [line for line in lines if len(re.findall(r"[\u4e00-\u9fff]", line)) >= 4]
        title = (title_candidates[0] if title_candidates else lines[0])[:70] if lines else f"Screen segment with no recognized text {slide_id}"
        slides.append({
            "id": slide_id, "start_sec": 0.0 if index == 0 else group[0]["time_sec"],
            "end_sec": groups[index+1][0]["time_sec"] if index+1 < len(groups) else duration,
            "title": title, "text": chosen["text"], "image_path": f"slides/{slide_id}.jpg",
            "method": "video_ocr", "uncertainty_sec": interval,
            "representative_time_sec": chosen["time_sec"],
            "representative_sample_id": chosen["id"],
            "sample_ids": [sample["id"] for sample in group],
            "quality": {"status": "machine_ocr_unreviewed", "screen_status": chosen["screen_status"],
                        "mean_ocr_confidence": chosen["mean_confidence"],
                        "note": "Chinese text and mathematical formulas may be misrecognized; inspect the representative image."},
        })
    return slides


def ocr_crop(crop, *, languages='chi_sim+eng', scale=3, body_region=(.13, .90)):
    import cv2
    import pytesseract
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    body = gray[round(gray.shape[0]*body_region[0]):round(gray.shape[0]*body_region[1]), :]
    signature = cv2.resize(body, (96, 54), interpolation=cv2.INTER_AREA)
    signature = cv2.GaussianBlur(signature, (3, 3), 0)
    visual_signature = base64.b64encode(signature.tobytes()).decode("ascii")
    if float(gray.mean()) < 12 and float(gray.std()) < 12:
        return {"text": "", "mean_confidence": 0.0, "words": [], "screen_status": "dark",
                "visual_signature": visual_signature, "ocr_pass": "dark_frame_detected"}
    enlarged = cv2.resize(body, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    # Fixed linear contrast keeps original strokes; do not invent or repair formula text.
    enlarged = cv2.normalize(enlarged, None, 0, 255, cv2.NORM_MINMAX)
    data = pytesseract.image_to_data(enlarged, lang=languages, config="--oem 1 --psm 6",
                                     output_type=pytesseract.Output.DICT, timeout=40)
    words, lines = [], {}
    for index, raw_text in enumerate(data["text"]):
        text = raw_text.strip()
        confidence = float(data["conf"][index])
        if not text or confidence < 0:
            continue
        words.append({"text": text, "confidence": confidence, "left": data["left"][index],
                      "top": data["top"][index], "width": data["width"][index],
                      "height": data["height"][index]})
        if confidence >= 20:
            key = (data["block_num"][index], data["par_num"][index], data["line_num"][index])
            lines.setdefault(key, []).append(text)
    text = "\n".join(" ".join(line) for line in lines.values())
    # Chinese OCR returns one token per character; remove only spaces between CJK characters.
    text = re.sub(r"(?<=[\u4e00-\u9fff]) +(?=[\u4e00-\u9fff])", "", text)
    confidence = sum(word["confidence"] for word in words)/len(words) if words else 0.0
    return {"text": text, "mean_confidence": round(confidence, 3), "words": words,
            "screen_status": "text_observed" if normalized_text(text) else "no_readable_text",
            "visual_signature": visual_signature, "ocr_pass": "tesseract_psm6"}


def run_ocr(video_path, output_dir, *, sample_interval=5.0, roi=(0.49, 0.055, 0.91, 0.47), resume=True,
            languages='chi_sim+eng', scale=3, body_region=(.13, .90)):
    import cv2
    import pytesseract
    os.environ.setdefault("OMP_THREAD_LIMIT", "1")
    video_path, output_dir = Path(video_path).resolve(), Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    source = inspect_video(video_path)
    duration = source["video_duration_sec"]
    if not (0 <= roi[0] < roi[2] <= 1 and 0 <= roi[1] < roi[3] <= 1):
        raise ValueError("ROI must be normalized left top right bottom inside the video")
    if not (0 <= body_region[0] < body_region[1] <= 1) or not math.isfinite(scale) or not 1 <= scale <= 4:
        raise ValueError('OCR body region or scale is invalid')
    if languages not in ('chi_sim', 'chi_sim+eng', 'eng'):raise ValueError('Unsupported OCR language')
    config = {"sample_interval_sec": sample_interval, "roi_normalized": list(roi),
              "ocr_engine": "tesseract", "languages": languages, "psm": 6, "oem": 1,
              "scale": scale, "body_region_inside_roi": [0, body_region[0], 1, body_region[1]],
              "min_word_confidence": 20, "grouping_version": "contiguous_visual_text_v2",
              "timestamp_offset_sec": 0}
    destination = output_dir / "slides.json"
    if resume and valid_completed(destination, source, config):
        print("OCR verified complete cache: " + str(destination), flush=True)
        return json.loads(destination.read_text(encoding="utf-8"))
    started = utc_now()
    raw_path, raw_meta = output_dir / "raw_ocr.jsonl", output_dir / "raw_ocr.provenance.json"
    frame_dir, slide_dir = output_dir / "ocr_frames", output_dir / "slides"
    frame_dir.mkdir(exist_ok=True)
    slide_dir.mkdir(exist_ok=True)
    cached = []
    if resume and raw_path.exists() and raw_meta.exists():
        metadata = json.loads(raw_meta.read_text(encoding="utf-8"))
        # A grouping change can reuse raw OCR observations, but extraction changes cannot.
        raw_config = {k: v for k, v in config.items() if k != "grouping_version"}
        cached_config = {k: v for k, v in metadata.get("config", {}).items() if k != "grouping_version"}
        if metadata.get("source_video", {}).get("sha256") == source["sha256"] and cached_config == raw_config:
            for line in raw_path.read_text(encoding="utf-8").splitlines():
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    break
                if (output_dir / item["image_path"]).exists():
                    cached.append(item)
                else:
                    break
    atomic_json(raw_meta, {"status": "running", "source_video": source, "config": config, "started_at": started})
    times = sample_times(duration, sample_interval)
    if [s["time_sec"] for s in cached] != times[:len(cached)]:
        cached = []
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError("Cannot decode registered source video")
    x1, y1, x2, y2 = [round(v*s) for v, s in zip(roi, [source["width"], source["height"], source["width"], source["height"]])]
    samples = list(cached)
    # Rewrite only the validated prefix so an interrupted trailing line cannot contaminate resume.
    with raw_path.open("w", encoding="utf-8") as stream:
        for sample in cached:
            stream.write(json.dumps(sample, ensure_ascii=False) + "\n")
        stream.flush()
        for index, timestamp in enumerate(times[len(cached):], start=len(cached)):
            capture.set(cv2.CAP_PROP_POS_MSEC, timestamp*1000)
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(f"Failed to decode video at {timestamp:.3f}s; partial OCR remains incomplete")
            actual_timestamp = float(capture.get(cv2.CAP_PROP_POS_MSEC))/1000
            crop = frame[y1:y2, x1:x2]
            relative_image = f"ocr_frames/F{index+1:04d}.jpg"
            cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 95])[1].tofile(str(output_dir / relative_image))
            sample = {"id": f"F{index+1:04d}", "time_sec": timestamp,
                      "decoded_frame_time_sec": round(actual_timestamp, 6),
                      "image_path": relative_image, **ocr_crop(crop, languages=languages, scale=scale, body_region=body_region)}
            samples.append(sample)
            stream.write(json.dumps(sample, ensure_ascii=False) + "\n")
            stream.flush()
            atomic_json(output_dir / "ocr.progress.json", {"status": "running", "samples_written": len(samples),
                        "total_samples": len(times), "latest_sample_time_sec": timestamp,
                        "duration_sec": duration, "started_at": started})
            if len(samples) % 20 == 0:
                print(f"OCR {len(samples)}/{len(times)} frames; reached {timestamp:.1f}/{duration:.1f}s", flush=True)
    capture.release()
    slides = group_samples(samples, duration, sample_interval)
    lookup = {sample["id"]: sample for sample in samples}
    for slide in slides:
        shutil.copyfile(output_dir / lookup[slide["representative_sample_id"]]["image_path"],
                        output_dir / slide["image_path"])
    tessdata = Path(shutil.which("tesseract")).parent / "tessdata"
    language_files = {name: sha256_file(tessdata / (name + ".traineddata"))
                      for name in ["chi_sim", "eng"] if (tessdata / (name + ".traineddata")).exists()}
    provenance = {"source_video": source, "config": config, "source": "video_ocr",
                  "started_at": started, "completed_at": utc_now(), "training_performed": False,
                  "tesseract_version": str(pytesseract.get_tesseract_version()),
                  "traineddata_sha256": language_files,
                  "packages": {name: importlib.metadata.version(name) for name in
                               ["pytesseract", "opencv-python-headless", "numpy", "pillow"]},
                  "raw_ocr_path": "raw_ocr.jsonl", "raw_ocr_sha256": sha256_file(raw_path),
                  "timestamp_offset_sec": 0, "observed_video_range_sec": [0, duration],
                  "limitations": ["Automatic contiguous screen-change segmentation may split animations or teacher occlusions.",
                                  "Page boundaries are sampled, not exact switch events; uncertainty is the sampling interval.",
                                  "Chinese and mathematical OCR is unreviewed. No external slides or preset topic text were used.",
                                  "The source audio lasts slightly longer than the video stream; no OCR evidence is invented for that tail."]}
    result = {"schema_version": "3.0", "status": "complete", "slides": slides,
              "coverage": {"video_duration_sec": duration, "sample_count": len(samples),
                           "expected_sample_count": len(times), "first_sample_sec": samples[0]["time_sec"],
                           "last_sample_sec": samples[-1]["time_sec"], "page_segment_count": len(slides),
                           "max_sampling_gap_sec": max((b["time_sec"]-a["time_sec"] for a, b in zip(samples, samples[1:])), default=0)},
              "provenance": provenance}
    atomic_json(destination, result)
    atomic_json(raw_meta, {"status": "complete", "source_video": source, "config": config,
                          "sample_count": len(samples), "completed_at": utc_now()})
    atomic_json(output_dir / "ocr.progress.json", {"status": "complete", **result["coverage"], "completed_at": utc_now()})
    print(f"OCR complete: {len(samples)} samples, {len(slides)} contiguous page segments", flush=True)
    return result
