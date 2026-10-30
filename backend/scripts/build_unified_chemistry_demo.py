"""Rebuild one same-source Chinese lesson without replacing the original evidence.

Run with .venv-media Python. Models are loaded only from the already pinned local
cache. The optional --register flag is the only way this script changes lessons.json.
Screen intervals/regions below are observations of this exact SHA-256, not a generic
screen detector. Original ASR and the full dense OCR pass remain separately auditable.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import importlib.metadata
import json
from pathlib import Path
import re
import shutil
import sys
import tempfile
import time
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from review.asr import atomic_json, inspect_video, sha256_file, utc_now

SOURCE_SHA256 = "bd955b05ac65bcae85890ecda939e8e027389f3e739f1ea3775299db5231df1d"
MODEL_REVISION = "536b0662742c02347bc0e980a01041f333bce120"
SCREEN_RANGE = (43.5, 47.5)
# Pixel rectangles measured on the original 1280x720 source, before OCR.
REGIONS = {
    "heading": (470, 0, 850, 95),
    "example_1": (100, 330, 290, 405),
    "example_2": (440, 330, 630, 405),
    "example_3": (95, 600, 285, 670),
}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def remap(value, old_id, lesson_id):
    if isinstance(value, dict):
        return {k: remap(v, old_id, lesson_id) for k, v in value.items()}
    if isinstance(value, list):
        return [remap(v, old_id, lesson_id) for v in value]
    if isinstance(value, str):
        return value.replace(f"/api/review/media/{old_id}/", f"/api/review/media/{lesson_id}/")
    return value


def prepare(video, reuse, output, lesson_id):
    from review.media import ocr_crop, run_ocr
    import cv2

    if output.resolve() == reuse.resolve() or output.resolve().is_relative_to(reuse.resolve()):
        raise ValueError("A new output directory is required; preserve the original lesson")
    source = inspect_video(video)
    if source["sha256"] != SOURCE_SHA256 or (source["width"], source["height"]) != (1280, 720):
        raise ValueError("This observed screen recipe only applies to the fixed original video")
    original_asr = read_json(reuse / "asr.json")
    activity = read_json(reuse / "activity.json")
    if original_asr["provenance"]["source_video"]["sha256"] != SOURCE_SHA256 or activity["source_sha256"] != SOURCE_SHA256:
        raise ValueError("ASR, behavior and original video must have one SHA-256")
    output.mkdir(parents=True, exist_ok=True)
    for name in ("asr.json", "audio.wav", "audio.provenance.json"):
        destination = "asr.unguided.json" if name == "asr.json" else name
        shutil.copyfile(reuse / name, output / destination)
        if sha256_file(reuse / name) != sha256_file(output / destination):
            raise ValueError("Copied evidence changed bytes")

    dense_path = output / "slides.dense-unfiltered.json"
    if dense_path.exists():
        dense = read_json(dense_path)
    else:
        dense = run_ocr(video, output, sample_interval=1.0,
                        roi=(.025, 0, .975, .96), languages="chi_sim", scale=1,
                        body_region=(0, 1))
        atomic_json(dense_path, dense)
    if dense["provenance"]["source_video"]["sha256"] != SOURCE_SHA256:
        raise ValueError("Dense OCR source mismatch")

    # Supplement the generic PSM6 pass with small layout regions; no text repair.
    capture = cv2.VideoCapture(str(video))
    observations = []
    screen_dir = output / "screen_observations"
    screen_dir.mkdir(exist_ok=True)
    for timestamp in [43.0, 43.5, 44.0, 44.5, 45.0, 45.5, 46.0, 46.5, 47.0, 47.5]:
        capture.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
        ok, frame = capture.read()
        if not ok:
            raise ValueError("Failed to decode a screen-boundary observation")
        filename = "screen_observations/at-" + str(timestamp) + ".jpg"
        cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 95])[1].tofile(str(output / filename))
        observed = SCREEN_RANGE[0] <= timestamp < SCREEN_RANGE[1]
        row = {"id": "Q" + str(len(observations) + 1).zfill(4), "time_sec": timestamp,
               "image_path": filename, "image_sha256": sha256_file(output / filename),
               "screen_visible": observed,
               "visibility_method": "human_visual_check_of_fixed_source_camera_cut",
               "regions": {}}
        if observed:
            for key, (x1, y1, x2, y2) in REGIONS.items():
                result = ocr_crop(frame[y1:y2, x1:x2], languages="chi_sim", scale=1,
                                  body_region=(0, 1))
                # Joining a wrapped line in one spatial label does not replace characters.
                result["joined_text"] = re.sub(r"\s+", "", result["text"])
                row["regions"][key] = {"rectangle_source_pixels": [x1, y1, x2, y2], **result}
        observations.append(row)
    capture.release()
    raw_regions = output / "screen_regions.jsonl"
    raw_regions.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in observations), encoding="utf-8")
    chosen = max((r for r in observations if r["screen_visible"]),
                 key=lambda r: sum(region["mean_confidence"] for region in r["regions"].values()))
    labels = [chosen["regions"][key]["joined_text"] for key in REGIONS]
    if any(not text for text in labels):
        raise ValueError("Screen vocabulary must come from actual nonempty OCR")
    (output / "slides").mkdir(exist_ok=True)
    shutil.copyfile(output / chosen["image_path"], output / "slides/S0001.jpg")
    slides = deepcopy(dense)
    slides["slides"] = [{"id": "S0001", "start_sec": SCREEN_RANGE[0], "end_sec": SCREEN_RANGE[1],
                         "title": labels[0], "text": "\n".join(labels), "image_path": "slides/S0001.jpg",
                         "method": "video_ocr", "uncertainty_sec": .5,
                         "representative_time_sec": chosen["time_sec"], "representative_sample_id": chosen["id"],
                         "sample_ids": [r["id"] for r in observations if r["screen_visible"]],
                         "quality": {"status": "machine_ocr_unreviewed", "screen_status": "text_observed",
                                     "note": "原视频区域OCR；化学结构式以原始画面核对。只在观测到投影的43.5至47.5秒保留课件证据，边界抽样误差0.5秒。"}}]
    slides["coverage"]["page_segment_count"] = 1
    slides["coverage"]["visible_courseware_union_sec"] = SCREEN_RANGE[1] - SCREEN_RANGE[0]
    slides["provenance"]["screen_selection"] = {
        "method": "visually_verified_fixed_source_interval_and_native_resolution_region_ocr",
        "visible_intervals_sec": [list(SCREEN_RANGE)], "boundary_uncertainty_sec": .5,
        "raw_regions_path": "screen_regions.jsonl", "raw_regions_sha256": sha256_file(raw_regions),
        "dense_unfiltered_slides_path": "slides.dense-unfiltered.json",
        "regions_source_pixels": REGIONS,
        "no_slide_evidence_outside_visible_intervals": True,
        "text_changes": "Remove whitespace inside each spatial OCR label only; no character substitutions",
    }
    slides["provenance"]["limitations"].append("Only the visually observed projection interval is retained as courseware; student camera views are not pages.")
    atomic_json(output / "slides.json", slides)
    atomic_json(output / "asr_prompt_vocabulary.json", {
        "source_video_sha256": SOURCE_SHA256, "source_evidence_id": "S0001",
        "source_observation_id": chosen["id"], "source_time_sec": chosen["time_sec"],
        "method": "native_resolution_screen_region_ocr_whitespace_join_only",
        "vocabulary": labels, "initial_prompt": "，".join(labels) + "。",
        "not_a_reference_transcript": True,
    })

    old_id = read_json(reuse / "bundle.json")["lesson"]["id"]
    copied = []
    for frame in activity["frames"]:
        prefix = f"/api/review/media/{old_id}/"
        if not frame["image_url"].startswith(prefix):
            raise ValueError("Behavior image belongs to another lesson")
        relative = Path(unquote(frame["image_url"][len(prefix):]))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Unsafe behavior path")
        destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(reuse / relative, destination)
        digest = sha256_file(reuse / relative)
        if digest != sha256_file(destination):
            raise ValueError("Behavior frame did not preserve source bytes")
        copied.append({"path": relative.as_posix(), "sha256": digest, "time_sec": frame["timestamp_sec"]})
    activity = remap(activity, old_id, lesson_id)
    activity["provenance"]["reuse"] = {"from_lesson_id": old_id,
        "parent_activity_sha256": sha256_file(reuse / "activity.json"),
        "same_source_sha256_verified": SOURCE_SHA256, "copied_frame_bytes_verified": len(copied),
        "mutation": "Only lesson media URL prefix and added reuse provenance; no predictions changed",
        "manifest": "activity/copied_frames.sha256.json"}
    atomic_json(output / "activity.json", activity)
    atomic_json(output / "activity/copied_frames.sha256.json", copied)
    print("Prepared actual OCR vocabulary:", labels, flush=True)


def transcribe(video, reuse, output):
    from faster_whisper import WhisperModel

    source = inspect_video(video)
    if source["sha256"] != SOURCE_SHA256:
        raise ValueError("Unexpected source")
    original = read_json(output / "asr.unguided.json")
    model_dir = reuse / "model_cache/models--Systran--faster-whisper-small/snapshots" / MODEL_REVISION
    expected_files = original["provenance"]["model_files"]
    for name, spec in expected_files.items():
        if sha256_file(model_dir / name) != spec["sha256"]:
            raise ValueError("Pinned local model file hash mismatch")
    if sha256_file(output / "audio.wav") != original["provenance"]["audio_sha256"]:
        raise ValueError("Decoded source audio changed")
    prompt = read_json(output / "asr_prompt_vocabulary.json")
    started = utc_now()
    timer = time.monotonic()
    model = WhisperModel(str(model_dir), device="cpu", compute_type="int8", cpu_threads=4,
                         local_files_only=True)
    iterator, info = model.transcribe(str(output / "audio.wav"), language="zh", beam_size=5,
        vad_filter=True, condition_on_previous_text=False, initial_prompt=None,
        hotwords="，".join(prompt["vocabulary"]), repetition_penalty=1.15)
    segments = []
    with (output / "asr.partial.jsonl").open("w", encoding="utf-8") as stream:
        for segment in iterator:
            row = {"id": "T" + str(len(segments) + 1).zfill(4),
                "start_sec": round(segment.start, 3), "end_sec": round(min(segment.end, source["duration_sec"]), 3),
                "text": segment.text.strip(), "source": "faster_whisper",
                "quality": {"status": "machine_transcript_ocr_vocabulary_prompted_unreviewed",
                    "avg_logprob": segment.avg_logprob, "no_speech_prob": segment.no_speech_prob,
                    "compression_ratio": segment.compression_ratio,
                    "note": "同源课件OCR词汇提示的机器转写；原无提示结果另存，未手工改写台词。"}}
            if row["text"] and row["end_sec"] > row["start_sec"]:
                segments.append(row)
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                stream.flush()
                print(row["id"], row["end_sec"], row["text"], flush=True)
    result = deepcopy(original)
    result["segments"] = segments
    config = result["provenance"]["config"]
    config.update({"beam_size": 5, "initial_prompt": None,
        "hotwords": "，".join(prompt["vocabulary"]), "condition_on_previous_text": False,
        "repetition_penalty": 1.15, "local_files_only": True})
    result["provenance"].update({"started_at": started, "completed_at": utc_now(),
        "audio_duration_sec": info.duration, "vad_speech_duration_sec": info.duration_after_vad,
        "generator_exhausted": True, "inference_wall_time_sec": round(time.monotonic() - timer, 3),
        "prompt_provenance": prompt, "unguided_asr_path": "asr.unguided.json",
        "unguided_asr_sha256": sha256_file(output / "asr.unguided.json"),
        "quality_note": "Same-video OCR vocabulary prompted ASR. Neither raw nor prompted output is human-annotated accuracy evidence.",
        "packages": {name: importlib.metadata.version(name) for name in original["provenance"]["packages"]}})
    result["coverage"] = {"processed_duration_sec": source["duration_sec"], "segment_count": len(segments),
        "first_segment_start_sec": segments[0]["start_sec"], "last_segment_end_sec": segments[-1]["end_sec"],
        "sum_segment_duration_sec": round(sum(s["end_sec"] - s["start_sec"] for s in segments), 3)}
    atomic_json(output / "asr.json", result)


def build(video, output, lesson_id, register):
    import review.builder as builder
    from review.courseware_service import overlay_bundle, persist_bundle
    from review.mutation import LessonMutationLock

    def perform():
        # Use the builder's validated implementation while owning its usual lease
        # through post-processing. This avoids a gap for a concurrent PDF import.
        with LessonMutationLock(output):
            bundle = builder._build_lesson(lesson_id, video, output, "认识氨基酸 · 同课语音、课件与可见行为")
            document = bundle.get("courseware")
            base = read_json(output / "bundle.base.json") if document else bundle
            base["transcript_unguided"] = read_json(output / "asr.unguided.json")["segments"]
            base["lesson"]["provenance"]["transcript_variants"] = {
                "displayed": "automatically_proofread_machine_asr" if base.get("transcript_proofreading") else "same_video_ocr_vocabulary_prompted_machine_asr",
                "raw_guided_field": "transcript", "proofread_field": "transcript_proofreading.segments",
                "unguided_export_field": "transcript_unguided", "unguided_artifact": "asr.unguided.json"}
            notices = ["语音为同源课件OCR词汇提示的机器转写；保留原36段无提示转写，未人工替换台词，未计算转写准确率。",
                       "课件只在原录像43.5至47.5秒投影可见时提供证据；其他镜头不沿用上一页。"]
            base["report"]["limitations"] = list(dict.fromkeys(base["report"].get("limitations", []) + notices))
            proofread_graph = bool(base.get("transcript_proofreading", {}).get("edit_count"))
            # Preserve the pre-proofreading candidates byte-for-byte. A new
            # model extraction uses a separate auditable candidate archive.
            if proofread_graph:
                from review.proofreading import digest
                graph_fingerprint = read_json(output / "graph.json")["input_fingerprint"]
                archive_name = "graph-candidates-proofread-" + digest(graph_fingerprint)[:12] + ".json"
            else:
                archive_name = "graph-candidates-original.json"
            archive_path = output / archive_name
            archive = read_json(archive_path) if archive_path.exists() else None
            filtered_base, archive = filter_graph_with_observed_courseware(base, candidate_archive=archive)
            archive["artifact_name"] = archive_name
            filtered_base["graph"]["provenance"]["display_policy"]["original_candidates_artifact"] = archive_name
            if not archive_path.exists():
                atomic_json(archive_path, archive)
            # Keep the extraction fingerprint, but cache the display policy so a
            # normal builder cache hit cannot restore ASR-only erroneous terms.
            graph_cache_path = output / "graph.json"
            cache = read_json(graph_cache_path)
            cache["graph"] = deepcopy(filtered_base["graph"])
            atomic_json(graph_cache_path, cache)
            if document:
                atomic_json(output / "bundle.base.json", filtered_base)
                overlaid = overlay_bundle(filtered_base, document)
                result, _ = filter_graph_with_observed_courseware(overlaid, candidate_archive=archive)
            else:
                result = filtered_base
            persist_bundle(output, result)
            return result

    if register:
        bundle = perform()
    else:
        # Builder normally registers the lesson. Keep the registry mutation isolated
        # and outside served artifacts unless the operator explicitly asks to register.
        original_root = builder.ROOT
        try:
            with tempfile.TemporaryDirectory(prefix="unified-lesson-build-") as temporary:
                builder.ROOT = Path(temporary)
                bundle = perform()
        finally:
            builder.ROOT = original_root
    return bundle


def filter_graph_with_observed_courseware(bundle, *, candidate_archive=None):
    """Return (filtered_bundle, original_candidate_archive), without I/O.

    Existing model-proposed labels require literal actual video-slide OCR or,
    after generation-time proofreading, separately traceable corrected ASR.
    Whitespace is ignored; this policy never changes the evidence characters.
    Call on bundle.base before a courseware overlay. If an overlay adds concepts,
    call again with the returned archive to retain the same model-candidate set.
    Persist the archive only once, under the caller's lesson mutation lease.
    """
    result = deepcopy(bundle)
    graph = result["graph"]
    source_hash = result["lesson"]["source_video"]["sha256"]
    if candidate_archive is None:
        candidate_archive = {
            "schema_version": "1.0", "source_video_sha256": source_hash,
            "candidate_semantics": "original_model_proposed_machine_candidates_unreviewed",
            "original_graph": deepcopy(bundle["graph"]),
            "notice": "原模型候选留存；原文出现不等于术语正确或教师确认。",
        }
    else:
        candidate_archive = deepcopy(candidate_archive)
        if candidate_archive.get("source_video_sha256") != source_hash:
            raise ValueError("Candidate archive belongs to another source")
    original_nodes = candidate_archive["original_graph"]["nodes"]
    originals = {n["id"]: n for n in original_nodes}
    literal = lambda text: re.sub(r"\s+", "", text)
    support = [(r["id"], literal(r.get("text", ""))) for r in bundle.get("slides", [])
               if r.get("method") == "video_ocr" and r.get("text", "").strip()]
    proofread = bundle.get("transcript_proofreading", {})
    use_proofread = bool(proofread.get("edit_count"))
    if use_proofread:
        support.extend((row["id"], literal(row["text"])) for row in proofread.get("segments", []))
    kept = []
    for node in graph["nodes"]:
        original = originals.get(node["id"])
        if original is None or node["label"] != original["label"]:
            continue
        label = literal(node["label"])
        refs = [identifier for identifier, text in support if label and label in text]
        if refs:
            node.setdefault("properties", {})["observed_courseware_literal_refs"] = [
                identifier for identifier in refs if any(page["id"] == identifier for page in bundle.get("slides", []))]
            if use_proofread:
                node["properties"]["proofread_literal_refs"] = [
                    identifier for identifier in refs if any(row["id"] == identifier for row in proofread.get("segments", []))]
            kept.append(node)
    kept_ids = {n["id"] for n in kept}
    graph["nodes"] = kept
    graph["edges"] = [e for e in graph["edges"] if e["source"] in kept_ids and e["target"] in kept_ids]
    graph.setdefault("provenance", {})["display_policy"] = {
        "name": "model_candidates_with_proofread_or_observed_courseware_support_v2" if use_proofread else "model_candidates_with_observed_video_courseware_literal_support_v1",
        "support_source": "automatically_proofread_asr_and_actual_video_slide_ocr" if use_proofread else "actual_video_slide_ocr_only", "comparison": "literal_after_whitespace_removal",
        "original_candidate_count": len(original_nodes), "displayed_candidate_count": len(kept),
        "hidden_candidate_ids": [n["id"] for n in original_nodes if n["id"] not in kept_ids],
        "original_candidate_ids": list(originals), "original_candidates_artifact": candidate_archive.get("artifact_name", "graph-candidates-proofread.json" if use_proofread else "graph-candidates-original.json"),
        "teacher_or_chemical_accuracy_validated": False,
        "notice": "候选词来自本轮模型提取并以校对语音或实际课件字面出现核对；自动校对未人工核准。" if use_proofread else "仅展示模型已提议且在本课实际课件OCR中有字面支持的候选词；未新增术语，未声称化学人工验证。",
    }
    for row in result.get("alignment", []):
        row["concept_ids"] = [identifier for identifier in row.get("concept_ids", []) if identifier in kept_ids]
    report = result["report"]
    old_labels = {n["label"] for n in bundle["graph"]["nodes"]} | {n["label"] for n in original_nodes}
    other_sections = [section for section in report.get("sections", []) if section.get("title") not in old_labels]
    sections = []
    for node in kept:
        refs = list(dict.fromkeys(node["properties"].get("source_refs", []) +
                                  node["properties"]["observed_courseware_literal_refs"] +
                                  node["properties"].get("proofread_literal_refs", [])))
        sections.append({"title": node["label"],
            "text": "模型候选词在自动校对语音或实测课件中出现。点击引用核对校对差异与原声；尚未教师确认。" if use_proofread else "模型已提议的候选词，同时在本课实际课件OCR中出现。点击引用核对原始语音和课件；尚未教师确认。",
            "evidence_ids": refs})
    report["sections"] = sections + other_sections
    statistics = report.setdefault("statistics", {})
    statistics.update({"concepts": len(kept), "relations": len(graph["edges"]),
                       "graph_candidates_original": len(original_nodes),
                       "graph_candidates_hidden": len(original_nodes) - len(kept)})
    report["summary"] = (f"完整源录像 {result['lesson']['duration_sec']:.2f} 秒，产生 "
        f"{len(result.get('transcript', []))} 条语音片段、{len(result.get('slides', []))} 个课件画面段，"
        f"展示 {len(kept)} 个得到{'自动校对语音或实际课件' if use_proofread else '实际课件OCR'}字面支持的模型候选概念。")
    notice = ("图谱依据生成时自动校对的语音与实际课件重新提取，原候选另存可查，未作化学人工验证。" if use_proofread else "图谱只展示模型已提议且得到本课实际课件OCR字面支持的候选词；原候选另存可查，未作化学人工验证。")
    report["limitations"] = list(dict.fromkeys(report.get("limitations", []) + [notice]))
    return result, candidate_archive


def verify(output):
    from review.activity import validate_activity
    from review.timebase import validate_media_timebase

    bundle, asr, slides, activity = [read_json(output / name) for name in ("bundle.json", "asr.json", "slides.json", "activity.json")]
    assert validate_media_timebase(asr, slides) == 90
    validate_activity(activity, bundle, output)
    assert bundle["lesson"]["source_video"]["sha256"] == SOURCE_SHA256
    assert len(read_json(output / "asr.unguided.json")["segments"]) == 36
    proofreading = bundle.get("transcript_proofreading")
    if proofreading:
        assert proofreading["source_asr_sha256"] == sha256_file(output / "asr.json")
        assert proofreading["source_video_sha256"] == SOURCE_SHA256
        assert proofreading["human_verified"] is False
        assert [(r["original_text"], r["start_sec"], r["end_sec"]) for r in proofreading["segments"]] == [
            (r["text"], r["start_sec"], r["end_sec"]) for r in bundle["transcript"]]
    assert [(r["start_sec"], r["end_sec"]) for r in slides["slides"]] == [SCREEN_RANGE]
    assert all(SCREEN_RANGE[0] <= r["start_sec"] < r["end_sec"] <= SCREEN_RANGE[1]
               for r in bundle["alignment"] if r["slide_id"])
    for row in read_json(output / "activity/copied_frames.sha256.json"):
        assert sha256_file(output / row["path"]) == row["sha256"]
    known = [f for f in activity["frames"] if f["participation_index"] is not None]
    unknown = [f for f in activity["frames"] if f["participation_index"] is None]
    summary = {"status": "verified", "source_sha256": SOURCE_SHA256,
        "asr_segments": len(asr["segments"]), "unguided_asr_segments_preserved": 36,
        "asr_prompt_source": "same_video_screen_region_ocr", "human_asr_accuracy_evaluated": False,
        "dense_ocr_samples": slides["coverage"]["sample_count"],
        "retained_courseware_intervals_sec": [list(SCREEN_RANGE)],
        "retained_courseware_pages": len(slides["slides"]), "courseware_text": slides["slides"][0]["text"],
        "activity_frames_verified": len(activity["frames"]), "frames_with_rule_index": len(known),
        "unknown_index_frames": len(unknown), "index_min": min(f["participation_index"] for f in known),
        "index_max": max(f["participation_index"] for f in known),
        "source_split": "train", "aric_online_used": False,
        "graph_concepts": len(bundle["graph"]["nodes"]),
        "automatic_proofread_edits": proofreading["edit_count"] if proofreading else 0,
        "automatic_proofread_uncertain_segments": proofreading["uncertain_segment_count"] if proofreading else 0,
        "timebase_validated": True, "all_copied_behavior_image_bytes_verified": True,
        "completed_at": utc_now()}
    atomic_json(output / "unified.verification.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--reuse", type=Path, default=ROOT / "artifacts/ts-chemistry-001-9")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/unified-chemistry-001-9")
    parser.add_argument("--lesson-id", default="unified-chemistry-001-9")
    parser.add_argument("--stage", choices=("all", "prepare", "asr", "build", "verify"), default="all")
    parser.add_argument("--register", action="store_true")
    args = parser.parse_args()
    stages = ("prepare", "asr", "build", "verify") if args.stage == "all" else (args.stage,)
    for stage in stages:
        if stage == "prepare": prepare(args.video, args.reuse, args.output, args.lesson_id)
        elif stage == "asr": transcribe(args.video, args.reuse, args.output)
        elif stage == "build": build(args.video, args.output, args.lesson_id, args.register)
        else: verify(args.output)
