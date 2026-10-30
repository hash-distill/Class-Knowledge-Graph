"""Reproducible, inference-only ASR with original-media timestamps."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

MODEL_ID = "Systran/faster-whisper-small"
MODEL_REVISION = "536b0662742c02347bc0e980a01041f333bce120"


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def inspect_video(path):
    path = Path(path).resolve()
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_format", "-show_streams", "-of", "json", str(path)],
        check=True, capture_output=True, text=True, encoding="utf-8",
    )
    probe = json.loads(result.stdout)
    video = next(stream for stream in probe["streams"] if stream["codec_type"] == "video")
    audio = next((stream for stream in probe["streams"] if stream["codec_type"] == "audio"), None)
    return {
        "filename": path.name, "sha256": sha256_file(path),
        "duration_sec": float(probe["format"]["duration"]),
        "size_bytes": path.stat().st_size, "width": video["width"], "height": video["height"],
        "has_audio": audio is not None,
        "video_duration_sec": float(video.get("duration", probe["format"]["duration"])),
        "audio_duration_sec": float(audio["duration"]) if audio and "duration" in audio else None,
        "video_start_sec": float(video.get("start_time", 0)),
        "audio_start_sec": float(audio.get("start_time", 0)) if audio else None,
    }


def valid_completed(path, source_video, config):
    if not Path(path).is_file():
        return False
    try:
        result = json.loads(Path(path).read_text(encoding="utf-8"))
        provenance = result["provenance"]
        return (result.get("status") == "complete"
                and provenance["source_video"]["sha256"] == source_video["sha256"]
                and provenance["config"] == config)
    except (ValueError, KeyError, TypeError):
        return False


def decode_audio(video_path, audio_path, duration):
    """Exclude decoder/codec tail padding beyond the original audio stream clock."""
    if isinstance(duration,bool) or not isinstance(duration,(int,float)) or not math.isfinite(duration) or duration<=0:
        raise ValueError('Audio stream duration must be finite and positive')
    subprocess.run(['ffmpeg','-nostdin','-y','-v','error','-i',str(video_path),'-map','0:a:0','-vn',
                    '-t',f'{duration:.9f}','-ac','1','-ar','16000','-c:a','pcm_s16le',str(audio_path)],check=True)


def run_asr(video_path, output_dir, *, cpu_threads=6, beam_size=3, resume=True, language='zh'):
    """Transcribe the entire same-source audio; only complete verified runs resume."""
    from faster_whisper import WhisperModel
    from huggingface_hub import snapshot_download

    video_path, output_dir = Path(video_path).resolve(), Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    source = inspect_video(video_path)
    if not source["has_audio"]:
        raise ValueError("The registered source video contains no audio stream")
    config = {
        "model_id": MODEL_ID, "model_revision": MODEL_REVISION,
        "device": "cpu", "compute_type": "int8", "cpu_threads": cpu_threads,
        "language": language, "beam_size": beam_size, "vad_filter": True,
        "condition_on_previous_text": True, "initial_prompt": None,
        "audio_sample_rate": 16000, "audio_channels": 1, "audio_time_offset_sec": 0,
        "timestamp_method": "native_faster_whisper_segment_timestamps_after_vad_mapping",
        "decoded_duration_policy": "bounded_to_original_audio_stream_v1",
    }
    destination = output_dir / "asr.json"
    if resume and valid_completed(destination, source, config):
        print("ASR verified complete cache: " + str(destination), flush=True)
        return json.loads(destination.read_text(encoding="utf-8"))
    started = utc_now()
    progress = {"status": "running", "phase": "model_download", "started_at": started,
                "source_video": source, "config": config}
    atomic_json(output_dir / "asr.progress.json", progress)
    print("ASR downloading fixed model revision " + MODEL_REVISION, flush=True)
    # Direct model download never executes repository code.
    model_dir = snapshot_download(
        MODEL_ID, revision=MODEL_REVISION, cache_dir=str(output_dir / "model_cache"),
        allow_patterns=["config.json", "preprocessor_config.json", "tokenizer.json", "vocabulary.*", "model.bin"],
    )
    model_files = {
        p.name: {"sha256": sha256_file(p), "size_bytes": p.stat().st_size}
        for p in sorted(Path(model_dir).iterdir()) if p.is_file()
    }
    audio_path = output_dir / "audio.wav"
    audio_meta = output_dir / "audio.provenance.json"
    audio_config = {"sample_rate": 16000, "channels": 1, "codec": "pcm_s16le", "offset_sec": 0,
                    "duration_policy": "bounded_to_original_audio_stream_v1"}
    audio_verified = (resume and audio_path.exists() and valid_completed(audio_meta, source, audio_config)
                      and json.loads(audio_meta.read_text(encoding="utf-8")).get("sha256") == sha256_file(audio_path))
    if not audio_verified:
        progress["phase"] = "audio_extraction"
        atomic_json(output_dir / "asr.progress.json", progress)
        decode_audio(video_path,audio_path,source['audio_duration_sec'])
        atomic_json(audio_meta, {"status": "complete", "sha256": sha256_file(audio_path),
                                "provenance": {"source_video": source, "config": audio_config}})
    progress["phase"] = "transcribing"
    atomic_json(output_dir / "asr.progress.json", progress)
    print("ASR model ready; transcribing complete audio", flush=True)
    model = WhisperModel(model_dir, device="cpu", compute_type="int8", cpu_threads=cpu_threads)
    segments_iter, info = model.transcribe(str(audio_path), language=language, vad_filter=True,
                                          beam_size=beam_size, condition_on_previous_text=True)
    segments = []
    processing_start = time.monotonic()
    partial_path = output_dir / "asr.partial.jsonl"
    with partial_path.open("w", encoding="utf-8") as stream:
        for segment in segments_iter:
            item = {"id": f"T{len(segments)+1:04d}", "start_sec": round(segment.start, 3),
                    "end_sec": round(min(segment.end, source["duration_sec"]), 3),
                    "text": segment.text.strip(), "source": "faster_whisper",
                    "quality": {"status": "machine_transcript_unreviewed",
                                "avg_logprob": segment.avg_logprob,
                                "no_speech_prob": segment.no_speech_prob,
                                "compression_ratio": segment.compression_ratio}}
            if not item["text"] or item["end_sec"] <= item["start_sec"]:
                continue
            segments.append(item)
            stream.write(json.dumps(item, ensure_ascii=False) + "\n")
            stream.flush()
            progress.update({"segments_written": len(segments), "latest_segment_end_sec": item["end_sec"],
                             "elapsed_transcription_sec": round(time.monotonic() - processing_start, 2)})
            atomic_json(output_dir / "asr.progress.json", progress)
            if len(segments) % 20 == 0:
                print(f"ASR {len(segments)} segments; reached {item['end_sec']:.1f}/{source['duration_sec']:.1f} sec", flush=True)
    provenance = {
        "source_video": source, "config": config, "started_at": started, "completed_at": utc_now(),
        "source": "faster_whisper", "training_performed": False,
        "model_url": f"https://huggingface.co/{MODEL_ID}/tree/{MODEL_REVISION}", "model_files": model_files,
        "packages": {name: importlib.metadata.version(name) for name in
                     ["faster-whisper", "ctranslate2", "huggingface-hub", "av", "onnxruntime"]},
        "audio_sha256": sha256_file(audio_path), "processed_range_sec": [0, source["duration_sec"]],
        "audio_duration_sec": info.duration, "vad_speech_duration_sec": info.duration_after_vad,
        "generator_exhausted": True, "timestamp_offset_sec": 0,
        "coverage_note": "Full audio decoded and ASR generator exhausted. Segment gaps include VAD silence; this is not a word accuracy assessment.",
    }
    result = {"schema_version": "3.0", "status": "complete", "segments": segments,
              "coverage": {"processed_duration_sec": source["duration_sec"],
                           "segment_count": len(segments),
                           "first_segment_start_sec": segments[0]["start_sec"] if segments else None,
                           "last_segment_end_sec": segments[-1]["end_sec"] if segments else None,
                           "sum_segment_duration_sec": round(sum(s["end_sec"]-s["start_sec"] for s in segments), 3)},
              "provenance": provenance}
    atomic_json(destination, result)
    progress.update({"status": "complete", "phase": "complete", "completed_at": utc_now()})
    atomic_json(output_dir / "asr.progress.json", progress)
    print(f"ASR complete: {len(segments)} segments, {destination}", flush=True)
    return result
