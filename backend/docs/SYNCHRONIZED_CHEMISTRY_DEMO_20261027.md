# Synchronized chemistry demo — 2026-10-27

## Data and purpose

Local `artifacts/unified-chemistry-001-9` aligns speech, recorded slides, and behavior from the same 90-second `Chemistry_001_part9.mp4`. Video SHA-256: `bd955b05ac65bcae85890ecda939e8e027389f3e739f1ea3775299db5231df1d`.

The clip belongs to the TS-Interact training set and is for functional demonstration only. Behavior reuses completed same-source inference and is not new held-out evaluation. ARIC is not used for online inference. Original `ts-chemistry-001-9` remains unchanged. Media, model files, and generated lesson artifacts are excluded from source releases.

## ASR

`asr.json` has 28 newly inferred machine segments using cached `Systran/faster-whisper-small@536b0662742c02347bc0e980a01041f333bce120`. The script checks local model hashes without downloads.

Decoding hints came from observed full-resolution regional OCR: “认识氨基酸” (introduction to amino acids), “甘氨酸” (glycine), “丙氨酸” (alanine), and “谷氨酸” (glutamic acid). These source-language terms are preserved as exact evidence; only internal whitespace was merged. `asr_prompt_vocabulary.json` records vocabulary, source seconds, and IDs. They are hints, not reference dialogue.

Settings: CPU int8, four threads, `beam_size=5`, `vad_filter=True`, `condition_on_previous_text=False`, `repetition_penalty=1.15`, `initial_prompt=None`, and observed vocabulary as `hotwords`. One complete decode took 76.5 s including loading; load affects timing.

`asr.unguided.json` retains the original 36 unguided segments byte-for-byte; `bundle.json` also includes them in `transcript_unguided`. The separate `asr.initial-prompt-first-pass.json` repeated “氨” (the original Chinese character) and was not selected for display.

The chosen output still includes source-language errors such as “二发氨基酸” and “缩疾”. No dialogue was manually rewritten, and no WER/CER was computed. Label it courseware-vocabulary-guided machine transcription. Shared vocabulary provenance means agreement with slides is not independent transcription validation.

## Observed slides

One-second sampling plus the video end processed 91 frames. `raw_ocr.jsonl` retains all OCR; `slides.dense-unfiltered.json` retains unfiltered screen segmentation.

Projection appears only at 43.5–47.5 s. Additional half-second boundary frames are stored in `screen_observations`: 43.0/47.5 s show students, and 43.5–47.0 s show the same page. The retained interval has 0.5 s sampling uncertainty.

Regional full-resolution OCR selected 44.5 s by the sum of mean regional confidences. It recognized the four terms listed above. Confidence selected a representative frame and is not character accuracy. Chemical structures should be inspected in the original `slides/S0001.jpg`.

`slides.json` retains only that observed page. Other shots use `no_slide_evidence`; the page is not extended into student footage. Original courseware files were unavailable. Companion PPTX/PDF must remain labeled as compilations of recorded observations.

## Behavior

The script verifies the video hash before reusing behavior. It copies and SHA-256-checks all 181 images; `activity/copied_frames.sha256.json` retains the manifest. It changes only lesson media URL prefixes and adds reuse provenance, preserving predictions, boxes, track IDs, and times.

Under the historical rules, 27 frames had valid indexes from 80 to 81.5012; 154 were `null`. Behavior is not visible during slide shots. These rule outputs do not measure psychological engagement, mastery, or teaching effectiveness. See [current overall scoring](CLASS_ENGAGEMENT_20261029.md) for the later derived method.

## Graph display

The original model proposed 12 candidates, including the ASR errors quoted above. Pure function `filter_graph_with_observed_courseware` retains only already-proposed labels also present in observed-slide OCR. Matching ignores whitespace only, without character substitution or new terms.

The recorded read-only result retained five source-language candidates: “甘氨酸”, “丙氨酸”, “谷氨酸”, “氨基酸”, and “氨基”, with ten same-block co-occurrence edges. Filtering removes dangling edges, match concept IDs, and report entries and updates counts/summary. Speech and behavior remain equal to their originals.

The function returns a new lesson and original-candidate archive. The pipeline stores `graph-candidates-original.json` under the lesson write lock before publication. Provenance records the display rule. Literal support means a term occurs in slide text; it does not establish teacher confirmation or chemistry accuracy.

`build()` applies this by default. The first build archives all candidates and caches filtered `graph.json`. Cache hits continue to show five and retain the original 12. With independent courseware, the same write lock covers base filtering, courseware overlay, a second filter, and publication.

```python
from scripts.build_unified_chemistry_demo import filter_graph_with_observed_courseware

# base is the lesson before independent courseware overlays; this function writes no files.
filtered_base, archive = filter_graph_with_observed_courseware(base)

# Save archive and filtered_base under one lesson write lock.
# Filter after applying courseware to avoid introducing new non-model candidates.
final, _ = filter_graph_with_observed_courseware(overlaid, candidate_archive=archive)
```

## Replay and verification

From `backend/`, with the actual local source video:

```powershell
& '.\.venv-media\Scripts\python.exe' '.\scripts\build_unified_chemistry_demo.py' `
  --video '<absolute-path-to-Chemistry_001_part9.mp4>'
```

By default the script uses a new directory, preserves the original lesson, and does not change `configs/lessons.json`. Native-builder registration uses a temporary directory, removed afterward. Register the new ID/set entry points only after checking outputs.

Stages: `--stage prepare`, `asr`, `build`, or `verify`. Preparation needs media dependencies and local Tesseract Chinese data. Fresh concept extraction needs a running local language model. Decoding uses [faster-whisper's official API](https://github.com/SYSTRAN/faster-whisper); the original run checked local loading, hints, and conditional decoding through Context7.

With complete ASR/screen samples, `--stage build` reruns neither ASR nor GPU inference. A 2026-10-27 check copied real evidence into a temporary directory, rebuilt twice from saved candidates, and verified the result: five displayed candidates, 12 archived, 181 behavior frames. Both builds plus verification took 3.749 s, with zero ASR/live-LLM calls and unchanged speech hashes. This checks cached reconstruction; fresh extraction still requires the model service.

Verify an existing package:

```powershell
& '.\.venv-media\Scripts\python.exe' '.\scripts\build_unified_chemistry_demo.py' `
  --video '<absolute-path-to-Chemistry_001_part9.mp4>' --stage verify
```

`unified.verification.json` records actual counts. Checks cover shared hashes, zero-offset clocks, complete sampling, screen intervals, behavior-image hashes, and preservation of 36 original ASR segments. Passing establishes those engineering constraints, not human-validated ASR, OCR, behavior, or concept accuracy.
