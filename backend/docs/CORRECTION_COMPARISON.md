# Correction-model comparison protocol

## Generate reports

`scripts/compare_correction_models.py` reads frozen runs and writes JSON, CSV, and Markdown. It does not call models or write to lesson directories. Runner scripts perform inference and GPU sampling. From `backend/`:

```powershell
python scripts/compare_correction_models.py `
  --media-dir artifacts/unified-chemistry-001-9 `
  --run qwen=artifacts/verification/cec4-comparison/qwen-run.json `
  --run cec4=artifacts/verification/cec4-comparison/cec4-run.json `
  --deterministic-run artifacts/verification/cec4-comparison/source-only.json `
  --output-dir artifacts/verification/cec4-comparison/report
```

Outputs: `comparison.json`, `comparison.md`, `segments.csv`, and `proposals.csv`. Segment CSV retains every source segment/output layer. Proposal CSV records candidates, guard decisions, and failed source localization. JSON also retains the full source-rule proposal ledger.

## Frozen inputs and output layers

The main experiment freezes 28 segments, times, and text from `unified-chemistry-001-9/asr.json`. Auxiliary evidence includes 36 unguided-ASR segments, 30 same-audio secondary-ASR segments, and observed slide vocabulary at 43.5–47.5 s. The lesson retains 181 behavior observations; they do not enter text-accuracy scoring.

Run models sequentially on the same GPU with identical source segments, auxiliary evidence, and guard version. Structured comparisons also require the same system/prompt, temperature, seed, context, token budget, and request format. Save actual requests/responses; temperature zero does not guarantee bitwise reproducibility.

Separate layers are:

- Raw proposals applied at original-text coordinates before semantic guards. Ambiguous repeated substrings/overlaps retain failures instead of guessed positions.
- Guarded text: all `proofread_transcript` `segments`.
- Source-only rules: the same vocabulary, secondary-ASR, and repetition/context rules with no model proposals. Accepted origins must not include model candidates.
- Native whole-text correction: raw response and parsed correction. Its prompt differs from structured proposals and must be reported as a different condition.

If source-only rules and both models produce the same 13 edits, those edits cannot all be credited to the models. Report accepted origins and additional changes relative to rules; additional changes still require listening checks.

## Run contract

`--run LABEL=PATH` can read a direct `proofread_transcript` result. To retain conditions/resources, use a wrapper:

```json
{
  "label": "cec4-controlled",
  "track": "controlled_json",
  "conditions": {
    "temperature": 0,
    "seed": 42,
    "n_ctx": 24576,
    "max_tokens": 5000,
    "request_format": "json_schema"
  },
  "runtime": {
    "scope": "single_model_loaded_correction_only",
    "wall_time_sec": 0,
    "baseline_device_used_mib": 0,
    "peak_device_used_mib": 0,
    "sampling_interval_sec": 0.2,
    "measurement_source": "nvidia-smi",
    "gpu_name": "fill from measurement"
  },
  "result": {}
}
```

Zero values illustrate types only; actual runs must use measurements. `runtime` is preserved verbatim, with absent records left empty. Do not infer VRAM from parameter count. Distinguish loading, correction, full-pipeline, and other-process usage; sampled peaks can miss spikes.

`result` requires `source_asr_sha256` and actual source digests. Video uses `source_video_sha256`; independent WAV uses `source_audio_sha256`, also recorded as original-ASR provenance `source_audio.sha256`. Guarded results need all segments with original `transcript_id`, `original_text`, `start_sec`, `end_sec`, and output `text`. Mismatched source hashes, text, order, or times are rejected.

Version 8 `model_proposals` retain parsed answers, `system_digest`, `prompt_digest`, target segments, and model metadata. Ledger rows contain `transcript_id`, `channel`, `candidate`, `disposition`, and `reason`; accepted edits have `origin`. Historical v7 lacks complete proposals, so its raw-model layer is unavailable.

Native output can use:

```json
{
  "track": "native_text",
  "native_outputs": [
    {
      "transcript_id": "T0001",
      "original_text": "Frozen original segment text",
      "corrected_text": "Parsed native correction",
      "metadata": {"raw_response": "Preserve the original response"}
    }
  ],
  "result": {
    "source_asr_sha256": "same immutable ASR sha256",
    "source_video_sha256": "same source video sha256"
  }
}
```

Native output must cover every original segment. A guarded layer exists only if guarded `segments` are also supplied. Label custom prompts as project prompts. A translation of an upstream prompt must be labeled as translated, rather than claimed to be the exact upstream configuration.

## Main-lesson quality limits

The chemistry lesson has no human verbatim transcript. Original/secondary ASR and OCR can all err, so the report does not compute true correction rate, harmful-edit rate, or CER.

Fourteen terminology groups are in `tests/fixtures/cec4/chemistry-001-9-terminology.json`. They retain actual sources but were frozen after inspecting this lesson and have selection bias. They test canonical spelling occurrence/retention/change only. Nested strings can count twice: source-language examples “氨基” (amino group) and “氨基酸” (amino acid). Fourteen groups are not fourteen human-labeled errors.

Character changes use per-segment Unicode-codepoint Levenshtein alignment. Adjacent edits form spans with input/output coordinates. Edit distance and span count are not corrected-error counts.

Changes to digits, Chinese numerals, negation, formula operators, and letter sequences raise review flags. Source-language examples “二发→阿尔法” and “未经→味精” may also trigger flags; the original strings are retained to describe linguistic guard behavior. Flag/rejection counts are not confirmed harmful edits.

## Independent references and external checks

Enable `--reference` only with a same-video, same-clock human/official verbatim reference aligned with every original segment:

```json
{
  "reference_kind": "official_verbatim",
  "human_verified": true,
  "independent_of_model_outputs": true,
  "used_as_model_input": false,
  "clock": "source_video_seconds",
  "source_video_sha256": "same source video sha256",
  "provenance": {"source": "official URL and immutable annotation hash"},
  "segments": [
    {"transcript_id": "T0001", "start_sec": 0, "end_sec": 12.3, "text": "Independent verbatim reference"}
  ]
}
```

These are checkable declarations, not independent authentication by the comparator. Similar-topic slides/plans/subtitles are not lesson ground truth. If only a whole-audio official reference exists, use one whole-audio evaluation segment rather than inventing word times from reference text.

Reference scoring reports strict CER and NFKC-normalized CER removing whitespace/Unicode punctuation while retaining case/math symbols. Both sum errors before dividing by total reference characters; do not average segment CERs. Independent WAV references use `time_base: "source_audio_seconds"` and `source_audio_sha256`; silent slide video does not require an invented audio source.

Strict/normalized repair and damage counts compare original ASR and corrections against the reference. Previously erroneous reference positions that match afterward are repaired; previously correct positions that become erroneous are damaged. Insertions are separate. Denominators are originally erroneous/correct positions respectively. Repeated text can have several optimal alignments, so results carry `alignment_dependent=true`. This differs from the fraction of harmful proposals.

Three official Chinese-LiPS test samples are an external check, not replacements for the chemistry lesson. `external-chinese-lips/model-inputs.json` is model-readable; `gold-references.json` is scorer-only and must not enter ASR hints, vocabulary, or correction context. Report separately because sources differ. Small size/unknown model-training data prevent claims of training exclusion or whole-dataset performance.

Use separate media/output directories and `--terminology empty-terms.json` containing `{"groups":[]}`. Aggregate sample CER numerators/denominators. After ASR is frozen in `external-chinese-lips/evaluation/<id>/asr.json`, prepare scorer references:

```powershell
python scripts/prepare_external_correction_references.py `
  --external-dir artifacts/verification/cec4-comparison/external-chinese-lips
```

The tool checks WAV/annotation hashes and derives duration from sample rate/frame count. Each ASR needs one complete `T0001`, from zero to actual WAV duration. It writes per-sample `reference.json` and evaluation `empty-terms.json` while preserving ASR. Run only at scoring time.

## Tests

```powershell
python -m pytest tests/test_correction_comparison.py -q
```

Tests use independent synthetic fixtures without weights/model calls. Actual run completeness, source preservation, and GPU measurements require inspecting artifacts.

## Sequential inference and GPU measurement

With pinned weights/runtimes installed:

```powershell
.venv-media/Scripts/python.exe scripts/run_correction_benchmark.py `
  --media-dir artifacts/unified-chemistry-001-9 `
  --video './local-media/Chemistry_001_part9.mp4' `
  --output-dir artifacts/verification/cec4-comparison `
  --external-dir artifacts/verification/cec4-comparison/external-chinese-lips `
  --request-timeout 900
```

The runner evaluates source-only rules, then loads Qwen and CEC4 sequentially. Before loading it stops recorded model roles through PID/path/start-time-verified helpers. Each model runs structured full-lesson proofreading, 28 native segments, and three external native clips. Historical runs used the same verified upstream correction prompt, temperature 0.1, and 1,000-token limit. This differed from model-card deterministic decoding and is a project comparison setup. Current translated prompts must be identified as translations; historical metrics do not verify them.

Responses are saved immediately with raw output, thinking text, usage, finish reason, parameters, and digests. Failed segments retain original text and explicit failure status. `--native-only` / `--controlled-only` fill separate tracks. `--force-controlled` archives existing results first. Changed auxiliary inputs, guards, or model configurations invalidate old controlled caches. Multi-run native resumes retain individual sampling files/peaks before aggregating observed peaks.

GPU target sampling is 0.2 s; record actual intervals, full device name, UUID, driver, idle/loaded usage, and phase peaks. A generation sampling gap over five seconds invalidates latency comparison while retaining raw times. Windows temporarily prevents idle sleep for the active thread and restores it on exit, without changing permanent power policy.

A historical 2026-10-28 attempt was affected by about seven hours of sleep; another timed out at 180 s. Both remain local in `attempts/`. The formal comparison used new completed runs for both models with 900 s HTTP limits and `gpu-samples-20261028-080448.jsonl`. Interrupted windows are excluded from formal latency conclusions. Local verification artifacts are excluded from source distributions.
