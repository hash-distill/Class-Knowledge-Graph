# CEC4 proofreading and measured comparisons

## Model roles

`configs/correction_llm.json` selects ChineseErrorCorrector4-4B Q4_K_M for transcript proofreading. ASR remains separate; CEC4 receives recognized text. Q&A/concept extraction still use Qwen3-4B from `configs/local_llm.json`.

Weights: [pinned GGUF revision](https://huggingface.co/mradermacher/ChineseErrorCorrector4-4B-GGUF/tree/33952accc8b460769e3820ba0a794f558eec8f0b), file `ChineseErrorCorrector4-4B.Q4_K_M.gguf`, SHA-256 `b86247b60c844140944610cec31806f3c21150c9d53ca91d914dbc23a5c188db`. This project did not train the weights.

On the recorded 8 GB VRAM machine, proofreading verifies process identity, unloads Q&A, and loads the corrector. Q&A is restored on completion or failure. A system file lock protects model tasks; concurrent requests return model-busy instead of switching models mid-request.

## Inspect a lesson

Open the [workbench](http://127.0.0.1:5175/?lesson=unified-chemistry-001-9), then expand the correction-model comparison in speech/courseware. Switch between guarded results and direct output; replay the source audio by timestamp. Model contributions and source-rule contributions are reported separately.

Direct output is for inspection. Graph/Q&A use validated proofreading evidence. Original ASR, timestamps, audio, and video remain unchanged. Complete JSON exports retain comparisons, requests, and measurement provenance.

## Rerun

With existing weights/runtimes, run from `backend/`:

```powershell
.venv-media/Scripts/python.exe scripts/fetch_correction_llm.py --verify-only
./scripts/start_local_llm.ps1
./start_review.ps1
```

Generate a separate comparison file:

```powershell
.venv-media/Scripts/python.exe scripts/prepare_proofreading.py `
  --media-dir artifacts/unified-chemistry-001-9 `
  --video './local-media/Chemistry_001_part9.mp4' `
  --output artifacts/verification/cec4-comparison/manual-cec4-run.json
```

The benchmark serializes models and temporarily occupies local model services. It writes only the comparison directory, preserving lesson files:

```powershell
.venv-media/Scripts/python.exe scripts/run_correction_benchmark.py `
  --media-dir artifacts/unified-chemistry-001-9 `
  --video './local-media/Chemistry_001_part9.mp4' `
  --output-dir artifacts/verification/cec4-comparison `
  --external-dir artifacts/verification/cec4-comparison/external-chinese-lips
```

See [Comparison protocol](CORRECTION_COMPARISON.md). VRAM figures are MiB from whole-device `nvidia-smi` samples, with pre-load baseline, loaded usage, and observed peak. Sampling can miss transient peaks and does not measure exact model-only allocations. Historical benchmark results do not validate subsequently translated English prompts.

## References and data scope

No confirmed human verbatim reference was found for the 90-second chemistry lesson. Same-name TS-Interact text describes rabbit populations rather than amino acids, so it is excluded from accuracy scoring. The main lesson reports edits, terminology occurrence, rule rejections, and measured resources only.

The external small sample uses three speakers/three clips from [official Chinese-LiPS](https://huggingface.co/datasets/BAAI/Chinese-LiPS), about 37 seconds total. It includes slide video, speaker video, and official human transcription. `PPT.mp4` is slide video, not editable PPTX.

External ASR independently recognizes WAV audio; references go only to the scorer. CER aggregates character errors before dividing by total reference characters. Strict and whitespace/punctuation-normalized CER are both retained. Damage to originally correct reference positions and inserted errors are separate. Training-data exclusion has not been established; this is not an independently held-out benchmark.

Chinese-LiPS is CC-BY-NC-SA-4.0. Samples remain local and are excluded from source packages. Retrieval records, digests, and pairing checks are in local `artifacts/verification/cec4-comparison/external-chinese-lips/`.
