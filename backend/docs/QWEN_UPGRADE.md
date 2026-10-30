# Qwen ASR and conservative proofreading

This separate experiment uses Qwen3-ASR-1.7B for audio and Qwen3.5-4B for Chinese proofreading. Models run sequentially. Q&A and concepts retain Qwen3-4B.

## Models and provenance

| Role | Model | Configuration | Output |
| --- | --- | --- | --- |
| Audio recognition | Qwen/Qwen3-ASR-1.7B, BF16 | `configs/qwen_asr.json` | Native transcript and predicted times |
| Alignment | Qwen/Qwen3-ForcedAligner-0.6B, BF16 | Same file | Predicted word times, not human annotations |
| Conservative correction | Qwen/Qwen3.5-4B, Q4_K_M | `configs/qwen35_llm.json` | Structured proposals, accepted edits, rejection reasons |

The Qwen3.5 GGUF is from an unsloth conversion repository. Configuration pins revision, size, and SHA-256 and retains upstream cards. This was inference only, with no training/fine-tuning.

## Four recorded samples

The main sample is the complete 90-second audio of the source-language lesson “认识氨基酸” (“Introduction to amino acids”). Three others are frozen official Chinese-LiPS test clips. New ASR receives audio only, Chinese language specified, without courseware vocabulary or human-reference prompts. Official references enter only scoring.

Old/new ASR each feed Qwen3/Qwen3.5, producing four conditions. Correctors share conservative prompts and generation budgets. Main-lesson old ASR used same-video OCR hints; external old ASR used SeacoParaformer. Main-lesson ASR differences are therefore configuration-specific.

See [Evaluation](QWEN_UPGRADE_EVALUATION.md) for metrics, failures, and replay. Local results are in `artifacts/verification/qwen-upgrade-20261028/`, particularly `evaluation/comparison.md` and `comparison.json`. `status=partial` means missing/failed work, not complete evaluation. Dated results do not validate later English prompt translations.

## Download and independent ASR

The actual ASR environment was `.venv-qwen-asr-local`, separate from media dependencies: PyTorch/torchaudio 2.6.0+cu124 and SymPy 1.13.1. Full versions are in local `asr/environment-freeze.txt`; pinned dependencies are `requirements-qwen-asr-local.txt`. A failed PyTorch 2.7.1 download was not used.

Only required packages were copied from an existing CUDA installation, without inheriting system site-packages. Key copied files were checked against source RECORD entries; the original environment was unchanged. Local `asr/setup-replay.ps1` records reconstruction; only syntax was checked, without building a second environment. Ensure the GPU is free. The Windows runner holds the project model lock and unloads ASR before alignment.

From `backend/`:

```powershell
.\.venv-media\Scripts\python.exe scripts/fetch_qwen35_llm.py
.\.venv-qwen-asr-local\Scripts\python.exe scripts/qwen_asr_remote_inputs.py --output artifacts/verification/qwen-upgrade-20261028/asr/model-inputs.json
.\.venv-qwen-asr-local\Scripts\python.exe scripts/qwen_asr_remote_download.py --config configs/qwen_asr.json --cache artifacts/verification/qwen-upgrade-20261028/asr/model-cache --output artifacts/verification/qwen-upgrade-20261028/asr
.\.venv-qwen-asr-local\Scripts\python.exe scripts/qwen_asr_remote_run.py --config configs/qwen_asr.json --inputs artifacts/verification/qwen-upgrade-20261028/asr/model-inputs.json --output artifacts/verification/qwen-upgrade-20261028/asr --cache artifacts/verification/qwen-upgrade-20261028/asr/model-cache --local-models --gpu 0 --min-free-mib 7000
```

The last line is the recorded local 8 GB configuration. Memory records distinguish device model, PyTorch allocation peak, and whole-GPU samples. Local results must not be labeled A6000 measurements when server connectivity was unavailable.

## Publish a separate lesson

`scripts/build_qwen_upgrade_demo.py` packages new ASR as a separate lesson after checking audio SHA, duration, full coverage, and zero offset. Native text/predicted times are preserved; proofreading is separate.

New ID: `qwen35-chemistry-001-9`; original: `unified-chemistry-001-9`. Same-video courseware, behavior predictions, and frame images are reused with per-image hash checks.

```powershell
.\.venv-media\Scripts\python.exe scripts/build_qwen_upgrade_demo.py --video './local-media/Chemistry_001_part9.mp4'
.\start_trained_demo.ps1 -LessonId qwen35-chemistry-001-9 -RestartApi
```

Proofreading switches to Qwen3.5 and restores Q&A afterward. Launchers check recorded process identity and manage only configured project services.

The collapsible model-experiment section in speech/courseware compares original ASR, new ASR, and conservative correction. They share the audio clock but retain their segmentation. Clicking replays the corresponding audio; full export retains experiment records.

Without an independent human transcript, the lesson has no CER. Accepted-edit counts are not correction accuracy, and source-rule changes are counted separately. The speech upgrade does not add evidence about psychological engagement to visible-behavior predictions.
