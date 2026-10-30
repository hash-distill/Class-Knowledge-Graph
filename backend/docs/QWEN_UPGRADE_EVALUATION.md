# Qwen ASR/correction upgrade evaluation

## Purpose and inputs

`scripts/compare_qwen_upgrade.py` compares ASR and correctors on the same four audios: the complete 90-second WAV of `unified-chemistry-001-9` and three frozen official Chinese-LiPS test clips. It checks WAV SHA-256 and preserves each ASR's original segmentation, times, and provenance.

| Item | Previous input | Upgrade input |
| --- | --- | --- |
| Main ASR | OCR-hinted Whisper, 28 segments | Actual Qwen3-ASR-1.7B output |
| External ASR | Independent SeacoParaformer output | Actual Qwen3-ASR-1.7B output |
| Corrector | Qwen3-4B | Qwen3.5-4B |

Each audio has four old/new ASR × old/new corrector conditions. Main-lesson unguided Whisper is diagnostic-only at `diagnostic_asr_runs.old_unguided`. Historical native correction used different prompts and cannot fill any controlled-comparison cell.

## Input/scoring separation

`prepare` and `run` do not read official references, gold labels, or human annotations. They read input manifests, audio identities, ASR, and runtime configuration. Native correction receives only one ASR's complete concatenated text, in its own segment order, without separators, pre-rewriting, OCR, a second recognizer, or answers.

Both models share `NATIVE_SYSTEM`: preserve spoken style, questions, repetition, and incomplete sentences; correct only clear contextual recognition errors. Conditions are `temperature=0.1`, `max_tokens=2000`, thinking disabled, plain text. No seed is sent, so server defaults affect regeneration. Saved inputs, outputs, and scores remain hash-verifiable. Historical measured runs used the prompts recorded in those files; they do not verify later English translations.

Only `report` reads scorer-only references. It validates pinned official annotation hashes and agreement between official text, gold text, and scoring references. References must cover the full WAV and declare independence from model outputs and exclusion from model inputs.

## Metrics

Recognizers can use different T IDs/segmentation. Score whole audio, without pairing their T IDs. The denominator is the official whole-audio reference character count.

| Metric | Definition |
| --- | --- |
| Strict CER | Exact Unicode-codepoint Levenshtein distance, retaining spaces/punctuation |
| Normalized CER | NFKC, then remove whitespace and Unicode punctuation; no number/synonym/spoken-form conversion |
| Repaired reference positions | Previously erroneous aligned reference positions that become correct |
| Damaged correct positions | Previously correct aligned reference positions that become erroneous |
| Insertions | Errors without a reference position, counted separately before/after correction |

Repeated text can have multiple optimal alignments. Tie order is substitution, deletion, insertion. Repair/damage rates depend on this alignment and are not human listening judgments.

External micro-averages use the same sample subset with all four completed cells and disclose IDs/denominators. Missing/failed text and metrics are `null`, never empty or original-text substitutes. Reports include raw-ASR baselines and CER changes with the other factor fixed; lower CER changes are better. Three clips and unknown training overlap limit conclusions.

The main lesson lacks an independent, same-source human verbatim transcript. CER, repair, and damage remain unevaluated. Other machine outputs, courseware, or same-name wrong-topic labels cannot replace it.

## Run order

From `backend/`, `run` calls already-started endpoints; the operator manages model switches/GPU ownership. `prepare` retains missing new-ASR slots; rerun it when ASR completes, then fill missing requests. After lesson packaging, use `prepare --new-primary-asr artifacts/qwen35-chemistry-001-9/asr.json`. The script checks native text, times, IDs, and source hashes before/after wrapping, binding guarded production and evaluation to the same unchanged ASR.

```powershell
.\.venv-media\Scripts\python.exe scripts/compare_qwen_upgrade.py prepare
.\.venv-media\Scripts\python.exe scripts/compare_qwen_upgrade.py run --asr old --corrector old_qwen --profile configs/local_llm.json
.\.venv-media\Scripts\python.exe scripts/compare_qwen_upgrade.py run --asr qwen3 --corrector old_qwen --profile configs/local_llm.json
# Switch the managed model service to Qwen3.5 before these requests.
.\.venv-media\Scripts\python.exe scripts/compare_qwen_upgrade.py run --asr old --corrector qwen35 --profile configs/qwen35_llm.json
.\.venv-media\Scripts\python.exe scripts/compare_qwen_upgrade.py run --asr qwen3 --corrector qwen35 --profile configs/qwen35_llm.json
.\.venv-media\Scripts\python.exe scripts/compare_qwen_upgrade.py report
```

Completed records with identical inputs/model/conditions are reused. Changed hashes/conditions cannot overwrite them. `--retry-failed` retries only failures and archives previous failed records.

Local records: `artifacts/verification/qwen-upgrade-20261028/evaluation/`. `run-plan.json` records inputs; `corrections/<corrector>/<ASR>/<sample>.json` stores requests; `comparison.json`, `comparison.md`, and `segments.csv` retain results. `gpu-*.jsonl` samples the whole GPU, not exclusive process memory, and can miss transient peaks. Pre-request loaded-model usage is separate; idle baseline/loading time were not measured.

## Guarded production results

Native whole-audio output is an unvalidated hypothesis for this evaluation. It has no original word times and cannot be assigned directly to original T IDs. Production continues to use structured proposals and provenance rules.

Optionally append separate guarded runs:

```powershell
.\.venv-media\Scripts\python.exe scripts/compare_qwen_upgrade.py report --guarded-run old:old_qwen=old-model-guarded.json --guarded-run old:qwen35=new-model-guarded.json
```

Allowed `ASR:corrector` values are `old`/`qwen3` and `old_qwen`/`qwen35`. Appended records retain candidates, rejection reasons, accepted origins, and guarded text. Changes shared by source rules and the model must not all be attributed to the model.
