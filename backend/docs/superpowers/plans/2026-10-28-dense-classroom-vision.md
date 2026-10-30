# Dense classroom vision improvement

**Objective:** improve real detection/behavior evidence for the 90-second Chemistry_001_part9 recording on an 8 GB GPU. Compare against preserved original observations and the PPT's seven-state goal. Detection, visible state, action evidence and psychological engagement are different quantities.

**Current evidence:** old pipeline gates all person boxes on YOLO26n-pose tracks; the later pose-enrichment pass cannot add missed people. The behavior checkpoint is YOLOv9e with only three classes. Existing current-frame coverage divides known states by detected people and cannot measure missed-person recall. A rear-wall reflection creates duplicate people and needs an explicitly reviewed exclusion region.

## 1 Baseline and data audit

- Preserve vision source/configs, frontend/backend source and original lesson/activity artifacts with SHA-256.
- Extract PPT requirements and inspect actual source frames before choosing inference settings.
- Audit local SCBehavior seven-state images/labels, actual class IDs, duplicate images and available splits. Keep source datasets read-only.
- Use a dedicated CUDA environment, pinned official weights and fixed sample times. Record latency and peak allocated GPU memory.

## 2 Detection experiment and regression protection

- Add tested source-coordinate crop translation, overlap suppression, reviewed exclusion regions and one-to-one association in a small independent dense-inference module.
- Compare old pose-gated detections, larger full-image inference, overlapping crops, and person/behavior fusion on fixed original frames. Count visible heads independently of model output on reviewed sample frames; distinguish reflections, people and uncertain behavior labels.
- Separate current detections from stable tracks. Unconfirmed tracks must not erase actual boxes; scene-local tracking must never imply identity or bridge cuts.
- Keep all detection provenance. Unknown behavior remains visible and unscored.

## 3 Lightweight seven-state behavior model

- Prepare reproducible image/label manifests with exact class mapping and cross-split duplicate checks.
- Fine-tune a small official YOLO checkpoint on local seven-state annotations using this GPU, without the target classroom frames in training.
- Select on validation only, report actual held-out results, model bytes, runtime and source limitations. A training completion is not proof of target-classroom accuracy.
- Integrate only observed usable outputs; keep the earlier checkpoint and results for rollback. Do not advertise PPT accuracy/speed numbers without measured evidence.

## 4 Application integration

- Extend activity validation, evidence generation and frontend types for new observed labels and detection provenance.
- Distinguish detector population, labeled-state support, scored-action support and unknown actions in the UI. Do not present no recognized action as validated zero engagement.
- Keep the blue design, title punctuation rule, equal column shells and single stable player.
- Re-run the complete source video into versioned artifacts; preserve original source media, ASR, teacher edits and old activity outputs. Publish only after schema/source checks and sample review.
- Synchronize frontend and desktop resources and provide a fresh service if the existing frozen API cannot read the new contract.

## 5 Acceptance

- Meaningful unit regressions for crop coordinates, duplicate suppression, exclusion regions, unmatched detections, cuts and unknown/zero semantics.
- Vision, API and frontend checks appropriate to changed modules, then browser inspection of actual annotated frames and knowledge/time navigation.
- Record exact measurements and limited evaluation scope under `artifacts/verification/vision-improvement-20261028/`. Report remaining unsupported PPT goals explicitly.
