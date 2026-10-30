# Vision change record: Introduction to amino acids

> This historical record describes the scoring state when the localization changes were completed. [Overall behavior engagement](CLASS_ENGAGEMENT_20261029.md) and [manual review](VISUAL_REVIEW_20261029.md) were added afterward. The current interface no longer displays all scores as unknown because of the old threshold. Historical values below remain for traceability; localization and evaluation limitations still apply. The original lesson title is “认识氨基酸”.

## Delivery state at the recorded revision

Person localization, state display, and review views were updated to inspect seven-class model candidates. The model had not passed human-annotated acceptance for this lesson. Engagement results did not meet the effect described in the project proposal slides.

Undetected actions were not interpreted as low engagement. Under the then-current rules, all 181 sampled frames had an unknown visible-behavior reference score. State composition, original frames, and earlier temporal-action scores remained available for inspection.

## Cause and changes

The original pipeline used pose tracks as person boxes. Small, occluded, and unconfirmed tracks could be missed. Later pose enrichment only processed existing boxes and could not recover missed people.

The replacement combines YOLO26n person detection and YOLO26n-pose. Both process full frames and overlapping tiles; results are fused and deduplicated in source-image coordinates. Behavior classification no longer determines whether a person gets a box. Regression cases use real coordinates for tile-edge partial bodies and duplicate boxes around the same head.

A MobileNetV3 Small classifier uses per-person crops and 3.15 MB weights. Its seven classes are writing, reading, looking up, turning the head, raising a hand, standing, and discussion. Low-confidence results remain unknown; geometric poses are separate. A hand-raising candidate also becomes unknown when reliable pose evidence explicitly contradicts it.

The rear-wall mirror exclusion is bound to this video segment's SHA-256. Only candidates meeting its region rules are excluded; the configuration is not transferable to every lesson. Boxes may still include teachers, occlusion fragments, and false detections. Their count is not class attendance.

## Recorded output for this lesson

| Source time | Previous candidate boxes | New candidate boxes |
| --- | ---: | ---: |
| 0 s | 21 | 39 |
| 40 s | 21 | 43 |
| 75 s | 23 | 43 |
| 89.9 s, projected screen | 0 | 0 |

The video contains 181 sampled frames, four shot segments, and 7,201 person observations across frames. Of these, 3,817 classifications passed the current confidence and pose-conflict checks. These are program-output counts, not detection recall, classification accuracy, or unique student counts.

A rerun on an RTX 4060 Laptop GPU took about 169 seconds for a 90-second recording. Peak PyTorch allocated memory was about 143 MiB, excluding drivers, the desktop, and other processes. This offline-review mode had not reached real-time speed.

## Evaluation limitations

The local SCBehavior subset had 400 images: 313 training, 47 validation, and 40 test images. Given annotated boxes, the classifier scored Macro-F1 0.81694 and accuracy 0.82473 on 833 test crops.

Training and test sets include adjacent frames from the same classroom. A hash audit found 120 near-neighbor training/test pairs and two validation/test pairs. Metrics may be optimistic and do not represent independent classrooms, this recording, or end-to-end detection accuracy. Only eight standing objects were in the test set, insufficient to establish stable performance for that class.

A separate seven-class YOLO26n candidate produced projected-text false detections and hand-raising errors in this lesson. It remains an experimental artifact and was not published as the active per-person classifier.

## Historical scoring and interface

The earlier visible-behavior reference retained weights for reading, writing, looking up, raising a hand, and discussion. Turning, standing, unknown, and geometric head-up states were unscored. A numeric result required at least three objects and scored support covering 60% of detected objects.

The proposal's `0.3 × previous + 0.7 × current` smoothing applied only to valid, continuous observations in one shot. Missing, cut, and low-support intervals reset it; unknowns were not zero-filled. No frame in this lesson met those historical conditions, so engagement acceptance was still incomplete at that revision.

The interface added compact IDs, state text, border-only overlays, and fullscreen inspection. Clicking an object showed provenance and evidence. State heatmaps and temporal-action scores were separate. Overview and behavior views marked candidates as unvalidated; the blue theme, responsive layout, and source-video timeline were retained.

## Historical validation and files

Backend tests reported 320 passed and six skipped; vision tests 82 passed; frozen-backend lifecycle tests 13 passed and one skipped. Frontend contract tests and the production build passed. Browser checks of the actual candidate package covered box counts, provenance, display modes, fullscreen, time navigation, no-person shots, model boundaries, and mobile layouts.

Previous activity files, bundles, and reports were backed up. Hashes of 358 original evidence files, the registry, and source video remained unchanged. A new backend and installer were built; the installation wizard was not run locally.

Local records are in `artifacts/verification/vision-improvement-20261028`: `activity-dense-final.json`, `source-preservation.json`, `split-overlap-audit.json`, `crop-training-report.json`, and `browser-preview/acceptance.json`. Previous outputs are in `publish-backup`; training scripts and rejected candidates were retained. These local artifacts are excluded from source distributions.

## Further acceptance

Annotate independent samples from this and other classrooms, including misses, duplicates, mirrors, roles, and visible behavior. Split by classroom or recording, report localization and state-classification metrics separately, then evaluate the rule estimate. Current candidates do not measure psychological engagement, mastery, or teaching effectiveness.

## Implementation references

Overlapping tiles refer to [Ultralytics documentation](https://docs.ultralytics.com/guides/sahi-tiled-inference/); class/data provenance refers to the [SCBehavior repository](https://github.com/20191864136/SCBehavior-Dataset). Actual parameters, weight hashes, and results are recorded in the local evidence files.
