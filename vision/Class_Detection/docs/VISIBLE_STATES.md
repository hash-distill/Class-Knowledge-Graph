# Current-frame behavior and posture states

This describes the earlier three-class pretrained/pose-derived path, retained for traceability. The [module README](../../README.md) and [current scoring](../../../backend/docs/CLASS_ENGAGEMENT_20261029.md) describe later workbench behavior.

`student_state` is the frame's primary state; `posture_tags` separately retain simultaneous poses. A reading prediction can also have a lowered-head tag. Original `action`, `label`, confidence, tracks, and temporal rule scores remain unchanged.

| State | Evidence | Interpretation |
| --- | --- | --- |
| `read` / `write` / `raise_hand` | Three actual classes in pinned SCB weights | Model clues, not human-annotated lesson accuracy |
| `head_up` | Confident nose, at least one eye, both shoulders; nose at least 0.28 shoulder-width above shoulder midpoint | Head/shoulder geometry, not listening/gaze at the teacher |
| `head_lowered` | Same points; nose/shoulder vertical gap at most 0.16 shoulder-width | Cannot distinguish reading, writing, or other activity alone |
| `hand_near_head` | Confident wrist within 0.48 shoulder-width of nose | No phone detection or sleep inference |
| Geometric `raise_hand` | Wrist 0.25 shoulder-width above nose and 0.8 above shoulders; elbow above shoulders | Separate strict raised-wrist rule, not the original model prediction |
| `standing` | Both shoulders and bilateral hips/knees/ankles; hip/knee/ankle descend; knee ≥160°, hip ≥155° | Standing geometry, not speaking or role identity |
| `unknown` | Missing, low-confidence, ambiguous, or unmatched boxes | Does not carry the previous frame or mean low participation |

Used keypoints need confidence ≥0.5 and shoulder width ≥12 source pixels. These are engineering thresholds, not lesson-calibrated classifiers; point confidence is not state accuracy. Primary states prefer original SCB outputs before geometry. Geometry records point IDs, measurements, rule descriptions, and provenance.

## Frame statistics contract

- `student_state_counts`: mutually exclusive primary states. Detections without stable tracks are added as unknown, so the sum equals `detected_people` and may exceed boxed `participants`.
- `participation.active_people` / `active_fraction`: explicit reading/writing/hand-raising count divided by detected people. Unrecognized does not mean not participating.
- `participation.observed_people` / `coverage`: people with behavior or posture evidence, separate from scoring support.
- `participation.index`: mean explicit-action weights (reading 0.80, writing 0.85, hand raising 0.95) ×100. Poses are unscored; `scored_people` / `score_coverage` show support. It is not psychological-engagement probability.
- With no people, fractions/scores remain `null`. With people but no recognized actions, fraction zero means only that none of these three actions was detected in that frame.

No reliable role classifier is present. Population is `visible_people_roles_unverified`: detected counts are not enrolled attendance, and teachers, others, reflections, or occlusion fragments are not automatically students. Tracks reset after cuts.

## Historical same-video derivation

`backend/scripts/derive_visible_states.py` re-extracts pose at all 181 existing timestamps and uses one-to-one IoU ≥0.35 association to original boxes, preserving behavior labels/scores. Provenance includes video/activity digests, model digests, raw pose-cache digest, and completion time.

The chemistry source SHA-256 is `bd955b05ac65bcae85890ecda939e8e027389f3e739f1ea3775299db5231df1d`. Recorded outputs: at 0 s, 20 head-up clues; at 35 s, 11 reading, one writing, three head-lowered, one head-up, four unknown; at 40 s, 15 reading, four writing, two unknown; at 60 s, 20 head-up and two unknown. At 45/89.9 s no usable people were detected and scores stayed unknown. No sample met the standing rule.

These are real-video model/rule outputs, not verified accuracy or independent held-out evaluation. Prioritize timestamps, visible counts, behavior/posture states, and source-image replay for inspection. Generated frames and results are excluded from source distributions.
