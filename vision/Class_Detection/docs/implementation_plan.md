# Archived CV pipeline refactor plan

Version 2.0, dated 2026-04-15. The historical priority was to establish the CV stack before frontend/graph work. This is a planning artifact, not the current implementation inventory or evidence of achieved metrics. See the [current vision module](../../README.md).

## Original assessment

| Issue | Recorded severity | Historical assessment |
| --- | --- | --- |
| ARIC unavailable for RGB detector training | Critical | Available files contained pre-extracted features without original images |
| Core placeholders | Critical | RTMPose/ST-GCN/PnP/OCR were stubs at that revision |
| YOLO version mismatch | Major | Implementation used YOLO11n while the proposal named YOLO26 |
| Deep package structure | Moderate | Seven packages held only one/two files each |

## Planned decisions

| Area | Earlier plan | Proposed replacement | Recorded rationale |
| --- | --- | --- | --- |
| Detector | YOLO11n | YOLO26 | User-selected family; planned edge deployment |
| Main data | Unavailable ARIC RGB | SCB-Dataset5 | Intended YOLO-format detection source |
| Pose | RTMPose/mmpose | YOLO26-pose, optional rtmlib | Shared stack and lighter fallback |
| Tracking | Separate ByteTrack | Built-in Ultralytics tracker | Existing BoT-SORT/ByteTrack integration |
| Dependencies | mmpose/mmcv/mmdet | Ultralytics + rtmlib | Reduce conflicts |

The original plan attributed 7,428 images, 106,830 annotations, and 20 categories to [SCB upstream](https://github.com/Whiffe/SCB-dataset). These were planning assumptions, not a completed local audit. Actual counts/IDs must come from the selected data revision/configuration.

| Proposed group | Labels |
| --- | --- |
| Individual student behavior | hand_raising, read, write, bow_head, turn_head, answer, yawn, stand, lean_desk, use_phone |
| Group behavior | discuss, clap, use_computer, talk |
| Teacher behavior | teacher, guide, board_writing, stage_interact |
| Environment | blackboard, screen |

Auxiliary sources were pretrained COCO-keypoint pose, ARIC's 32-class taxonomy for comparison, and optional collected classroom data for adaptation/demo. Planning statements do not grant redistribution rights.

## Proposed architecture

~~~text
Video (1080p, 25 fps)
  -> YOLO26 detection/tracking (planned yolo26s.pt + ByteTrack)
       -> Person/teacher/screen boxes and track IDs
       -> YOLO26-pose (17 keypoints)
            -> ST-GCN action score
            -> PnP head-pose/gaze proxy
       -> Screen/board crop -> PaddleOCR text changes
  -> VSAM Gaussian temporal alignment
  -> CAS/CTES rule engine
  -> Structured JSON
~~~

| Function | Planned model | File/tool | Intended task |
| --- | --- | --- | --- |
| Detection | YOLO26-S | yolo26s.pt | Fine-tuned categories, not generic-weight guarantees |
| Tracking | ByteTrack | Built-in tracker | Local continuity |
| Pose | YOLO26-N-pose | yolo26n-pose.pt | COCO17 keypoints |
| Action | ST-GCN | Custom training | Temporal skeleton classification |
| Gaze proxy | PnP + fallback | cv2.solvePnP | Head-angle rule mapping |
| OCR | PaddleOCR | Pretrained | Screen-text changes |

These were intended components. Head pose does not establish gaze, attention, or comprehension. The equations below are proposed rule scores, not validated measurements.

## Planned module layout

Paths below are relative to the historical CV project, now vision/Class_Detection/:

~~~text
configs/
  scb_yolo.yaml           # Dataset classes
  pipeline_legacy.yaml   # Historical thresholds/windows/weights
  stgcn.yaml              # ST-GCN training
src/
  __init__.py            # Exports
  detector.py            # Detection/tracking
  pose.py                # Keypoints
  action.py              # ST-GCN/rule interface
  gaze.py                # Head pose and torso fallback
  ocr_anchor.py          # Screen OCR
  vsam.py                # Temporal alignment
  scoring.py             # CAS/CTES
  pipeline.py            # Orchestration
  schema.py              # Data models
models/
  stgcn.py                # Network
  graph.py                # Skeleton topology
scripts/
  train_det.py
  train_pose.py
  train_stgcn.py
  eval_det.py
  infer_video.py
  smoke_test.py
tools/
  download_scb.py
  dataset_audit.py
  visualize.py
tests/
  test_scoring.py
  test_vsam.py
  test_gaze.py
docs/
  implementation_plan.md
~~~

The original project-level requirements/README now have their application entry points in vision/. This diagram records intended responsibilities rather than certifying file completeness.

## Historical phases

### Phase 1: infrastructure

- [ ] Update requirements and create pipeline/data/ST-GCN configuration.
- [ ] Rewrite detector, pose, gaze, action, VSAM, scoring, and schemas.
- [ ] Add orchestration, ST-GCN model, and skeleton graph.
- [ ] Add video inference and update training/evaluation scripts.
- [ ] Add unit checks and documentation.

### Phase 2: data and training

- [ ] Obtain SCB-Dataset5 and audit integrity.
- [ ] Train detection; optionally fine-tune pose.
- [ ] Extract skeleton sequences and train ST-GCN.

Unchecked boxes preserve the archived plan's status and are not a current progress ledger.

### Phase 3: proposed ablations

| ID | Comparison | Hypothesis to test |
| --- | --- | --- |
| A1 | Hard sync vs Gaussian VSAM | Higher correlation with human trends |
| A2 | PnP vs PnP/fallback | Better back-row coverage |
| A3 | Rules vs ST-GCN | Better small-sample classification |
| A4 | Weighted mean vs max | Avoid channel dilution |
| A5 | No penalty vs exponential dispersion penalty | Sensitivity to heterogeneous rule scores |

These are hypotheses, not findings.

## Proposed formulas

Gaussian weight: W(t) = exp(-(t - (t_ocr + μ))² / (2σ²)). The proposed text-change delay μ was 3 s; width σ was 1.5 s.

Aligned concept rule score: Score_Ki = Σ(W(t) × CAS(t)) / ΣW(t).

Historical individual rule index: CAS = max(w1 × S_action, w2 × S_gaze). Max allowed either action or a gaze proxy to contribute; it did not validate attention or concept absorption.

Historical class rule index: CTES = mean(CAS) × exp(-λ × population_std(CAS)). Dispersion was the penalty. The original names referred to absorption/teaching effectiveness, but these equations alone do not measure either construct.

## Proposed verification

From vision/Class_Detection/:

~~~bash
# Mock smoke check:
python scripts/smoke_test.py --mock
pytest tests/ -v
python scripts/infer_video.py --source <video_path> --config configs/pipeline_legacy.yaml --save
python scripts/eval_det.py --weights <best.pt> --data configs/scb_yolo.yaml
~~~

| Layer | Planned metric | Target, not achieved result |
| --- | --- | --- |
| Perception | Detection mAP@50 | ≥0.75 |
| Perception | Tracking IDF1 | ≥0.65 |
| Perception | Action F1 | ≥0.70 |
| Alignment | Spearman correlation with human trend | ≥0.60 |
| System | 1080p per-frame end-to-end latency | <100 ms |

No completion/accuracy claim follows from this archived plan. Current results/limitations are in module and acceptance documents.
