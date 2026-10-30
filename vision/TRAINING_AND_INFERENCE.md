# Historical training and inference manual

> Recorded workflow, 2026-10-25. For the pretrained three-class path use [Pretrained usage](PRETRAINED_USAGE.md); for the current workbench use the [module README](README.md). `configs/pipeline.yaml` switched to pretrained mode. Historical experiments below explicitly use `configs/pipeline_legacy.yaml` and need separate data, environment, and weights. Old classes/formulas are not measured capabilities of the current path.

These commands retain detection, pose, ST-GCN, and OCR experiments. Earlier wording referred to a “13-class standard”; actual examples below use three-/seven-class mappings. Inspect dataset configuration and weight metadata before running.

## Environment

From `vision/`:

```bash
conda create -n classkg python=3.12 -y
conda activate classkg
pip install -U pip
pip install -r requirements.txt
# If ultralytics is missing:
pip install -U ultralytics
cd Class_Detection
```

Run subsequent commands from `vision/Class_Detection/` unless stated otherwise. This environment is separate from application setup.

## Data preparation and audit

```bash
python tools/build_scb5_unified.py
python tools/dataset_audit.py \
  --dataset-root ../SCB5_yolo_unified \
  --output artifacts/reports/scb5_audit.json
```

The builder reads nine subsets under `../SCB-Dataset/SCB-Dataset`, writes `../SCB5_yolo_unified`, remaps local IDs to global `0..2`, deduplicates images while merging annotations, clips coordinates to `[0,1]`, and discards invalid boxes. The dataset audit reports class distributions and coordinate errors; inspect these before training.

## Detection training

The historical seven-class SCBehavior run:

```bash
python scripts/train_det.py \
  --data configs/scb_dataset_yolo.yaml \
  --model yolo26m.pt \
  --epochs 60 --imgsz 960 --batch 32 --device 3 \
  --workers 8 --patience 10 --name scbehavior_yolo26m --cache
```

Seven categories: writing, reading, looking up, turning the head, raising a hand, standing, discussion. This historical detector is separate from the newer per-person crop classifier.

The historical SCB-5 screen detector (`scb5_yolo26m_e402` in original notes) supplied `screen_board` as an environment model. Weights are not shipped. Configure an actual checkpoint in the legacy pipeline instead of assuming it exists.

## Optional pose/ST-GCN training

```bash
python scripts/train_pose.py \
  --data <pose_dataset_yaml> --model yolo26n-pose.pt \
  --epochs 80 --imgsz 640 --batch 32 --device 0 --name classroom_pose

# keypoints-dir needs train/, val/, and label.json.
python scripts/train_stgcn.py \
  --config configs/stgcn.yaml --keypoints-dir <keypoints_dataset_dir> \
  --epochs 100 --device 0
```

## Detection evaluation

```bash
python scripts/eval_det.py \
  --weights yolo26m.pt --data configs/scb_yolo.yaml \
  --imgsz 960 --batch 32 --device 0

python scripts/eval_det.py \
  --weights artifacts/runs/detect/scb5_yolo26m_e40/weights/best.pt \
  --data configs/scb_yolo.yaml --imgsz 960 --batch 16 --device 0
```

Prefer selected `best.pt` over assuming `last.pt` is best. Verify weight/dataset class mappings agree.

## Historical video inference

With a trained behavior checkpoint and separately configured environment model:

```bash
python scripts/infer_video.py \
  --source ../classroom.mp4 --config configs/pipeline_legacy.yaml \
  --det-weights artifacts/runs/detect/scbehavior_yolo26m/weights/best.pt \
  --pose-weights yolo26n-pose.pt --device 0 --interval-sec 1.0 --save \
  --output artifacts/results/latest
```

`--det-weights` supplies behavior; environment weights are configured separately. The historical tracker was `configs/bytetrack_low.yaml`. If sampling gaps break tracking, `--interval-sec 0` tries every frame. Tracks are video-local, not identities.

Generic COCO `yolo26m.pt` has no classroom screen/blackboard class, so it cannot automatically trigger screen OCR. Supply the actual pixel crop to test legacy OCR without a trained environment detector:

```bash
python scripts/infer_video.py \
  --source ../classroom.mp4 --config configs/pipeline_legacy.yaml \
  --device 0 --interval-sec 1.0 --ppt-crop "100,50,800,600" \
  --save --output artifacts/results/latest
```

Equivalent PowerShell:

```powershell
python scripts/infer_video.py `
  --source ../classroom.mp4 --config configs/pipeline_legacy.yaml `
  --device 0 --interval-sec 1.0 --ppt-crop "100,50,800,600" `
  --save --output artifacts/results/latest
```

Replace `x1,y1,x2,y2` with the real region. Without `--det-weights`, the historical route may download `yolo26m.pt`; generic person detection does not provide seven-class behavior recognition.

## Smoke and standalone OCR

```bash
# Mock pipeline check, not model accuracy:
python scripts/smoke_test.py --mock

# Source distributions exclude demo video; supply a real local file.
python scripts/test_ocr_standalone.py --source /path/to/video.mp4
python scripts/test_ocr_standalone.py --source /path/to/video.mp4 --bbox "100,50,800,600"
```

Standalone OCR avoids detection/pose/action models. Crops are saved under `Class_Detection/artifacts/`; inspect coordinates. A smaller crop can reduce irrelevant output but does not guarantee accuracy.

## Outputs

```text
Class_Detection/artifacts/runs/detect/<run_name>/
  weights/best.pt
  weights/last.pt
  args.yaml
  results.csv
  results.png
  confusion_matrix.png
  BoxPR_curve.png

Class_Detection/artifacts/results/
  <source>_<timestamp>.mp4
  <source>_<timestamp>.json
```

## Troubleshooting

1. Missing `ultralytics`: install in the selected experiment environment.
2. VRAM exhaustion: reduce batch (16 → 8 → 4), then image size (960 → 768 → 640).
3. CUDA `device-side assert triggered`: check class IDs/coordinate bounds, then restart the process to clear the failed CUDA context.
4. Path errors: run examples from `vision/Class_Detection/`.

## Historical mappings

```text
Behavior model
0 write
1 read
2 lookup
3 turn_head
4 raise_hand
5 stand
6 discuss

Environment model
0 student       (unused by the historical environment path)
1 teacher
2 screen_board  (OCR input)
```

The old pipeline assigned behavior boxes to students and combined action/gaze into CAS. These are legacy assumptions, not verified role, gaze, listening, distraction, or psychological-engagement measurements. Turning the head alone does not establish inattention. Consult current boundaries before interpreting outputs.
