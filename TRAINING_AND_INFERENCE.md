# 训练与预测运行手册

本文档集中维护本项目全部训练、评估、推理命令。
建议按“13 类标准流程”先跑通一轮，再进行对比实验。

---

## 1. 环境准备

```bash
cd /mnt/Data4/24zhs/Class-Knowledge-Graph

# 可按需修改环境名
conda create -n classkg python=3.12 -y
conda activate classkg

pip install -U pip
pip install -r requirements.txt

# 若提示缺少 ultralytics
pip install -U ultralytics
```

---

## 2. 数据准备与校验

### 2.1 构建 13 类数据集（推荐）

```bash
cd /mnt/Data4/24zhs/Class-Knowledge-Graph/Class_Detection
python tools/build_scb_13cls.py
```

说明：
- 输出目录为 `../SCB_yolo_dataset_13cls`。
- 脚本会将类别重映射为连续 ID `0..12`。
- 脚本会自动清洗越界坐标到 `[0, 1]`，并丢弃无效框。

### 2.2 数据完整性审计

```bash
cd /mnt/Data4/24zhs/Class-Knowledge-Graph/Class_Detection

python tools/dataset_audit.py \
  --dataset-root ../SCB_yolo_dataset_13cls \
  --output artifacts/reports/scb13_audit.json
```

### 2.3 快速检查类别分布与坐标越界

```bash
cd /mnt/Data4/24zhs/Class-Knowledge-Graph

# 类别分布（13 类应为 0..12）
awk '{c[$1]++} END {for (k in c) print k, c[k]}' SCB_yolo_dataset_13cls/labels/train/*.txt | sort -n
awk '{c[$1]++} END {for (k in c) print k, c[k]}' SCB_yolo_dataset_13cls/labels/val/*.txt | sort -n

# 越界检查（正常应无输出）
find SCB_yolo_dataset_13cls -name "*.txt" -exec \
awk '$2 < 0 || $2 > 1 || $3 < 0 || $3 > 1 || $4 < 0 || $4 > 1 || $5 < 0 || $5 > 1 {print FILENAME, $0}' {} +
```

---

## 3. 检测模型训练

以下命令均在 `Class_Detection` 目录执行。

### 3.1 13 类标准训练（推荐）

```bash
cd /mnt/Data4/24zhs/Class-Knowledge-Graph/Class_Detection

/home/24xyx/miniconda3/envs/Class_Detection/bin/python scripts/train_det.py \
  --data configs/scb_yolo_13cls.yaml \
  --model yolo26m.pt \
  --epochs 40 \
  --imgsz 960 \
  --batch 16 \

  --device 0 \
  --workers 8 \
  --patience 20 \
  --name scb13_yolo26m_e40 \
  --cache
```

### 3.2 5 类合并训练（对照）

```bash
cd /mnt/Data4/24zhs/Class-Knowledge-Graph/Class_Detection

python scripts/train_det.py \
  --data configs/scb_yolo_merged.yaml \
  --model yolo26m.pt \
  --epochs 60 \
  --imgsz 960 \
  --batch 16 \
  --device 0 \
  --workers 8 \
  --patience 20 \
  --name scb_yolo26m_merged \
  --cache
```

### 3.3 原始 SCB 配置训练（仅对照）

```bash
cd /mnt/Data4/24zhs/Class-Knowledge-Graph/Class_Detection

python scripts/train_det.py \
  --data configs/scb_yolo.yaml \
  --model yolo26s.pt \
  --epochs 20 \
  --imgsz 960 \
  --batch 16 \
  --device 0 \
  --workers 8 \
  --name scb_yolo26s \
  --cache
```

---

## 4. 可选训练

### 4.1 YOLO Pose 微调（可选）

```bash
cd /mnt/Data4/24zhs/Class-Knowledge-Graph/Class_Detection

python scripts/train_pose.py \
  --data <pose_dataset_yaml> \
  --model yolo26n-pose.pt \
  --epochs 80 \
  --imgsz 640 \
  --batch 32 \
  --device 0 \
  --name classroom_pose
```

### 4.2 ST-GCN 动作分类训练（可选）

要求 `--keypoints-dir` 下已有 `train/`、`val/` 与 `label.json`。

```bash
cd /mnt/Data4/24zhs/Class-Knowledge-Graph/Class_Detection

python scripts/train_stgcn.py \
  --config configs/stgcn.yaml \
  --keypoints-dir <keypoints_dataset_dir> \
  --epochs 100 \
  --device 0
```

---

## 5. 评估命令

### 5.1 检测评估

```bash
cd /mnt/Data4/24zhs/Class-Knowledge-Graph/Class_Detection

python scripts/eval_det.py \
  --weights artifacts/runs/detect/scb13_yolo26m_e40/weights/best.pt \
  --data configs/scb_yolo_13cls.yaml \
  --imgsz 960 \
  --batch 16 \
  --device 0
```

提示：评估和部署优先使用 `best.pt`，不要默认使用 `last.pt`。

---

## 6. 预测与推理命令

### 6.1 视频推理（端到端）

#### 使用训练好的 best.pt 推理（13 类全功能版，推荐）

```bash
cd /mnt/Data4/24zhs/Class-Knowledge-Graph/Class_Detection

python scripts/infer_video.py \
  --source ../classroom.mp4 \
  --config configs/pipeline.yaml \
  --det-weights artifacts/runs/detect/scb13_yolo26m_e40/weights/best.pt \
  --pose-weights yolo26n-pose.pt \
  --device 0 \
  --interval-sec 1.0 \
  --save \
  --output artifacts/results/latest
```

> **说明**：此模式下，模型不仅能精确识别学生动作，还能自动检测视频里的黑板/屏幕区域，并**自动触发 OCR 文字识别**。

#### 使用预训练 yolo26m.pt 直接推理（降级体验版，无需训练）

官方预训练权重（COCO 80类）无法自动识别“黑板”或“屏幕”，因此**默认不会触发 OCR**。为了在免训练的情况下体验完整的知识点提取流程，你必须**手动指定 PPT/屏幕 的区域坐标** (`--ppt-crop`)，强制开启区域 OCR。

**Linux / macOS (Bash)**
```bash
cd /mnt/Data4/24zhs/Class-Knowledge-Graph/Class_Detection

python scripts/infer_video.py \
  --source ../classroom.mp4 \
  --config configs/pipeline.yaml \
  --device 0 \
  --interval-sec 1.0 \
  --ppt-crop "100,50,800,600" \
  --save \
  --output artifacts/results/latest
```

**Windows (PowerShell)**
```powershell
# 确保在 Class_Detection 目录下运行
python scripts/infer_video.py `
  --source ../classroom.mp4 `
  --config configs/pipeline.yaml `
  --device 0 `
  --interval-sec 1.0 `
  --ppt-crop "100,50,800,600" `
  --save `
  --output artifacts/results/latest
```

> **提示**：不指定 `--det-weights` 时，默认下载使用官方 `yolo26m.pt`。请记得将 `"100,50,800,600"` (x1, y1, x2, y2) 替换为你真实视频里的 PPT 坐标。

### 6.2 冒烟测试（快速验证链路）

```bash
cd /mnt/Data4/24zhs/Class-Knowledge-Graph/Class_Detection
python scripts/smoke_test.py --mock
```

### 6.3 独立测试 OCR (PPT文字提取)

你可以独立测试文字提取能力，不加载 YOLO/骨骼/动作识别等大模型，加快调试速度。

```bash
cd /mnt/Data4/24zhs/Class-Knowledge-Graph/Class_Detection

# 1. 默认测试 (使用目录自带的 test.mp4)
python scripts/test_ocr_standalone.py

# 2. 指定其他视频
python scripts/test_ocr_standalone.py --source "/path/to/your/video.mp4"

# 3. 指定视频并限制 PPT 的坐标范围 (x1,y1,x2,y2)，能大幅提升识别率并减少杂乱输出
python scripts/test_ocr_standalone.py --source "test.mp4" --bbox "100,50,800,600"
```
> 提示：测试过程中，每一帧被识别的裁剪画面都会被保存在 `Class_Detection/artifacts/` 目录下。你可以查看这些图片，验证 `--bbox` 坐标是否刚好能框住 PPT 内容。

---

## 7. 训练产物与结果目录

检测训练默认输出：

```text
Class_Detection/artifacts/runs/detect/<run_name>/
├── weights/best.pt
├── weights/last.pt
├── args.yaml
├── results.csv
├── results.png
├── confusion_matrix.png
└── BoxPR_curve.png 等
```

推理默认输出：

```text
Class_Detection/artifacts/results/
├── <source>_<timestamp>.mp4
└── <source>_<timestamp>.json
```

---

## 8. 常见问题

1. 报错 `No module named ultralytics`：执行 `pip install -U ultralytics`。
2. 显存不足：优先降低 `--batch`（16->8->4），再降低 `--imgsz`（960->768->640）。
3. CUDA `device-side assert triggered`：
   - 先检查标签类别是否越界；
   - 再检查坐标是否越界；
   - 触发后请重启训练进程，避免使用已污染的 CUDA 上下文。
4. 目录路径错误：建议始终先 `cd /mnt/Data4/24zhs/Class-Knowledge-Graph/Class_Detection` 后执行脚本。

---

## 9. 13 类映射关系（原始 ID -> 新 ID）

```text
0->0(hand_raising), 1->1(read), 2->2(write), 5->3(discuss),
6->4(talk), 7->5(answer), 8->6(stage_interact), 13->7(stand),
15->8(teacher), 16->9(guide), 17->10(board_writing),
18->11(blackboard), 19->12(screen)
```
