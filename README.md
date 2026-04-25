# 智启教学：融合知识图谱与多模态识别的课堂教学评估辅助系统

> 面向中小学真实课堂的智能教学评估系统，通过多模态 CV 感知与知识图谱联动，量化课堂教学效果。

---

## 0. 运行说明（已精简）

本 README 仅保留系统设计与项目说明，训练/预测命令已统一迁移到独立手册：

- [TRAINING_AND_INFERENCE.md](TRAINING_AND_INFERENCE.md)

建议先阅读运行手册中的“13 类标准流程”，再回到本 README 了解系统设计细节。

## 1. 项目概述

本项目面向真实线下中小学课堂，构建一套 **端到端的 CV 感知-评估 Pipeline**，解决三类核心问题：

运行与实验复现入口：
- 训练与预测手册：[TRAINING_AND_INFERENCE.md](TRAINING_AND_INFERENCE.md)
- 实施方案说明：[Class_Detection/docs/implementation_plan.md](Class_Detection/docs/implementation_plan.md)

1. **后排学生遮挡严重**：分辨率低、互相遮挡导致检测/追踪不稳定
2. **小样本动作识别难收敛**：课堂行为种类多但标注数据有限
3. **教学内容与学生反应的时间延迟**：知识点出现与学生反馈间存在 Time Lag，硬同步产生语义漂移

### 1.1 核心创新点

- **VSAM（Visual-Semantic Alignment Model）**：基于高斯先验的时间衰减窗口实现视觉与语义的零样本软对齐
- **双流感知架构**：YOLO26 检测/追踪 + YOLO26-Pose 姿态提取统一化
- **CAS/CTES 非线性评分体系**：max 融合避免指标稀释 + 方差指数惩罚检出极化课堂
- **PnP + 躯干 Fallback 视线估计**：替代不可靠的表情识别，兼容低分辨率场景

---

## 2. 技术路线总览

### 2.1 系统架构

```
视频输入 (1080p@25fps)
    │
    ▼
┌──────────────────────────────────┐
│  YOLO26 Detection + ByteTrack    │  ← yolo26m.pt + 内置追踪器
│  13 类目标检测 + Track ID 管理   │
└──────────┬───────────────────────┘
           │
    ┌──────┴──────┐
    ▼              ▼
┌─────────┐  ┌──────────────┐
│YOLO26   │  │ Screen/Board │  ← 裁切 screen/blackboard 区域
│Pose     │  │ OCR (Paddle) │  ← 检测知识点文字变化
│(17 kpts)│  └──────┬───────┘
└────┬────┘         │
     │              │
     ├───────┐      │
     ▼       ▼      ▼
┌────────┐ ┌────┐ ┌──────────────┐
│ST-GCN  │ │PnP │ │ VSAM         │
│动作分类 │ │视线 │ │ 高斯时间对齐  │
│S_action│ │解算 │ │ Score_Ki      │
└───┬────┘ │S_g │ └──────┬───────┘
    │      └─┬──┘        │
    ▼        ▼            ▼
┌────────────────────────────────┐
│ CAS / CTES 评分引擎            │
│ → 结构化 JSON → 知识图谱更新   │
└────────────────────────────────┘
```

### 2.2 模型选型

| 功能 | 模型 | 权重文件 | 输出 |
|------|------|---------|------|
| 目标检测 | **YOLO26-M** | `yolo26m.pt` | 13 类目标 BBox + 置信度 |
| 多目标追踪 | **ByteTrack** | ultralytics 内置 | 跨帧 Track ID |
| 姿态估计 | **YOLO26-N-Pose** | `yolo26n-pose.pt` | 17 COCO 关键点 (x, y, conf) |
| 动作分类 | **ST-GCN** | 自训练 | 行为标签 + 置信度 |
| 视线解算 | **PnP + Fallback** | cv2.solvePnP | 头部欧拉角 → 专注度分数 |
| 屏幕 OCR | **PaddleOCR** | 预训练 | 知识点文字锚点 |

### 2.3 YOLO26 特性说明

YOLO26 是 Ultralytics 于 2026 年 1 月发布的最新 YOLO 系列模型，具备以下核心特性：

- **End-to-End NMS-Free**：去除 NMS 后处理，降低推理延迟，硬件适配性更强
- **ProgLoss + STAL**：渐进损失平衡 + 小目标感知标签分配，提升拥挤场景精度
- **MuSGD 优化器**：融合 SGD 与 Muon 技术，收敛更快更稳定
- **多任务统一**：检测/分割/姿态/分类/OBB 共用同一框架

---

## 3. 数据集

### 3.1 主训练数据集：SCB-Dataset5

**SCB-Dataset5**（Student Classroom Behavior Dataset v5）是目前最全面的课堂行为检测公开数据集。

| 属性 | 值 |
|------|-----|
| **图像数量** | 7,428 张 |
| **标注数量** | 106,830 个 |
| **类别数** | 20 |
| **标注格式** | YOLO (`.txt`: `class_id x_center y_center width height`) |
| **图像来源** | 真实课堂监控视频截帧 |
| **下载地址** | https://github.com/Whiffe/SCB-dataset |
| **许可** | 学术研究用途 |

#### 完整类别映射表（已降维版）

为解决 20 类复杂动作中单帧视觉特征歧义问题，目前系统已将特征降维合并为 5 大语义特征：

```yaml
# configs/scb_yolo_merged.yaml
names:
  0: active_student     # 高活跃互动 (举手/回答/讨论/上台/鼓掌等)
  1: focus_student      # 常规专注动作 (书写/阅读/低头/站立/用电脑等)
  2: distracted_student # 注意力分散 (转头/打哈欠/趴桌/用手机等)
  3: teacher            # 执教人员 (教师/指导/板书)
  4: screen_board       # 环境锚点 (黑板/屏幕)
```

#### 行为语义分组

| 类型 | 包含旧版行为 | 课堂含义 |
|------|---------|---------|
| **积极行为** | hand_raising, discuss, talk, answer, stage_interact, clap | 学生高度参与 |
| **正常听讲** | read, write, bow_head, use_computer, stand | 常规专注听课 |
| **游离行为** | turn_head, yawn, lean_desk, use_phone | 注意力分散/犯困 |
| **教师行为** | teacher, guide, board_writing | 教学活动提取 |
| **环境要素** | blackboard, screen | 知识点锚点提取 |

### 3.2 辅助数据参考

| 数据集 | 用途 | 说明 |
|--------|------|------|
| **COCO-Keypoints** | YOLO26-Pose 预训练基础 | 已由 ultralytics 预训练权重覆盖 |
| **ARIC** (UESTC) | 行为分类体系参考 | 32 类活动分类标签体系；仅提供特征向量 |

---

## 4. CV 模块详细设计

### 4.1 检测与追踪模块 (`src/detector.py`)

**职责**：从视频帧中检测所有目标并维护跨帧追踪 ID。

```python
from ultralytics import YOLO

# 初始化 YOLO26 检测器
det_model = YOLO("yolo26m.pt")

# 单帧检测
results = det_model.predict(frame, conf=0.25, device="0")

# 带追踪的视频处理（内置 ByteTrack）
results = det_model.track(source=video_path, tracker="bytetrack.yaml", persist=True)
```

**关键设计**：
- `persist=True`：跨帧保持 Track ID 一致性
- 目标分类后按角色分发：学生 BBox → 姿态估计流；屏幕/黑板 BBox → OCR 流
- 短时丢失轨迹续接：ByteTrack 内置 `track_buffer` 参数控制

### 4.2 姿态估计模块 (`src/pose.py`)

**职责**：对每个学生 BBox 提取 17 个 COCO 关键点。

```python
pose_model = YOLO("yolo26n-pose.pt")
results = pose_model(frame)
# keypoints shape: (N_persons, 17, 3)  → [x, y, confidence]
```

**17 个 COCO 关键点含义**：

```
0: 鼻子        5: 左肩       10: 右手腕     15: 左踝
1: 左眼        6: 右肩       11: 左髋       16: 右踝
2: 右眼        7: 左肘       12: 右髋
3: 左耳        8: 右肘       13: 左膝
4: 右耳        9: 左手腕     14: 右膝
```

**下游用途**：
- 关键点 0-4（面部）→ PnP 视线解算
- 关键点 5-6（肩部）+ 0（鼻）→ 躯干向量 Fallback
- 完整 17 点时序（T=30 帧）→ ST-GCN 动作分类

### 4.3 动作分类模块 (`src/action.py`)

**职责**：基于关键点时序判断学生当前行为类别。

**方案一：ST-GCN（主路径）**

ST-GCN（时空图卷积网络）将人体骨骼建模为图结构：
- **节点**：17 个关键点
- **空间边**：骨骼连接（如 肩→肘→腕）
- **时间边**：同一关键点在相邻帧间的连接

```
输入: (N, C, T, V, M)
N = batch_size
C = 3 (x, y, confidence)
T = 30 (时间窗口帧数)
V = 17 (关键点数)
M = 1 (单人)
```

**动作标签映射**（从 SCB-Dataset5 行为中提取关键动作）：

| 动作标签 | 描述 | S_action 映射 |
|---------|------|:----------:|
| writing | 书写/记笔记 | 0.85 |
| reading | 阅读 | 0.80 |
| hand_raising | 举手 | 0.95 |
| discussing | 讨论 | 0.70 |
| leaning | 趴桌 | 0.15 |
| using_phone | 使用手机 | 0.10 |
| yawning | 打哈欠 | 0.20 |
| looking_around | 四处张望 | 0.25 |
| attending | 正常听讲 | 0.70 |

**方案二：规则降级（ST-GCN 训练数据不足时启用）**

```python
def rule_based_action(keypoints_seq):
    """基于关键点运动统计特征的规则分类"""
    head_motion = calc_head_motion_range(keypoints_seq)
    hand_height = calc_hand_relative_height(keypoints_seq)
    body_lean   = calc_body_lean_angle(keypoints_seq)

    if hand_height > THRESHOLD_RAISE:
        return "hand_raising", 0.90
    if body_lean > THRESHOLD_LEAN:
        return "leaning", 0.80
    ...
```

### 4.4 视线估计模块 (`src/gaze.py`)

**职责**：计算学生视线方向和专注度分数 $S_{gaze}$。

#### PnP 主路径

利用面部关键点（鼻、眼、耳）映射至标准 3D 人脸模型，通过 `cv2.solvePnP` 解算欧拉角：

```python
# 3D 人脸模型（标准化坐标）
FACE_3D = np.array([
    [0.0,   0.0,    0.0],      # 鼻尖
    [-30.0,  40.0, -30.0],     # 左眼
    [ 30.0,  40.0, -30.0],     # 右眼
    [-60.0, -10.0, -50.0],     # 左耳
    [ 60.0, -10.0, -50.0],     # 右耳
])

# 从 YOLO26-Pose 关键点 [0,1,2,3,4] 取面部 2D 坐标
face_2d = keypoints[[0,1,2,3,4], :2].astype(np.float64)

# PnP 解算
success, rvec, tvec = cv2.solvePnP(FACE_3D, face_2d, camera_matrix, dist_coeffs)
# rvec → rotation matrix → Euler angles (pitch, yaw, roll)
```

**视线方向分类**：

| 视线方向 | Pitch 范围 | Yaw 范围 | $S_{gaze}$ |
|---------|-----------|---------|:---------:|
| 聚焦黑板/屏幕 | -15° ~ 15° | -20° ~ 20° | 0.85 ~ 1.0 |
| 低头专注（课桌） | -50° ~ -15° | -15° ~ 15° | 0.60 ~ 0.80 |
| 四处张望 | 其他 | \|yaw\| > 35° | 0.10 ~ 0.30 |

#### 躯干向量 Fallback

当面部关键点置信度 < 阈值时，使用肩部-鼻部向量估算朝向：

```python
def torso_fallback(kpts):
    """用肩-颈方向估算视线大致朝向"""
    mid_shoulder = (kpts[5] + kpts[6]) / 2  # 双肩中点
    nose = kpts[0]                          # 鼻尖
    direction = nose[:2] - mid_shoulder[:2]
    # 归一化后映射为专注度
    ...
```

### 4.5 VSAM 时间对齐模块 (`src/vsam.py`)

**职责**：将知识点出现时刻与学生反应进行软对齐。

**核心问题**：教师展示知识点 $K_i$ 后，学生群体反应存在 $\mu$ 秒的延迟（教育心理学规律）。硬同步（直接取同时刻状态）会导致语义漂移。

**解决方案**：高斯先验时间衰减窗口

```python
def gaussian_weight(t, t_ocr, mu=3.0, sigma=1.5):
    """
    t     : 当前时刻
    t_ocr : OCR 检测到知识点出现的时刻
    mu    : 先验延迟（学生反应峰值滞后秒数）
    sigma : 窗口宽度
    """
    center = t_ocr + mu
    return math.exp(-((t - center) ** 2) / (2 * sigma ** 2))

def score_knowledge_point(cas_values, timestamps, t_ocr, mu=3.0, sigma=1.5):
    """知识点吸收度 = 高斯加权 CAS 平均"""
    weights = [gaussian_weight(t, t_ocr, mu, sigma) for t in timestamps]
    return sum(w * s for w, s in zip(weights, cas_values)) / sum(weights)
```

**OCR 锚点触发逻辑**：
1. 每 N 帧对 screen/blackboard BBox 区域执行 PaddleOCR
2. 计算文本哈希差异（Levenshtein distance / hash 变化）
3. 超过阈值 → 判定为"知识点切换"→ 记录 $t_{ocr}$

### 4.6 评分引擎 (`src/scoring.py`)

#### 个体积极度 CAS

$$CAS = \max(w_1 \cdot S_{action},\ w_2 \cdot S_{gaze}) \cdot P_{negative}$$

**设计理由**：采用 `max` 而非加权平均：
- 学生积极记笔记（$S_{action}$ 高）但低头不看黑板（$S_{gaze}$ 低）→ 仍视为专注
- 学生静坐盯黑板（$S_{gaze}$ 高）但无明显动作（$S_{action}$ 低）→ 仍视为专注
- 加权平均会导致两个合理信号互相稀释
- $P_{negative}$：负面行为惩罚因子（使用手机/打哈欠等行为 × 0.5）

#### 课堂教学效果指数 CTES

$$CTES = \mu_{CAS} \cdot \exp(-\lambda \cdot \sigma_{CAS})$$

- $\mu_{CAS}$：全班 CAS 均值
- $\sigma_{CAS}$：全班 CAS 标准差
- $\lambda$：惩罚系数（默认 1.0）

**设计理由**：不仅看平均分，更将全班方差作为指数级惩罚。有效区分：
- 均值 0.7 + 方差 0.05 → CTES ≈ 0.55（理想状态：大多数学生专注）
- 均值 0.7 + 方差 0.30 → CTES ≈ 0.28（危险状态：两极分化严重）

---

## 5. 训练与预测（说明已迁移）

为避免 README 过长，训练、评估、推理的全部命令已统一整理至：

- [TRAINING_AND_INFERENCE.md](TRAINING_AND_INFERENCE.md)

该手册包含：
1. 环境与依赖安装。
2. 13 类数据构建与清洗。
3. 检测/姿态/ST-GCN 训练命令。
4. 评估、视频推理与冒烟测试命令。
5. 常见报错与排查建议。

---

## 6. 结构化输出协议

CV 模块通过 JSON Schema 向下游（知识图谱/大模型）传递评估数据：

```json
{
  "timestamp": "2026-05-10T10:15:30Z",
  "frame_id": 750,
  "knowledge_anchor": {
    "entity": "分数加减法",
    "trigger_time": "10:15:25Z",
    "gaussian_weight": 0.95,
    "score_k": 0.82
  },
  "classroom_metrics": {
    "CTES_score": 0.81,
    "mean_CAS": 0.78,
    "std_CAS": 0.15,
    "active_tracks": 35,
    "behavior_distribution": {
      "attending": 20,
      "writing": 8,
      "discussing": 3,
      "bow_head": 2,
      "lean_desk": 1,
      "use_phone": 1
    }
  },
  "student_states": [
    {
      "track_id": 7,
      "bbox": [100.0, 150.0, 200.0, 350.0],
      "action": {
        "label": "writing",
        "confidence": 0.89,
        "source": "stgcn"
      },
      "gaze": {
        "pitch": -22.5,
        "yaw": 5.0,
        "focus_score": 0.90,
        "source": "pnp"
      },
      "CAS": 0.89
    }
  ]
}
```

---

## 7. 项目结构

```text
Class-Knowledge-Graph/
├── README.md
├── TRAINING_AND_INFERENCE.md   # 训练/评估/推理命令统一手册
├── requirements.txt
├── test_GPU.py
├── yolo26s.pt
├── SCB_yolo_dataset/
├── SCB_yolo_dataset_13cls/
├── SCB_yolo_dataset_merged/
└── Class_Detection/
  ├── configs/
  ├── docs/
  │   └── implementation_plan.md
  ├── models/
  ├── scripts/
  ├── src/
  ├── tests/
  └── tools/
```

---

## 8. 评测指标体系

| 层级 | 指标 | 目标值 | 说明 |
|------|------|:------:|------|
| 感知层 | 检测 mAP@50 | ≥ 0.75 | YOLO26 在 SCB-Dataset5 上的检测精度 |
| 感知层 | 追踪 IDF1 | ≥ 0.65 | ByteTrack 跨帧 ID 一致性 |
| 感知层 | 动作分类 F1 | ≥ 0.70 | ST-GCN 行为分类加权 F1 |
| 感知层 | 视线分类准确率 | ≥ 0.75 | PnP 视线方向三分类 |
| 对齐层 | Score_Ki Spearman 相关性 | ≥ 0.60 | VSAM 输出与人工观察趋势 |
| 系统层 | 端到端延迟 (1080p) | < 100ms | 单帧完整 Pipeline 处理时间 |
| 系统层 | 吞吐 FPS | ≥ 15 | 满足实时监控需求 |

---

## 9. 风险与预案

| 风险 | 预案 |
|------|------|
| 后排关键点缺失 | 启用躯干向量 Fallback，低置信帧降权 |
| OCR 错触发 | 翻页检测 + 文本变化阈值双门控 |
| 追踪 ID 抖动 | ByteTrack `track_buffer` + 短时续接策略 |
| ST-GCN 训练数据不足 | 降级为规则分类方法 |
| 现场演示异常 | 离线视频回放模式 + 容错脚本 |
| GPU 显存不足 | 降低 batch size / 切换 Nano 模型 |

---

## 10. 快速开始

按以下顺序执行即可：

1. 阅读并执行运行手册：[TRAINING_AND_INFERENCE.md](TRAINING_AND_INFERENCE.md)
2. 优先使用“13 类标准流程”完成首轮训练与验证。
3. 训练后使用 best.pt 进行评估与视频推理。

---

## 11. 环境依赖

完整依赖见 `requirements.txt`。核心依赖：

| 包 | 版本 | 用途 |
|----|------|------|
| ultralytics | ≥ 8.3.50 | YOLO26 检测/姿态/追踪 |
| torch | ≥ 2.5.0 (CUDA) | 深度学习后端 |
| opencv-python | ≥ 4.10.0 | PnP 解算 & 图像处理 |
| paddleocr | ≥ 2.8.1 | 屏幕文字识别 |
| scipy | ≥ 1.12.0 | 科学计算 |
| pydantic | ≥ 2.8.0 | 数据模型校验 |

---

## 12. 参考文献

- YOLO26: Ultralytics (2026). *YOLO26: Edge-First, NMS-Free Object Detection*. [docs.ultralytics.com](https://docs.ultralytics.com)
- ST-GCN: Yan et al. (2018). *Spatial Temporal Graph Convolutional Networks for Skeleton-Based Action Recognition*. AAAI 2018.
- ByteTrack: Zhang et al. (2022). *ByteTrack: Multi-Object Tracking by Associating Every Detection Box*. ECCV 2022.
- RTMPose: Jiang et al. (2023). *RTMPose: Real-Time Multi-Person Pose Estimation based on MMPose*. arXiv:2303.07399.
- SCB-Dataset: Whiffe et al. *Student Classroom Behavior Dataset*. [github.com/Whiffe/SCB-dataset](https://github.com/Whiffe/SCB-dataset)
- ARIC: Xu et al. (2024). *ARIC: An Activity Recognition Dataset in Classroom Surveillance Images*. arXiv:2410.12337.

---

## 13. 许可证

本项目用于学术研究和计算机设计大赛参赛使用。
