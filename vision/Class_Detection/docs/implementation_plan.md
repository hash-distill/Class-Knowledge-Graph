# 智启教学 CV Pipeline 重构实施计划

> **版本**：v2.0
> **更新日期**：2026-04-15
> **优先目标**：跑通完整 CV 技术栈（前端和知识图谱后续再做）

---

## 一、项目审视总结

### 1.1 原方案存在的关键问题

| 编号 | 问题 | 严重度 | 说明 |
|------|------|--------|------|
| P1 | ARIC 数据集无法用于检测任务 | 🔴 致命 | ARIC 不提供原始图像，仅提供预提取特征向量，无法训练 YOLO |
| P2 | 核心模块全部是 Placeholder | 🔴 致命 | RTMPose/ST-GCN/PnP/OCR 全部只有 stub 函数 |
| P3 | 使用 YOLO11n 而非声明的 YOLO26 | 🟡 严重 | 技术版本不一致会在答辩中被扣分 |
| P4 | 子包结构过深 | 🟡 一般 | 7 个子包仅含 1-2 个文件，增加维护成本 |

### 1.2 核心决策变更

| 维度 | 原方案 | 新方案 | 原因 |
|------|--------|--------|------|
| 检测模型 | YOLO11n | **YOLO26** | 用户要求；NMS-free，Edge-first，2026 最前沿 |
| 主训练数据集 | ARIC（不可用） | **SCB-Dataset5** | 7428 张图，106830 标注，20 个课堂行为类，YOLO 原生格式 |
| 姿态估计 | RTMPose (mmpose) | **YOLO26-Pose** + rtmlib 备选 | YOLO26 原生支持 pose，统一技术栈；rtmlib 作轻量备选 |
| 追踪器 | 独立集成 ByteTrack | **ultralytics 内置 tracker** | ultralytics 已内置 BoT-SORT/ByteTrack |
| 依赖管理 | mmpose+mmcv+mmdet 重依赖 | **精简为 ultralytics + rtmlib** | 减少依赖冲突风险 |

---

## 二、数据集方案

### 2.1 主数据集：SCB-Dataset5

- **来源**：https://github.com/Whiffe/SCB-dataset
- **规模**：7,428 张图像，106,830 个标注
- **格式**：原生 YOLO 标注格式（`.txt`，`class_id x_center y_center width height`）
- **20 个类别**（完整覆盖中小学课堂场景）：

| 分组 | 类别 |
|------|------|
| 学生个体行为 | hand_raising, read, write, bow_head, turn_head, answer, yawn, stand, lean_desk, use_phone |
| 学生群体行为 | discuss, clap, use_computer, talk |
| 教师行为 | teacher, guide, board_writing, stage_interact |
| 环境要素 | blackboard, screen |

### 2.2 辅助数据源

| 数据集 | 用途 | 说明 |
|--------|------|------|
| COCO-Keypoints | YOLO26-Pose 的预训练基础 | 已由 ultralytics 预训练，直接使用权重 |
| ARIC | 行为分类体系参考 | 32 类分类标签体系用于论文对比引用 |
| 自采中小学数据（如有） | 领域微调 + 演示素材 | 提供最强竞争力，建议后续补充 |

---

## 三、重构后技术栈

### 3.1 Pipeline 架构

```
视频输入 (1080p@25fps)
    │
    ▼
┌──────────────────────────────────┐
│  YOLO26 Detection + Tracking     │  ← yolo26s.pt + ByteTrack (内置)
│  人体/教师/屏幕 BBox + Track ID  │
└──────────┬───────────────────────┘
           │
    ┌──────┴──────┐
    ▼              ▼
┌─────────┐  ┌──────────────┐
│YOLO26   │  │ Screen/Board │  ← 从检测结果中裁切 screen/blackboard 区域
│Pose     │  │ OCR (Paddle) │  ← 提取知识点文字变化
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
    │        │            │
    ▼        ▼            ▼
┌────────────────────────────────┐
│ CAS / CTES 评分引擎            │
│ 结构化 JSON 输出               │
└────────────────────────────────┘
```

### 3.2 模型选型

| 功能 | 模型 | 权重文件 | 说明 |
|------|------|---------|------|
| 目标检测 | YOLO26-S | `yolo26s.pt` | 检测人体/教师/屏幕等 20 类目标 |
| 多目标追踪 | ByteTrack | ultralytics 内置 | 跨帧 Track ID 维护 |
| 姿态估计 | YOLO26-N-Pose | `yolo26n-pose.pt` | 17 关键点（COCO 格式） |
| 动作分类 | ST-GCN | 自训练 | 基于关键点时序的图卷积网络 |
| 视线解算 | PnP + Fallback | cv2.solvePnP | 头部欧拉角 → 专注度映射 |
| 屏幕OCR | PaddleOCR | 预训练 | 知识点文字变化检测 |

---

## 四、重构后项目结构

```text
Class Knowledge Graph/
├── configs/
│   ├── scb_yolo.yaml           # SCB-Dataset5 YOLO 配置（20 类）
│   ├── pipeline.yaml           # 全局 pipeline 配置（阈值/窗口/权重）
│   └── stgcn.yaml              # ST-GCN 训练配置
├── src/
│   ├── __init__.py             # 包初始化 & 公共导出
│   ├── detector.py             # YOLO26 检测 + 内置追踪
│   ├── pose.py                 # YOLO26-Pose 关键点提取
│   ├── action.py               # ST-GCN 动作分类
│   ├── gaze.py                 # PnP 头部姿态 + 躯干 Fallback
│   ├── ocr_anchor.py           # PaddleOCR 屏幕文字检测
│   ├── vsam.py                 # VSAM 高斯时间对齐
│   ├── scoring.py              # CAS/CTES 评分引擎
│   ├── pipeline.py             # ★ 端到端 Pipeline 编排器
│   └── schema.py               # Pydantic 数据模型定义
├── models/                     # 自定义模型定义
│   ├── __init__.py
│   ├── stgcn.py                # ST-GCN 网络结构
│   └── graph.py                # 骨骼图拓扑定义
├── scripts/
│   ├── train_det.py            # YOLO26 检测训练
│   ├── train_pose.py           # YOLO26-Pose 微调
│   ├── train_stgcn.py          # ST-GCN 训练
│   ├── eval_det.py             # 检测评估
│   ├── infer_video.py          # ★ 视频推理主入口
│   └── smoke_test.py           # 冒烟测试（全链路验证）
├── tools/
│   ├── download_scb.py         # SCB 数据集下载脚本
│   ├── dataset_audit.py        # 数据集完整性审计
│   └── visualize.py            # 检测/姿态/热力图可视化
├── tests/                      # 单元测试
│   ├── test_scoring.py
│   ├── test_vsam.py
│   └── test_gaze.py
├── docs/                       # 文档
│   └── implementation_plan.md  # 本文件
├── requirements.txt            # Python 依赖
└── README.md                   # 项目说明
```

---

## 五、分阶段实施

### Phase 1：基础设施重建（当前阶段）

- [ ] 更新 `requirements.txt`（YOLO26 + 精简依赖）
- [ ] 创建 `configs/pipeline.yaml`（全局配置）
- [ ] 更新 `configs/scb_yolo.yaml`（SCB-Dataset5 20 类）
- [ ] 重写 `src/detector.py`（YOLO26 + 内置追踪）
- [ ] 重写 `src/pose.py`（YOLO26-Pose 关键点）
- [ ] 重写 `src/gaze.py`（PnP 实装 + Fallback）
- [ ] 重写 `src/action.py`（ST-GCN 接口 + 规则降级）
- [ ] 重写 `src/vsam.py`（增强版 VSAM）
- [ ] 重写 `src/scoring.py`（CAS/CTES 增强）
- [ ] 重写 `src/schema.py`（Pydantic 数据模型）
- [ ] 创建 `src/pipeline.py`（端到端编排）
- [ ] 创建 `models/stgcn.py` + `models/graph.py`
- [ ] 创建 `scripts/infer_video.py`
- [ ] 更新训练/评估脚本
- [ ] 创建基础单元测试
- [ ] 更新 `README.md`

### Phase 2：数据集接入与训练

- [ ] 下载 SCB-Dataset5
- [ ] 运行 `dataset_audit` 验证数据完整性
- [ ] YOLO26 检测训练（SCB-Dataset5）
- [ ] YOLO26-Pose 微调（如需要）
- [ ] ST-GCN 训练（基于提取的关键点序列）

### Phase 3：消融实验与优化

| 实验编号 | 变量 | 对比内容 | 预期结论 |
|---------|------|---------|---------|
| A1 | 时间对齐 | 硬同步 vs VSAM 高斯软对齐 | VSAM 提升 Score_Ki 相关性 |
| A2 | 视线估计 | 仅 PnP vs PnP + Fallback | Fallback 提升后排召回率 |
| A3 | 动作识别 | 简单规则 vs ST-GCN | ST-GCN 在小样本下优势 |
| A4 | CAS 融合 | 加权平均 vs max 非线性 | max 避免互相稀释 |
| A5 | CTES 惩罚 | 无惩罚 vs 方差指数惩罚 | 检出极化课堂能力 |

---

## 六、关键公式体系

### 6.1 高斯对齐权重

$$W(t) = \exp\left(-\frac{(t - (t_{ocr}+\mu))^2}{2\sigma^2}\right)$$

- $t_{ocr}$：知识点出现时刻（OCR 检测到文字变化的时刻）
- $\mu$：先验延迟参数（默认 3 秒，可配置）
- $\sigma$：反应窗口宽窄（默认 1.5 秒）

### 6.2 知识点吸收度

$$Score_{K_i} = \frac{\sum W(t) \cdot CAS(t)}{\sum W(t)}$$

### 6.3 个体积极度（CAS）

$$CAS = \max(w_1 \cdot S_{action},\ w_2 \cdot S_{gaze})$$

采用 `max` 非线性融合：学生积极记笔记（动作高）或静坐紧盯黑板（视线高），均视为专注。

### 6.4 课堂教学效果指数（CTES）

$$CTES = \mu_{CAS} \cdot \exp(-\lambda \cdot \sigma_{CAS})$$

方差 $\sigma_{CAS}$ 作为指数级惩罚项，识别"两极分化"的危险课堂状态。

---

## 七、验证计划

### 自动验证

```bash
# 冒烟测试（全链路 mock）
python scripts/smoke_test.py --mock

# 单元测试
pytest tests/ -v

# 视频推理测试
python scripts/infer_video.py --source <video_path> --save

# YOLO26 检测评估
python scripts/eval_det.py --weights <best.pt> --data configs/scb_yolo.yaml
```

### 指标目标

| 层级 | 指标 | 目标值 |
|------|------|--------|
| 感知层 | 检测 mAP@50 | ≥ 0.75 |
| 感知层 | 追踪 IDF1 | ≥ 0.65 |
| 感知层 | 动作分类 F1 | ≥ 0.70 |
| 对齐层 | Score_Ki 与人工趋势 Spearman 相关性 | ≥ 0.60 |
| 系统层 | 端到端延迟（1080p 单帧） | < 100ms |
