# 预训练权重与来源

本项目直接下载并使用已有权重，不在本机训练。固定文件清单在 `configs/pretrained_models.json`；下载器仅使用 Python 标准库，下载阶段不会反序列化模型或启动推理。来源和文件摘要于 2026-10-25 核对。

## 固定文件

| 清单键 | 本地文件（相对项目目录） | 固定来源 | 大小（字节） |
| --- | --- | --- | ---: |
| `behavior_scb` | `artifacts/models/scb_yolov9e.pt` | 作者链接的 Hugging Face Space，提交 `dd6c9381099400c4c64135335cb531bdad35ed1d`，原文件 `best.pt` | 117275802 |
| `pose_coco` | `artifacts/models/yolo26n-pose.pt` | Ultralytics 官方 assets，发布 `v8.4.0`，asset ID `340096236` | 7878574 |

SHA-256：

```text
behavior_scb  4ff890b66e69ed809f7bcbfa56eb4348404a1298457d772d728291d4027f6561
pose_coco     eb3bb8268828aeaf515cec23a4bfafd793944a86fe9af94ba7823609c14522a9
```

行为权重来自 [nontaphatfirm/Classroom-Behavior-Detector 作者仓库](https://github.com/nontaphatfirm/Classroom-Behavior-Detector) 的 Deployment 链接。作者说明采用 SCB 数据，并链接到 [Student-Behavior-Detector Space](https://huggingface.co/spaces/nontaphat/Student-Behavior-Detector)。本项目固定其中的提交版本，不跟随 `main`；SHA-256 和大小取自该版本的 [Git LFS 指针](https://huggingface.co/spaces/nontaphat/Student-Behavior-Detector/raw/dd6c9381099400c4c64135335cb531bdad35ed1d/best.pt)。本地更名为 `scb_yolov9e.pt`，不会改变权重内容。实际权重中的架构配置为 `yolov9e.yaml`，因此使用 YOLOv9e 名称。

姿态权重来自 [Ultralytics 官方 v8.4.0 发布](https://github.com/ultralytics/assets/releases/tag/v8.4.0)。大小和 SHA-256 与 [GitHub release asset API](https://api.github.com/repos/ultralytics/assets/releases/assets/340096236) 一致。发布标签本身可更新，因此必须同时保留并检查固定摘要。

## 下载与校验

在 `Class_Detection` 目录运行：

```powershell
python scripts/fetch_pretrained.py
python scripts/fetch_pretrained.py --verify-only
```

只下载一个模型或显式修复损坏文件：

```powershell
python scripts/fetch_pretrained.py --model pose_coco
python scripts/fetch_pretrained.py --model behavior_scb --repair
```

- 默认保存到项目内 `artifacts/models`，与调用者当前工作目录无关；可用 `--output-dir` 指定目录。
- 已存在的文件每次都检查完整大小和 SHA-256。校验失败会报错，不会把同名文件当作可信权重，也不会静默覆盖。
- 下载使用同目录唯一 `.part` 文件，逐块计算摘要；只有大小与摘要都一致才原子替换目标文件。网络中断、长度不符或摘要不符会清理本次临时文件。
- `--repair` 仅在新文件完全验证后替换旧文件；下载失败时保留旧文件以便排查。旧文件仍不可用于推理。
- `--verify-only` 不访问网络，缺失或损坏即返回非零退出码。
- 脚本不调用 Ultralytics 自动下载，不会将新模型版本或其他类别权重作为降级替代。下载成功只证明文件完整，不证明目标课堂准确率。

离线测试：

```powershell
python -m unittest discover -s tests -p test_model_download.py
```

测试使用内存响应和临时文件，不隐式联网、不下载真实权重，覆盖错误大小、错误摘要、中断清理、缓存校验和修复保留原文件等行为。

## 模型语义与验证边界

对下载且校验通过的两个 PyTorch ZIP 文件，使用标准库 `zipfile` 与 `pickletools` 静态读取 `data.pkl`，未执行其中的 pickle。已核对：

| 模型 | 权重内配置 | 权重内类别 |
| --- | --- | --- |
| 行为检测 | `yolov9e.yaml` | `0: hand-raising`、`1: reading`、`2: writing` |
| 人体姿态 | `yolo26n-pose.yaml` | `0: person`；关键点形状 `[17, 3]` |

这些类别及关键点形状也记录在清单中。随后在 `torch 2.8.0+cpu`、`ultralytics 8.4.188`、`dill 0.4.1` 环境中完整加载两个模型，逐项断言任务、类别编号及名称、架构配置、关键点形状与清单一致。行为权重需要 `dill` 依赖；模型加载设置 `YOLO_AUTOINSTALL=false`，避免加载阶段隐式安装依赖，并把 `YOLO_CONFIG_DIR` 设为项目内 `artifacts/ultralytics`。

2026-10-25 使用项目既有 `test_images/image_test.jpg`（原图 3648 × 2736）各执行一次 CPU 推理，统一设置 `imgsz=960`、`conf=0.25`、`save=False`。行为模型返回 4 个框（3 个 `reading`、1 个 `writing`），姿态模型返回 18 个 `person` 框，关键点输出形状为 `[18, 17, 3]`。这是两个模型各自的单图可运行性检查，不是两者融合结果；未使用人工标签计算准确率，也未训练模型。结果数量受阈值与输入尺寸影响，不作为固定性能指标。

**类别编号及确切名称以权重内的 `names` 为准**，不能从数据集 README 猜测顺序，也不能把此模型描述为通用的七类课堂行为模型。姿态模型提供人体关键点；姿态输出不等同于注意力、理解程度或学习效果。目标课堂上的遮挡、小目标、座位视角与视频行为效果，需要另行标注评估；来源页面或作者报告不是本项目的独立测试结果。

## 许可来源必须分开理解

1. 固定版本的 [Space README](https://huggingface.co/spaces/nontaphat/Student-Behavior-Detector/blob/dd6c9381099400c4c64135335cb531bdad35ed1d/README.md) 标记 `license: mit`。这只是该 Space 的许可元数据，不能据此认定模型、训练数据及所有上游权益均为 MIT。
2. [SCB 上游仓库](https://github.com/Whiffe/SCB-dataset#4-commercial-use-restrictions) 声明学术研究、个人学习和非商业用途限制，并要求商业用途取得书面许可。该仓库明确把模型权重列入其限制范围。其声明不能被下游 Space 的 MIT 标记自动消除。
3. [Ultralytics 官方许可页面](https://www.ultralytics.com/license) 表示其训练模型默认采用 AGPL-3.0，同时提供 Enterprise 许可。框架与模型的这部分要求需要单独满足。

因此，这组权重可公开下载，但本项目**不把它们标为完全宽松许可或已取得商业授权**。本次接入没有获得任何额外商业许可，也不将下载校验等同于许可确认。
