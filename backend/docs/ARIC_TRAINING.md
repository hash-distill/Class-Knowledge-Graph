# ARIC 全景特征离线基线

本实现的任务是**原图级多标签行为出现判断**：利用整张课堂原图的既有浅层视觉特征，判断标注中是否出现各类行为。一张原图中多个人有同一行为时，该标签仍只计一次。

这不是原 ARIC 个体特征方法的复现，也不是逐人检测、逐人分类、人数统计或积极度模型。原个体特征包缺失；现有全景特征的确切编码器权重、浅层输出位置、归一化和空间 token 顺序尚未核验，不能把本分类头直接用于原始 RGB 图像或当前课堂视频。在线状态在 `contract.json` 中固定为 `offline_research_only`，`raw_rgb_compatible=false`。

输入仅来自本地 `ARIC_supplement/ARIC_CLIP_global_view_features.tar.gz`。已实际检查 `.npy` 数组为 `(1,401,768)`、`float32`、有限数值。每张原图对全部 401 个 token 求均值，再做 L2 归一化，得到 768 维输入；不假设 token 对应某种 20×20 空间网格。`numpy.load(..., allow_pickle=False)` 阻止对象反序列化。程序以 tar 流读取，不展开 41.5 GB 压缩包，只缓存选定原图的降维向量。

目标仅从 `annotations/*.json` 的 `instances[].label` 提取。生成式 caption、实例路径中的类别名、框坐标、音频和原数据集训练/测试标签索引均不作为分类器输入。旧索引仅用于审计其中的原图交叉情况。

## 划分和评测

- 从所有 annotation 原图 ID 中按种子和 SHA256 排序选择默认 2,048 张，不按压缩包先出现的连续编号取样。
- 按原图 ID 独立重新划分为 60% 训练、20% 验证、20% 测试；同一原图不跨集合。课次/原录像 ID 缺失，不声称跨课堂独立验证；相邻原图和同一教室仍可能跨集合。
- 只依据训练集正例数和负例数筛选类别，默认两者至少各 15 张。`metrics.json` 同时保留被排除类别和所有保留类别的分组支持度。
- 训练集拟合逐维均值/标准差和类别权重；使用固定的独立 logistic 分类头、Adam 和 L2 正则。类别权重来自训练支持度并限制最大为 10。
- 训练 240 epoch，在验证 BCE 最低处选模型；每类阈值只用验证集 F1 选择，网格为 0.10–0.90、步长 0.05。验证集无正例或全正例时固定阈值 0.5。测试集只做最后评测。
- 输出 Macro-F1、Micro-F1、mAP、精确匹配率、Hamming loss、每类 precision/recall/F1/AP、`tn/fp/fn/tp`。缺正例或缺负例的 AP 明确为 `null`，mAP 给出实际参与类别数。
- 对比训练集标签频率、全负、全正三个基线。频率基线每类默认 0.5 判定；全正基线补充说明 F1 对常量预测的表现。恒定频率分数的 AP 等于相应测试标签的出现比例。另保存模型使用默认 0.5 阈值的测试结果。
- sigmoid 输出受类别加权影响，是未校准分数，不是经独立校准的真实行为概率。

## 复跑

依赖仅 Python 和 NumPy，测试另需 pytest。实际本地运行环境和耗时保存在产物中；CPU 训练小型头不要求 GPU。

```powershell
cd D:\全球校园人工智能精英大赛\4C\课堂复盘
& D:\Anaconda\python.exe scripts/train_aric.py `
  --dataset-root 'E:\Research\Class Detection Dataset' `
  --output artifacts/learning/aric --max-images 2048 --seed 41 --epochs 240
& .\.venv-media\Scripts\python.exe -m pytest tests/test_aric_learning.py -q
```

第一次缓存需要顺序解压扫描压缩包，因为 gzip 无随机索引；后续复用缓存时，同时核对原图 ID、源压缩包的绝对解析路径、文件大小、纳秒修改时间，以及降维方法和源张量形状。任一来源/合同项不同就拒绝，并要求使用新输出目录；检查发生在改写已有审计产物之前。该来源指纹用于发现正常文件替换，并非整个 41.5 GB 压缩包的密码学内容校验。更改数据源、选样或种子时请另设输出目录，避免混用不同实验的产物。

源压缩包已移走时不能通过缓存准备路径跳过来源核验；使用下述 `--from-bundle` 可在不访问源压缩包的情况下复跑已明确导出的训练包。

缓存完成后，在另一台机器上只需复制 `learning/aric.py`、`learning/__init__.py`、`scripts/train_aric.py`、`training_bundle.npz` 和相邻的 `contract.json`，用 Linux 或 Windows 的 NumPy 环境运行：

```bash
python scripts/train_aric.py --from-bundle /path/to/training_bundle.npz --output /path/to/rerun --seed 41 --epochs 240
```

训练包包含只读用途的降维特征、原图 ID、标签和明确的三组行号。程序拒绝行号重复、重叠、遗漏或原图 ID 重复。不要把另一种编码器的 768 维向量当作等价输入。

## 产物与预测接口

默认目录为 `artifacts/learning/aric/`：

| 文件 | 内容 |
|---|---|
| `data_audit.json` | 数据规模、行为支持度、旧索引原图重叠 |
| `split_manifest.json` | 全部选定原图、真实标签、annotation SHA256、三组归属 |
| `features.npz` / `features.json` | 768 维缓存；各源 npy 成员 SHA256、大小和来源路径 |
| `training_bundle.npz` | 可移植训练包，包含全部候选类别和分组行号 |
| `preparation.json` | 缓存/准备耗时、缓存及训练包哈希、运行环境 |
| `model.npz` | 权重、偏置、训练均值/标准差、类别、验证阈值 |
| `contract.json` | 输入合同、任务、预处理已知/未知项、禁止未经核验的 RGB 接入 |
| `metrics.json` | 实际评测、基线、支持度、配置、CPU/耗时、模型哈希和导出路径 |
| `test_predictions.npz` | 测试原图 ID、真实目标、分数、阈值、类别，供复核 |

```python
import numpy as np
from learning.aric import load_head, predict_reduced

model = load_head('artifacts/learning/aric/model.npz')
with np.load('artifacts/learning/aric/features.npz', allow_pickle=False) as cache:
    scores = predict_reduced(model, cache['features'][:2])
present = scores >= model['thresholds']
```

缓存与模型重载验证的是相同降维输入的可复现推理；不证明新视频、图像编码或跨课次泛化能力。数据是否可用于参赛公开展示/再分发仍须单独核对授权。
