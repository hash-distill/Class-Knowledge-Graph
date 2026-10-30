# Qwen ASR 与纠错升级评测

## 目的与输入

`scripts/compare_qwen_upgrade.py` 在相同 4 个音频上比较 ASR 和纠错模型。
主课堂是 `unified-chemistry-001-9` 的完整 90 秒 WAV。
另 3 段沿用已冻结的 Chinese-LiPS 官方测试音频。
脚本核验实际 WAV 的 SHA-256，并保留每个 ASR 的原分段、时间和来源信息。

| 项目 | 旧输入 | 升级输入 |
| --- | --- | --- |
| 主课堂 ASR | 现有 OCR 词汇提示 Whisper，28 段 | Qwen3-ASR-1.7B 的实际输出 |
| 外部 ASR | 现有独立 SeacoParaformer 输出 | Qwen3-ASR-1.7B 的实际输出 |
| 纠错模型 | 当前 Qwen3-4B | Qwen3.5-4B |

每个音频形成旧 ASR／新 ASR与旧纠错／新纠错的 4 格比较。
主课堂无提示 Whisper 另存于 `diagnostic_asr_runs.old_unguided`，只作诊断。
历史 native 纠错使用过其他提示，不能替代本次公平比较中的任何一格。

## 模型输入与评分隔离

`prepare` 和 `run` 不读取官方参考稿、金标和人工标注。
它们只读取模型输入清单、音频身份、ASR 文件和运行配置。
原生纠错只接收一个 ASR 的完整拼接文本；不接收课件 OCR、第二识别器或参考答案。
拼接保持 ASR 自身段落顺序，不添加分隔符，也不预先改写文字。

两种纠错模型使用脚本内同一 `NATIVE_SYSTEM`。
提示要求保留口语、提问、重复和不完整句，只修正上下文可明确判断的误识。
固定条件是 `temperature=0.1`、`max_tokens=2000`、关闭 thinking、纯文本输出。
请求未发送随机种子，服务端默认种子会影响重新生成的可重复性。
保存的输入、输出与评分结果可按哈希复核。

`report` 才读取 scorer-only 参考文件。
脚本复核官方人工标注文件的固定哈希，检查其原文与金标、评分参考三者相同。
参考稿必须覆盖实际完整 WAV，并声明独立于模型输出、未用于模型输入。

## 评分单位与解释

两个识别器可以有不同的 T 编号和分段。
脚本按整段音频评分，不把不同识别器的 T 编号逐项配对。
每次评分的分母是对应官方整段参考的字符数。

| 指标 | 定义 |
| --- | --- |
| 严格 CER | Unicode 码点的精确 Levenshtein 距离，保留空白和标点 |
| 规范化 CER | 先做 NFKC，再去空白和 Unicode 标点；不转换数字、同义词和口语 |
| 修复参考字符 | 原先对齐为错误、纠错后对齐为正确的参考字符位置 |
| 损伤原正确字符 | 原先对齐正确、纠错后对齐为错误的参考字符位置 |
| 插入错误 | 不对应参考字符位置，单独统计纠错前后数量 |

重复文字的最优对齐可能不唯一。
脚本使用确定的平局顺序：替换、删除、插入。
修复率和损伤率依赖该对齐，不等同于人工核听的改动判定。

外部微平均只使用 4 格均完成的相同样本集合，并公开样本 ID 和分母。
缺失或失败输出的文本和指标为 `null`，不替换成空串或原文。
汇总同时提供原始 ASR 基线和固定另一项后的 CER 差值。
CER 差值越小越好；只有 3 段外部样本，训练重叠情况未知。

主课堂没有独立、同源、逐字核验的人工稿。
其 CER、修复率和损伤率保持未评估。
不同机器转写、课件或同名异题标注都不能替代人工稿。

## 执行顺序

以下命令从课堂复盘目录执行。
`run` 只访问已经启动的端点；主任务负责模型切换与 GPU 使用权。
新 ASR 尚未完成时，`prepare` 会保留缺失槽位。
新 ASR 到齐后重新执行 `prepare`，再补跑对应请求。
主课堂发布包装后，使用 `prepare --new-primary-asr artifacts/qwen35-chemistry-001-9/asr.json`。
脚本核对包装前后的原生文字、时间、编号及来源哈希。
这样生产守卫与对照试验绑定同一个识别文件，包装不会改变识别内容。

```powershell
.\.venv-media\Scripts\python.exe scripts/compare_qwen_upgrade.py prepare
.\.venv-media\Scripts\python.exe scripts/compare_qwen_upgrade.py run --asr old --corrector old_qwen --profile configs/local_llm.json
.\.venv-media\Scripts\python.exe scripts/compare_qwen_upgrade.py run --asr qwen3 --corrector old_qwen --profile configs/local_llm.json
# 主任务切到 Qwen3.5 后执行。
.\.venv-media\Scripts\python.exe scripts/compare_qwen_upgrade.py run --asr old --corrector qwen35 --profile configs/qwen35_llm.json
.\.venv-media\Scripts\python.exe scripts/compare_qwen_upgrade.py run --asr qwen3 --corrector qwen35 --profile configs/qwen35_llm.json
.\.venv-media\Scripts\python.exe scripts/compare_qwen_upgrade.py report
```

同一输入、模型和请求条件下的完成记录会复用。
输入哈希或条件发生变化时，脚本拒绝覆盖旧记录。
`--retry-failed` 仅重试失败记录，并单独保留前次失败文件。

运行记录保存在 `artifacts/verification/qwen-upgrade-20261028/evaluation/`。
`run-plan.json` 是模型输入计划；`corrections/<纠错>/<ASR>/<样本>.json` 保存逐次请求。
`comparison.json`、`comparison.md` 和 `segments.csv` 保存可复核的结果。
`gpu-*.jsonl` 是整卡采样，不是某个进程独占的显存。
纠错阶段采样峰值可能遗漏采样间隙的瞬时峰值。
模型已加载时的请求前占用单列记录，空闲基线和模型加载耗时保持未测量。

## 与生产守卫结果的关系

整段 native 输出是未经验证的模型假设，仅用于本次评分。
它没有原 ASR 的逐词时间，不能按原 T 编号直接写入生产时间线。
生产守卫继续使用原有结构化候选与来源规则。

报告可附加独立的生产守卫记录，例如：

```powershell
.\.venv-media\Scripts\python.exe scripts/compare_qwen_upgrade.py report --guarded-run old:old_qwen=实际旧模型守卫记录.json --guarded-run old:qwen35=实际新模型守卫记录.json
```

`ASR:纠错` 中的 ASR 只允许 `old`、`qwen3`。
纠错只允许 `old_qwen`、`qwen35`。
附加结果保留原模型候选、拒绝原因、接受编辑来源和守卫后文本。
来源规则与模型共享的修改不能全部算作模型收益。
