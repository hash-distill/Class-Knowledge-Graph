# 纠错模型比较协议

## 运行比较

`scripts/compare_correction_models.py` 读取已冻结的运行结果，输出 JSON、CSV 和 Markdown。它不请求模型，也不写入课堂目录。模型推理和 GPU 采样由运行脚本完成。

在项目目录运行：

```powershell
python scripts/compare_correction_models.py `
  --media-dir artifacts/unified-chemistry-001-9 `
  --run qwen=artifacts/verification/cec4-comparison/qwen-run.json `
  --run cec4=artifacts/verification/cec4-comparison/cec4-run.json `
  --deterministic-run artifacts/verification/cec4-comparison/source-only.json `
  --output-dir artifacts/verification/cec4-comparison/report
```

结果文件包括 `comparison.json`、`comparison.md`、`segments.csv` 和 `proposals.csv`。`segments.csv` 保存全部原段及每个输出层的文本。`proposals.csv` 保存模型候选、守卫处置和原文定位失败；JSON 另存全部来源规则候选账本。

## 冻结输入和输出层

主实验固定 `unified-chemistry-001-9/asr.json` 的 28 段、时间和原文。36 段无提示 ASR、30 段同音频第二识别、43.5—47.5 秒实测投影词汇是辅助证据。181 个行为观测继续由课堂流程保存；它们不参与文字正确率评分。

模型应顺序运行，使用同一显卡、原段、辅助来源和守卫版本。结构化比较还需相同 system、prompt、temperature、seed、n_ctx、max_tokens 和 request_format。记录实际发送的请求及返回值，采样温度为 0 也不能保证跨运行逐位相同。

比较器分开记录这些结果：

- 原模型候选：按原文坐标应用候选，尚未经过语义守卫。重复子串无法定位或候选重叠时，保留失败原因，不猜测目标位置。
- 守卫后文本：`proofread_transcript` 返回的全部 `segments`。
- 纯来源规则：空模型候选下运行同一来源词汇、第二识别与重复词上下文规则。接受来源不得包含模型候选。
- 原生整段纠错：保存原模型输出与解析后的纠错文本。原生提示与结构化候选提示不同，报告会把它们列为不同条件。

纯来源规则与两个模型都得到相同的 13 处修改时，这 13 处不能全部归因于两个模型。报告同时给出接受来源和相对规则的额外变化。额外变化仍需核听。

## 运行结果契约

`--run LABEL=PATH` 可直接读取 `proofread_transcript` 结果。为保留采样和显存记录，建议使用以下包装：

```json
{
  "label": "cec4-controlled",
  "track": "controlled_json",
  "conditions": {
    "temperature": 0,
    "seed": 42,
    "n_ctx": 24576,
    "max_tokens": 5000,
    "request_format": "json_schema"
  },
  "runtime": {
    "scope": "single_model_loaded_correction_only",
    "wall_time_sec": 0,
    "baseline_device_used_mib": 0,
    "peak_device_used_mib": 0,
    "sampling_interval_sec": 0.2,
    "measurement_source": "nvidia-smi",
    "gpu_name": "fill from measurement"
  },
  "result": {}
}
```

示例中的 0 仅展示字段类型，实际运行必须填入实测值。`runtime` 原样保留；缺少记录时输出为空，不能从模型参数量推算峰值显存。明确测量范围，区分模型加载、纠错、全流程和其他进程占用。采样峰值可能漏掉采样间隔内的尖峰。

`result` 必须含 `source_asr_sha256` 和实际来源哈希。录像来源使用 `source_video_sha256`。独立 WAV 来源使用 `source_audio_sha256`；原 ASR 的 provenance 需记录真实 `source_audio.sha256`。守卫结果还需全部 `segments`，每段包含原 `transcript_id`、`original_text`、`start_sec`、`end_sec` 和结果 `text`。比较器拒绝来源哈希、原文、顺序或时间不一致的结果。

v8 的 `model_proposals` 保存原始解析答案、`system_digest`、`prompt_digest`、目标段和模型 metadata。`proposal_ledger` 的每行包含 `transcript_id`、`channel`、`candidate`、`disposition` 和 `reason`。接受编辑另有 `origin`。历史 v7 没有保存完整模型提议，比较器将原模型层标为不可用。

原生输出可放在包装的 `native_outputs` 中：

```json
{
  "track": "native_text",
  "native_outputs": [
    {
      "transcript_id": "T0001",
      "original_text": "冻结的原始段文本",
      "corrected_text": "解析后的原生模型纠错文本",
      "metadata": {"raw_response": "保留原始返回"}
    }
  ],
  "result": {
    "source_asr_sha256": "same immutable ASR sha256",
    "source_video_sha256": "same source video sha256"
  }
}
```

原生输出也必须覆盖全部原段。只有同时提供守卫 `segments` 时，才有守卫后结果层。配置使用项目自定原生提示时，应明确写明“项目提示”；未经核验的提示不能称为官方推荐配置。

## 主课的质量边界

当前化学课没有人工逐字稿。原 ASR、第二识别器和投影 OCR 都可能出错。报告不计算纠正率、真实误改率或 CER。

14 组术语词形记录在 `tests/fixtures/cec4/chemistry-001-9-terminology.json`。每组保留实际来源，词表在观察本课后固定。它只检查规范写法的出现、保留和改变，存在选择偏差。嵌套词形可以重复计数，例如“氨基”和“氨基酸”。14 组词形不能当作 14 个人工标注错误。

字符修改用逐段 Levenshtein 最短编辑对齐，单位是 Unicode 码点。相邻修改合为一个跨度，保留输入、输出的起止位置。字符编辑距离和跨度数量不是纠正数量。

数字、中文数词、否定字、公式运算符与字母序列变化会触发待审提示。“二发→阿尔法”和“未经→味精”也可能触发提示。保护项提示和守卫拒绝只能称为计数，不能当作已确认误改。

## 独立逐字参考和外部测试

只有同一录像、同一时钟的人工或官方逐字参考才可启用 `--reference`。参考文件需满足以下声明，并与全部原段一一对应：

```json
{
  "reference_kind": "official_verbatim",
  "human_verified": true,
  "independent_of_model_outputs": true,
  "used_as_model_input": false,
  "clock": "source_video_seconds",
  "source_video_sha256": "same source video sha256",
  "provenance": {"source": "official URL and immutable annotation hash"},
  "segments": [
    {"transcript_id": "T0001", "start_sec": 0, "end_sec": 12.3, "text": "独立逐字参考"}
  ]
}
```

这些字段是可检查的来源声明，比较器不能独立认证声明真伪。另一个主题相近的 PPT、教案或字幕不能充当本课真值。只拿到整段官方参考时，应把整段音频作为一个原始 ASR 评测段；不要根据参考文字猜测词级时间。

参考评测同时输出严格 CER 和去空白、去 Unicode 标点后的 CER。后者先执行 NFKC，保留字母大小写和数学符号。两套 CER 都汇总字符错误数后再除以参考字符数，不取各段 CER 的简单平均。

独立 WAV 的参考改用 `time_base: "source_audio_seconds"` 和 `source_audio_sha256`。不需要给无声 PPT 视频编造音轨来源。

另有严格及去标点后的参考字符修复、损伤计数。先把原始 ASR 和纠错文本分别与参考对齐。原来错误、后来匹配的参考位置计为修复；原来匹配、后来错误的位置计为损伤。插入错误单独计数。修复率分母为原来错误的参考位置，损伤率分母为原来正确的参考位置。重复文本可能有多个最优对齐，因此字段标记 `alignment_dependent=true`。这些指标与“所有模型提议中的误改比例”含义不同。

Chinese-LiPS 的 3 个官方 test 样本作为独立外部检查，不能替换主化学课。`external-chinese-lips/model-inputs.json` 仅供 ASR、OCR 与模型读取。`gold-references.json` 只供评分，不能进入转写提示、课件词汇或纠错上下文。参考与主课没有同一录像关系，必须分别报告。样本少且模型训练集未知，这次外部检查不能证明没有训练数据重叠，也不能代表全数据集表现。

对外部样本运行时，使用单独媒体目录与输出目录，并传入不含本课词表的文件，例如 `--terminology empty-terms.json`，文件内容为 `{"groups":[]}`。每个样本可以有一份参考报告，汇总时应合计 CER 分子和分母。

3 段独立 ASR 已冻结到 `external-chinese-lips/evaluation/<id>/asr.json` 后，可生成评分参考：

```powershell
python scripts/prepare_external_correction_references.py `
  --external-dir artifacts/verification/cec4-comparison/external-chinese-lips
```

工具检查实际 WAV 和标注 SHA，再读取采样率与帧数计算时长。每个 ASR 必须只有一个完整音频评测段 `T0001`，范围为 0 至实际 WAV 时长。它写入各样本的 `reference.json` 和评测目录的 `empty-terms.json`，保留 ASR 原文件。该脚本仅供评分阶段调用。

## 验证

```powershell
python -m pytest tests/test_correction_comparison.py -q
```

测试使用独立合成夹具，不加载权重、不请求模型。真实模型完成度、来源保留和 GPU 测量仍需检查运行产物。

## 顺序推理和 GPU 实测

确认固定权重与运行时已安装后运行：

```powershell
.venv-media/Scripts/python.exe scripts/run_correction_benchmark.py `
  --media-dir artifacts/unified-chemistry-001-9 `
  --video 'E:\Research\Class Detection Dataset\TS-Interact\Chemisty\video\Chemistry_001_part9.mp4' `
  --output-dir artifacts/verification/cec4-comparison `
  --external-dir artifacts/verification/cec4-comparison/external-chinese-lips `
  --request-timeout 900
```

脚本先运行零模型候选的来源规则，再顺序加载 Qwen 和 CEC4。每次加载前停止两个已记录的模型角色；启动和停止都调用已有 PID、程序路径与进程开始时间核验脚本。每个模型运行结构化全课校对、28 段原生纠错及 3 段外部原生纠错。原生提示对两个模型采用同一条经核验的上游专业纠错提示；温度为 0.1、上限 1000 tokens。它与模型卡的确定性解码设置有差异，应称为本项目比较设置。

每段返回立即保存，保留原返回、思考文本、token 用量、finish_reason、请求参数及摘要。失败段保留原文并显式计为失败。`--native-only` 和 `--controlled-only` 可单独补跑。`--force-controlled` 会先归档现有结果；辅助输入、守卫实现、模型配置发生变化时，旧结构化结果不能直接复用。原生断点若跨多次运行，保留各次采样文件和峰值，再汇总观测峰值。

GPU 采样目标间隔为 0.2 秒，记录实际间隔、完整 GPU 名称、UUID、驱动、空闲基线、加载后占用及各阶段采样峰值。超过 5 秒的生成采样空档会标记延迟比较无效，原始耗时继续保留。Windows 运行期间仅对当前线程临时阻止空闲休眠，退出后恢复；不改永久电源策略。

2026 年 10 月 28 日的一次旧尝试受到约 7 小时休眠影响，另一次使用 180 秒 HTTP 时限而超时。这些结果保存在 `attempts/`。正式比较使用两模型均完成、HTTP 时限同为 900 秒的新结构化运行。对应采样文件为 `gpu-samples-20261028-080448.jsonl`；旧中断窗口不计入正式延迟结论。
