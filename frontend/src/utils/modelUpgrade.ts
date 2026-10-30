import type { ModelUpgradeExperiment } from '../types/lessonReview'

type Obj = Record<string, any>
const fail = (message: string): never => { throw new Error(`新模型试验无效：${message}`) }
function object(value: unknown): Obj {
  if (!value || typeof value !== 'object' || Array.isArray(value)) fail('缺少对象记录。')
  return value as Obj
}
function list(value: unknown): any[] {
  if (!Array.isArray(value)) fail('缺少列表记录。')
  return value as any[]
}
function text(value: unknown): string {
  if (typeof value !== 'string' || !value.trim()) fail('缺少非空文本或来源标识。')
  return value as string
}
function sameSegment(row: Obj, source: Obj | undefined): boolean {
  return Boolean(source && row.start_sec === source.start_sec && row.end_sec === source.end_sec && row.text === source.text)
}

/** Run-local transcript IDs never become global evidence or imply one-to-one alignment. */
export function validateModelUpgradeExperiment(data: Obj): void {
  if (data.model_upgrade_experiment === undefined) return
  const experiment = object(data.model_upgrade_experiment)
  if (experiment.schema_version !== '1.0' || experiment.reference_status !== 'no_human_verbatim_reference') fail('版本或人工参考状态不符。')
  if (experiment.source_video_sha256 !== data.lesson.source_video.sha256) fail('录像来源与当前课堂不符。')
  const asr = object(data.lesson.provenance.asr)
  const audioHash = asr.audio_sha256 || asr.source_audio?.sha256
  if (text(experiment.source_audio_sha256) !== text(audioHash)) fail('音频来源与当前课堂不符。')
  if (typeof experiment.evaluated_at !== 'string' || !Number.isFinite(Date.parse(experiment.evaluated_at))) fail('缺少有效运行时间。')
  list(experiment.limitations).forEach(text)
  const transcript = new Map<string, Obj>(data.transcript.map((row: Obj) => [row.id, row]))
  const runs = new Map<string, Obj>()
  for (const raw of list(experiment.asr_runs)) {
    const run = object(raw)
    if (!['baseline', 'qwen3_asr'].includes(run.id) || runs.has(run.id)) fail('识别运行标识缺失或重复。')
    text(run.model_label); text(run.source_asr_sha256)
    if (run.runtime !== undefined) object(run.runtime)
    const seen = new Set<string>()
    for (const rawSegment of list(run.segments)) {
      const row = object(rawSegment), id = text(row.id)
      if (seen.has(id)) fail('同一次识别的段落 ID 重复。')
      seen.add(id); text(row.text)
      if (![row.start_sec, row.end_sec].every(value => typeof value === 'number' && Number.isFinite(value) && value >= 0)
        || row.start_sec > row.end_sec || row.end_sec > data.lesson.duration_sec) fail('识别片段时间超出课堂范围。')
      if (run.id === 'qwen3_asr' && !sameSegment(row, transcript.get(id))) fail('新识别段落与当前原始转写不符。')
    }
    if (!seen.size) fail('缺少实际识别片段。')
    if (run.id === 'qwen3_asr' && seen.size !== transcript.size) fail('新识别未完整对应当前原始转写。')
    runs.set(run.id, run)
  }
  if (!runs.has('baseline') || !runs.has('qwen3_asr')) fail('缺少原识别或新识别记录。')
  const currentAsrHash = text(asr.source_asr_sha256)
  if (runs.get('qwen3_asr')!.source_asr_sha256 !== currentAsrHash) fail('新识别文件来源不符。')
  const correction = object(experiment.correction), proof = object(data.transcript_proofreading)
  text(correction.model_label); text(correction.model_sha256); text(correction.status)
  if (correction.source_asr_sha256 !== currentAsrHash || proof.source_asr_sha256 !== currentAsrHash) fail('校对输入与新识别文件来源不符。')
  if (proof.provenance.model_label !== undefined && correction.model_label !== proof.provenance.model_label) fail('校对模型标签与实际校对来源不符。')
  if (proof.provenance.model_sha256 !== undefined && correction.model_sha256 !== proof.provenance.model_sha256) fail('校对模型文件与实际校对来源不符。')
  if (correction.status !== proof.status) fail('校对状态与实际校对记录不符。')
  for (const field of ['accepted_edit_count', 'model_accepted_edit_count', 'source_rule_accepted_count']) {
    if (!Number.isInteger(correction[field]) || correction[field] < 0) fail('校对计数必须为非负整数。')
  }
  const acceptedEdits: Obj[] = proof.segments.flatMap((row: Obj) => row.edits)
  const modelAccepted = acceptedEdits.filter(edit => edit.origin === 'model').length
  if (correction.accepted_edit_count !== acceptedEdits.length || correction.model_accepted_edit_count !== modelAccepted
    || correction.source_rule_accepted_count !== acceptedEdits.length - modelAccepted
    || (proof.edit_count !== undefined && proof.edit_count !== correction.accepted_edit_count)) fail('校对计数与实际接受改动不符。')
  const corrected = new Map<string, Obj>(proof.segments.map((row: Obj) => [row.transcript_id, row]))
  const seen = new Set<string>()
  for (const raw of list(correction.segments)) {
    const row = object(raw), id = text(row.transcript_id), source = corrected.get(id)
    if (seen.has(id) || !sameSegment(row, source) || row.original_text !== source?.original_text) fail('校对段落与实际校对记录不符。')
    text(row.text)
    if (JSON.stringify(list(row.uncertainties)) !== JSON.stringify(source!.uncertainties)) fail('校对未确认内容与实际记录不符。')
    seen.add(id)
  }
  if (seen.size !== corrected.size) fail('校对片段未完整对应实际校对记录。')
}

/** Keep each lane's own boundaries; no positional or transcript-ID zip across ASR runs. */
export function modelUpgradeLanes(experiment: ModelUpgradeExperiment, time: number, all = false) {
  const lanes = [
    ...(['baseline', 'qwen3_asr'] as const).map(id => {
      const run = experiment.asr_runs.find(run => run.id === id)!
      return { id, title: id === 'baseline' ? '原识别' : '新识别', model: run.model_label, segments: run.segments }
    }),
    { id: 'correction', title: '保守校对', model: experiment.correction.model_label,
      segments: experiment.correction.segments.map(row => ({ ...row, id: row.transcript_id })) },
  ]
  return lanes.map(lane => ({ ...lane, segments: all ? lane.segments : lane.segments.filter(row => row.start_sec <= time && time < row.end_sec) }))
}

/** Millisecond precision for independent predicted boundaries, without minute-floor truncation. */
export function modelUpgradeTime(value: number): string {
  return `${value.toLocaleString('en-US', { maximumFractionDigits: 3, useGrouping: false })} 秒`
}
