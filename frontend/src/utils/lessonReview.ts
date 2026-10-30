import type { LessonPackage, LessonSummary, ReviewChatResponse, TimedEvidence, SlideSegment } from '../types/lessonReview'
import { validateCourseware } from './coursewareValidation'
import { validateActivityAnalysis, activityMarkdown } from './activityAnalysis'
import { validateModelUpgradeExperiment } from './modelUpgrade'
import { validateTeacherEvidence } from './teacherWorkspace'
type Obj = Record<string, any>
export function isObservedSlide(slide: SlideSegment): boolean {
  const quality = slide.quality
  const screenStatus = quality && typeof quality === 'object' && 'screen_status' in quality ? quality.screen_status : undefined
  return slide.method !== 'no_slide_evidence' && screenStatus !== 'no_slide' && screenStatus !== 'no_slide_evidence'
}
function object(value: unknown, field: string): Obj {
  if (!value || typeof value !== 'object' || Array.isArray(value)) throw new Error(`复盘数据格式错误：${field}`)
  return value as Obj
}
function list(value: unknown, field: string): any[] {
  if (!Array.isArray(value)) throw new Error(`复盘数据缺少列表：${field}`)
  return value
}
function string(value: unknown, field: string): string {
  if (typeof value !== 'string' || !value.trim()) throw new Error(`复盘数据缺少字段：${field}`)
  return value
}
function finite(value: unknown, field: string): number {
  if (typeof value !== 'number' || !Number.isFinite(value) || value < 0) throw new Error(`复盘数据时间无效：${field}`)
  return value
}
function ids(value: unknown, field: string): string[] { return list(value, field).map((v, i) => string(v, `${field}[${i}]`)) }
function interval(row: Obj, duration: number, field: string) {
  string(row.id, `${field}.id`); finite(row.start_sec, `${field}.start_sec`); finite(row.end_sec, `${field}.end_sec`)
  if (row.end_sec < row.start_sec || row.end_sec > duration + 1) throw new Error(`证据时间超出课程范围：${field}`)
}
export function reviewMediaUrl(value: unknown): string {
  if (!value) return ''
  if (typeof value !== 'string' || !value.startsWith('/api/review/media/') || value.includes('\\') || /(?:^|\/)\.\.(?:\/|$)/.test(value)) throw new Error('课程媒体必须使用同源 /api/review/media/ 地址。')
  return value
}
export function parseLessonList(value: unknown): LessonSummary[] {
  return list(object(value, 'lessons').lessons, 'lessons').map((raw, index) => {
    const row = object(raw, `lessons[${index}]`)
    return { id: string(row.id, 'lesson.id'), title: string(row.title, 'lesson.title'), duration_sec: finite(row.duration_sec, 'lesson.duration_sec'), status: string(row.status, 'lesson.status') }
  })
}
export function parseLessonPackage(value: unknown): LessonPackage {
  const data = object(value, 'package')
  if (data.schema_version !== '3.0' || data.data_source !== 'lesson_review') throw new Error('请选择 v3 lesson_review 课程包。')
  const lesson = object(data.lesson, 'lesson')
  string(lesson.id, 'lesson.id'); string(lesson.title, 'lesson.title')
  const duration = finite(lesson.duration_sec, 'lesson.duration_sec')
  if (!duration) throw new Error('课程时长必须大于零。')
  reviewMediaUrl(lesson.video_url)
  const video = object(lesson.source_video, 'lesson.source_video')
  string(video.filename, 'source_video.filename'); string(video.sha256, 'source_video.sha256')
  object(lesson.modalities, 'lesson.modalities'); object(lesson.provenance, 'lesson.provenance')
  const knownIds = new Set<string>()
  for (const field of ['transcript', 'slides', 'alignment']) for (const raw of list(data[field], field)) {
    const row = object(raw, field)
    interval(row, duration, field)
    if (knownIds.has(row.id)) throw new Error(`证据 ID 重复：${row.id}`)
    knownIds.add(row.id)
    if (field === 'alignment') { ids(row.transcript_ids, 'transcript_ids'); ids(row.concept_ids, 'concept_ids') }
    else if (typeof row.text !== 'string') throw new Error(`证据缺少文本：${row.id}`)
    if (field === 'transcript') string(row.source, 'transcript.source')
    if (field === 'slides') { reviewMediaUrl(row.image_url); finite(row.uncertainty_sec, 'slide.uncertainty_sec'); string(row.method, 'slide.method') }
  }
  if (data.evidence_blocks !== undefined) {
    const transcript = new Map<string, Obj>(data.transcript.map((row: Obj) => [row.id, row]))
    for (const raw of list(data.evidence_blocks, 'evidence_blocks')) {
      const block = object(raw, 'evidence_block')
      interval(block, duration, 'evidence_block')
      if (knownIds.has(block.id)) throw new Error(`证据 ID 重复：${block.id}`)
      knownIds.add(block.id)
      if (block.kind !== 'transcript') throw new Error(`检索段类型无效：${block.id}`)
      string(block.text, 'evidence_block.text')
      const refs = ids(block.source_refs, 'evidence_block.source_refs')
      if (!refs.length) throw new Error(`检索段缺少原始转写来源：${block.id}`)
      const sources = refs.map(id => {
        const source = transcript.get(id)
        if (!source) throw new Error(`检索段来源不属于原始转写：${id}`)
        return source
      })
      if (Math.abs(block.start_sec - Math.min(...sources.map(source => source.start_sec))) > .01 || Math.abs(block.end_sec - Math.max(...sources.map(source => source.end_sec))) > .01) throw new Error(`检索段时间与原始转写不符：${block.id}`)
    }
  }
  validateCourseware(data, reviewMediaUrl)
  for (const row of [...(data.courseware?.pages || []), ...(data.courseware?.matches || []), ...(data.courseware_evidence || []), ...(data.transcript_revisions || []), ...(data.revised_evidence_blocks || [])]) knownIds.add(row.id)
  validateProofreading(data, duration, knownIds)
  validateTeacherEvidence(data, knownIds)
  validateCorrectionComparison(data)
  validateModelUpgradeExperiment(data)
  validateActivityAnalysis(data, reviewMediaUrl)
  const graph = object(data.graph, 'graph')
  const nodeIds = new Set<string>()
  for (const raw of list(graph.nodes, 'graph.nodes')) {
    const node = object(raw, 'graph.node')
    string(node.id, 'node.id'); string(node.label, 'node.label')
    if (nodeIds.has(node.id)) throw new Error(`图谱节点 ID 重复：${node.id}`)
    nodeIds.add(node.id); ids(object(node.properties, 'node.properties').source_refs, 'node.source_refs')
  }
  for (const raw of list(graph.edges, 'graph.edges')) {
    const edge = object(raw, 'graph.edge')
    if (!nodeIds.has(edge.source) || !nodeIds.has(edge.target)) throw new Error('图谱关系引用了不存在的节点。')
    ids(object(edge.properties, 'edge.properties').source_refs, 'edge.source_refs')
  }
  const report = object(data.report, 'report')
  string(report.title, 'report.title')
  if (typeof report.summary !== 'string') throw new Error('报告摘要格式错误。')
  for (const section of list(report.sections, 'report.sections')) {
    string(section.title, 'report.section.title')
    if (typeof section.text !== 'string') throw new Error('报告段落格式错误。')
    ids(section.evidence_ids, 'report.evidence_ids')
  }
  ids(report.limitations, 'report.limitations'); list(data.snapshots, 'snapshots')
  return data as LessonPackage
}
function validateCorrectionComparison(data: Obj) {
  if (data.correction_comparison === undefined) return
  const comparison = object(data.correction_comparison, 'correction_comparison')
  const fail = (message: string): never => { throw new Error(`校对模型对比无效：${message}`) }
  if (comparison.schema_version !== '1.0' || comparison.scope !== 'same_lesson_controlled_model_comparison' || comparison.reference_status !== 'no_human_verbatim_reference') fail('版本、对比范围或人工参考状态不符。')
  if (comparison.source_video_sha256 !== data.lesson.source_video.sha256) fail('录像来源与当前课堂不符。')
  if (typeof comparison.evaluated_at !== 'string' || !Number.isFinite(Date.parse(comparison.evaluated_at))) fail('缺少有效评测时间。')
  ids(comparison.limitations, 'correction_comparison.limitations')
  const transcript = new Map<string, Obj>(data.transcript.map((row: Obj) => [row.id, row]))
  const runIds = new Set<string>()
  for (const raw of list(comparison.runs, 'correction_comparison.runs')) {
    const run = object(raw, 'comparison.run')
    if (!['qwen', 'cec4'].includes(run.id) || runIds.has(run.id)) fail('模型标识缺失或重复。')
    runIds.add(run.id)
    string(run.model_label, 'comparison.model_label'); string(run.model_sha256, 'comparison.model_sha256')
    for (const field of ['accepted_edit_count', 'model_proposal_count', 'guard_rejected_count', 'term_presence_count', 'terminology_group_count', 'model_accepted_edit_count', 'source_rule_accepted_count', 'additional_changed_spans_vs_rules']) {
      if (run[field] !== undefined && run[field] !== null && (!Number.isInteger(run[field]) || run[field] < 0)) fail(`${field} 必须为非负整数或未知。`)
    }
    for (const field of ['status', 'native_status']) if (run[field] !== undefined) string(run[field], `comparison.${field}`)
    if (run.memory !== undefined && run.memory !== null) {
      const memory = object(run.memory, 'comparison.memory')
      if (memory.unit !== 'MiB' || memory.scope !== 'whole_gpu_sampled') fail('显存测量单位或范围不符。')
      for (const field of ['baseline_mib', 'loaded_mib', 'peak_mib', 'sample_interval_ms']) {
        if (memory[field] !== undefined && memory[field] !== null && (typeof memory[field] !== 'number' || !Number.isFinite(memory[field]) || memory[field] < 0 || (field === 'sample_interval_ms' && memory[field] === 0))) fail(`${field} 测量值无效。`)
      }
    }
    // Comparison TIDs deliberately stay local to each run: they are not new evidence.
    for (const field of ['segments', 'native_segments']) {
      if (field === 'native_segments' && run[field] === undefined) continue
      const seen = new Set<string>()
      for (const raw of list(run[field], `comparison.${field}`)) {
        const row = object(raw, 'comparison.segment'), source = transcript.get(row.transcript_id)
        if (!source || seen.has(row.transcript_id) || row.original_text !== source.text || row.start_sec !== source.start_sec || row.end_sec !== source.end_sec) fail('段落、原文或时间与当前原始转写不符。')
        if (typeof row.text !== 'string') fail('段落缺少模型输出文本。')
        list(row.uncertainties, 'comparison.uncertainties'); seen.add(row.transcript_id)
      }
    }
  }
  if (!runIds.size) fail('缺少模型运行记录。')
}
function validateProofreading(data: Obj, duration: number, knownIds: Set<string>) {
  if (data.transcript_proofreading === undefined) {
    if (data.proofread_evidence_blocks?.length) throw new Error('校对证据缺少校对来源记录。')
    return
  }
  const proof = object(data.transcript_proofreading, 'transcript_proofreading')
  string(proof.schema_version, 'proofreading.schema_version'); string(proof.status, 'proofreading.status')
  string(proof.method, 'proofreading.method'); string(proof.model, 'proofreading.model')
  if (proof.human_verified !== false || proof.source_video_sha256 !== data.lesson.source_video.sha256) throw new Error('机器校对的核验状态或录像来源不符。')
  string(proof.source_asr_sha256, 'proofreading.source_asr_sha256')
  if (proof.counts !== undefined) object(proof.counts, 'proofreading.counts')
  object(proof.provenance, 'proofreading.provenance')
  if (proof.provenance.model_label !== undefined && (typeof proof.provenance.model_label !== 'string' || !proof.provenance.model_label.trim() || proof.provenance.model_label.length > 160 || /[\u0000-\u001f\u007f]/.test(proof.provenance.model_label))) throw new Error('校对模型显示标签无效。')
  const transcript = new Map<string, Obj>(data.transcript.map((row: Obj) => [row.id, row]))
  for (const raw of list(proof.alternate_evidence || [], 'proofreading.alternate_evidence')) {
    const row = object(raw, 'alternate_asr'); interval(row, duration, 'alternate_asr')
    if (knownIds.has(row.id)) throw new Error(`证据 ID 重复：${row.id}`)
    if (row.kind !== 'alternate_asr' || typeof row.text !== 'string') throw new Error('辅助语音识别来源格式无效。')
    string(row.original_id, 'alternate_asr.original_id'); string(row.variant, 'alternate_asr.variant')
    knownIds.add(row.id)
  }
  const sources = new Set([...knownIds, ...(data.courseware_evidence || []).map((row: Obj) => row.id)])
  const corrected = new Map<string, Obj>()
  const usedTranscript = new Set<string>()
  for (const raw of list(proof.segments, 'proofreading.segments')) {
    const row = object(raw, 'proofreading.segment'); interval(row, duration, 'proofreading.segment')
    if (knownIds.has(row.id)) throw new Error(`证据 ID 重复：${row.id}`)
    const original = transcript.get(row.transcript_id)
    if (!original || usedTranscript.has(row.transcript_id) || row.original_text !== original.text || row.start_sec !== original.start_sec || row.end_sec !== original.end_sec) throw new Error(`校对段与原始转写不符：${row.id}`)
    if (typeof row.text !== 'string') throw new Error(`校对段缺少文本：${row.id}`)
    const refs = ids(row.source_refs, 'proofreading.source_refs')
    if (!refs.includes(original.id) || refs.some(id => !sources.has(id))) throw new Error(`校对段来源无效：${row.id}`)
    let reconstructed = original.text as string
    for (const raw of list(row.edits, 'proofreading.edits')) {
      const edit = object(raw, 'proofreading.edit'); string(edit.original, 'edit.original'); string(edit.replacement, 'edit.replacement'); string(edit.reason, 'edit.reason')
      if (ids(edit.evidence_ids, 'edit.evidence_ids').some(id => !sources.has(id))) throw new Error('校对改动引用了未知证据。')
      if (!reconstructed.includes(edit.original)) throw new Error('校对改动与原文不符。')
      reconstructed = reconstructed.replace(edit.original, edit.replacement)
    }
    if (reconstructed !== row.text) throw new Error('校对文本与已记录改动不符。')
    list(row.uncertainties, 'proofreading.uncertainties')
    if (object(row.quality, 'proofreading.quality').status !== 'automatically_proofread_unreviewed') throw new Error('机器校对必须保留待核验状态。')
    knownIds.add(row.id); corrected.set(row.id, row); usedTranscript.add(row.transcript_id)
  }
  for (const raw of list(data.proofread_evidence_blocks || [], 'proofread_evidence_blocks')) {
    const row = object(raw, 'proofread_block'); interval(row, duration, 'proofread_block')
    if (knownIds.has(row.id)) throw new Error(`证据 ID 重复：${row.id}`)
    if (row.kind !== 'transcript_proofread') throw new Error('校对检索段类型无效。')
    string(row.text, 'proofread_block.text')
    const refs = ids(row.source_refs, 'proofread_block.source_refs')
    const segments = ids(row.proofreading_ids, 'proofread_block.proofreading_ids').map(id => corrected.get(id))
    if (!segments.length || segments.some(row => !row) || refs.some(id => !sources.has(id) && !corrected.has(id))) throw new Error('校对检索段来源无效。')
    const present = segments as Obj[]
    if (Math.abs(row.start_sec - Math.min(...present.map(s => s.start_sec))) > .01 || Math.abs(row.end_sec - Math.max(...present.map(s => s.end_sec))) > .01) throw new Error('校对检索段时间与校对来源不符。')
    if (row.text !== present.map(segment => segment.text).join(' ')) throw new Error('校对检索段文本与校对来源不符。')
    knownIds.add(row.id)
  }
}
export function formatReviewTime(seconds: number): string {
  const s = Math.max(0, Math.floor(Number.isFinite(seconds) ? seconds : 0))
  return `${Math.floor(s / 60).toString().padStart(2, '0')}:${(s % 60).toString().padStart(2, '0')}`
}
export function activeAt<T extends { start_sec: number; end_sec: number }>(rows: T[], time: number): T[] { return rows.filter(row => row.start_sec <= time && time < row.end_sec) }
export function evidenceIndex(data: LessonPackage): Map<string, TimedEvidence> {
  const entries: TimedEvidence[] = [...data.transcript, ...data.slides, ...(data.evidence_blocks || []), ...(data.courseware_evidence || []), ...(data.revised_evidence_blocks || []), ...(data.transcript_proofreading?.alternate_evidence || []), ...(data.transcript_proofreading?.segments || []).map(row => ({ ...row, kind: 'transcript_proofread' })), ...(data.proofread_evidence_blocks || []), ...(data.teacher_transcript || []).map(row => ({ ...row, kind: 'transcript_teacher' })), ...(data.teacher_evidence_blocks || []), ...(data.activity_analysis?.evidence || [])]
  const transcript = new Map(data.transcript.map(row => [row.id, row.text]))
  for (const segment of data.alignment) entries.push({ ...segment, text: segment.transcript_ids.map(id => transcript.get(id) || '').filter(Boolean).join(' ') || '课程对齐片段' })
  return new Map(entries.map(row => [row.id, row]))
}
export function resolveEvidence(data: LessonPackage, refs: string[]): TimedEvidence[] {
  const index = evidenceIndex(data)
  return [...new Set(refs)].map(id => index.get(id)).filter((row): row is TimedEvidence => Boolean(row)).sort((a, b) => a.start_sec - b.start_sec || a.end_sec - b.end_sec)
}
export function conceptIntervals(data: LessonPackage, nodeId: string): TimedEvidence[] {
  const node = data.graph.nodes.find(n => n.id === nodeId)
  if (!node) return []
  // Preserve distinct observations; repeated concepts never imply coverage across gaps.
  return resolveEvidence(data, [...node.properties.source_refs, ...data.alignment.filter(a => a.concept_ids.includes(nodeId)).map(a => a.id)])
}
export function parseChatResponse(value: unknown, data: LessonPackage): ReviewChatResponse {
  const result = object(value, 'chat')
  string(result.answer, 'answer'); string(result.model, 'model'); string(result.request_id, 'request_id')
  if (result.warnings !== undefined && (!Array.isArray(result.warnings) || result.warnings.length > 8 || result.warnings.some((warning: unknown) => typeof warning !== 'string' || warning.length > 500))) throw new Error('AI 回复的来源核验提示格式无效。')
  if (!['local_llm', 'remote_llm'].includes(result.source)) throw new Error('AI 回复未声明真实模型来源。')
  const index = evidenceIndex(data)
  const blockIds = new Set((data.evidence_blocks || []).map(block => block.id))
  const strictIds = new Set([...(data.courseware_evidence || []), ...(data.revised_evidence_blocks || []), ...(data.transcript_proofreading?.alternate_evidence || []), ...(data.transcript_proofreading?.segments || []), ...(data.proofread_evidence_blocks || []), ...(data.teacher_transcript || []), ...(data.teacher_evidence_blocks || []), ...(data.activity_analysis?.evidence || [])].map(row => row.id))
  for (const id of ids(result.evidence_ids, 'evidence_ids')) if (!index.has(id)) throw new Error(`AI 回复包含非本课证据：${id}`)
  for (const raw of list(result.citations, 'citations')) {
    const citation = object(raw, 'citation'); interval(citation, data.lesson.duration_sec, 'citation')
    string(citation.kind, 'citation.kind')
    if (typeof citation.text !== 'string') throw new Error('AI 引用文本缺失。')
    const source = index.get(citation.id)
    if (!source) throw new Error(`AI 引用不属于本课：${citation.id}`)
    const tolerance = blockIds.has(citation.id) || strictIds.has(citation.id) ? .01 : 1
    if (Math.abs(citation.start_sec - source.start_sec) > tolerance || Math.abs(citation.end_sec - source.end_sec) > tolerance) throw new Error(`AI 引用时间与原始证据不符：${citation.id}`)
    if (blockIds.has(citation.id) && (citation.kind !== 'transcript' || citation.text !== source.text)) throw new Error(`AI 检索段引用文本或类型与原始证据不符：${citation.id}`)
    if (strictIds.has(citation.id) && (citation.kind !== source.kind || citation.text !== source.text)) throw new Error(`AI 附加引用文本或类型与原始证据不符：${citation.id}`)
  }
  return result as ReviewChatResponse
}
export function buildReviewMarkdown(data: LessonPackage): string {
  const { lesson, report } = data
  const citationLines = (refs: string[]) => resolveEvidence(data, refs).map(e => `- [${e.id}] ${formatReviewTime(e.start_sec)}–${formatReviewTime(e.end_sec)} ${e.text}`).join('\n')
  return [
    `# ${report.title || lesson.title}`, `课程：${lesson.title}（${formatReviewTime(lesson.duration_sec)}）`,
    `来源：${lesson.source_video.filename}\nSHA-256：${lesson.source_video.sha256}`, report.summary,
    ...report.sections.map(section => `## ${section.title}\n\n${section.text}\n\n${citationLines(section.evidence_ids)}`),
    ...activityMarkdown(data),
    ...(data.transcript_proofreading ? [`## 机器校对转写（待人工核验）\n\n${data.transcript_proofreading.segments.map(row => `- [${row.id}] ${formatReviewTime(row.start_sec)}–${formatReviewTime(row.end_sec)} ${row.text}\n  原始转写 [${row.transcript_id}]：${row.original_text}\n  来源：${row.source_refs.join(', ')}`).join('\n')}`] : []),
    ...(data.courseware ? [`## 独立课件来源\n\n${data.courseware.filename}\nSHA-256：${data.courseware.sha256}\n\n${data.courseware.matches.map(m => `- ${formatReviewTime(m.start_sec)}–${formatReviewTime(m.end_sec)} ${m.status} ${m.page_id || '未匹配'}`).join('\n')}`, `## 课件辅助修订（原始转写保留）\n\n${(data.transcript_revisions || []).map(r => `- [${r.id}] ${formatReviewTime(r.start_sec)} 原文：${r.original_text}\n  建议：${r.corrected_text}\n  出处：${r.edits.flatMap(e => e.page_ids).join(', ')}`).join('\n') || '没有符合条件的修订。'}`] : []),
    `## 使用边界\n\n${report.limitations.map(line => `- ${line}`).join('\n')}`,
    `## 生成方式\n\n\`\`\`json\n${JSON.stringify(report.generated_by, null, 2)}\n\`\`\``,
    `## 处理来源\n\n\`\`\`json\n${JSON.stringify(lesson.provenance, null, 2)}\n\`\`\``,
    '机器转写、OCR 和自动图谱关联需结合原录像核验；行为线索不能证明知识掌握或教学效果。',
  ].join('\n\n')
}
