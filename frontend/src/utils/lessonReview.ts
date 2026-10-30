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
  if (!value || typeof value !== 'object' || Array.isArray(value)) throw new Error(`Invalid review data format: ${field}`)
  return value as Obj
}
function list(value: unknown, field: string): any[] {
  if (!Array.isArray(value)) throw new Error(`Review data is missing a list: ${field}`)
  return value
}
function string(value: unknown, field: string): string {
  if (typeof value !== 'string' || !value.trim()) throw new Error(`Review data is missing a field: ${field}`)
  return value
}
function finite(value: unknown, field: string): number {
  if (typeof value !== 'number' || !Number.isFinite(value) || value < 0) throw new Error(`Invalid review data time: ${field}`)
  return value
}
function ids(value: unknown, field: string): string[] { return list(value, field).map((v, i) => string(v, `${field}[${i}]`)) }
function interval(row: Obj, duration: number, field: string) {
  string(row.id, `${field}.id`); finite(row.start_sec, `${field}.start_sec`); finite(row.end_sec, `${field}.end_sec`)
  if (row.end_sec < row.start_sec || row.end_sec > duration + 1) throw new Error(`Evidence time falls outside the lesson: ${field}`)
}
export function reviewMediaUrl(value: unknown): string {
  if (!value) return ''
  if (typeof value !== 'string' || !value.startsWith('/api/review/media/') || value.includes('\\') || /(?:^|\/)\.\.(?:\/|$)/.test(value)) throw new Error('Lesson media must use a same-origin /api/review/media/ URL.')
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
  if (data.schema_version !== '3.0' || data.data_source !== 'lesson_review') throw new Error('Please select v3 lesson_review lesson package.')
  const lesson = object(data.lesson, 'lesson')
  string(lesson.id, 'lesson.id'); string(lesson.title, 'lesson.title')
  const duration = finite(lesson.duration_sec, 'lesson.duration_sec')
  if (!duration) throw new Error('Lesson duration must be greater than zero.')
  reviewMediaUrl(lesson.video_url)
  const video = object(lesson.source_video, 'lesson.source_video')
  string(video.filename, 'source_video.filename'); string(video.sha256, 'source_video.sha256')
  object(lesson.modalities, 'lesson.modalities'); object(lesson.provenance, 'lesson.provenance')
  const knownIds = new Set<string>()
  for (const field of ['transcript', 'slides', 'alignment']) for (const raw of list(data[field], field)) {
    const row = object(raw, field)
    interval(row, duration, field)
    if (knownIds.has(row.id)) throw new Error(`Evidence ID Duplicate: ${row.id}`)
    knownIds.add(row.id)
    if (field === 'alignment') { ids(row.transcript_ids, 'transcript_ids'); ids(row.concept_ids, 'concept_ids') }
    else if (typeof row.text !== 'string') throw new Error(`Evidence is missing text: ${row.id}`)
    if (field === 'transcript') string(row.source, 'transcript.source')
    if (field === 'slides') { reviewMediaUrl(row.image_url); finite(row.uncertainty_sec, 'slide.uncertainty_sec'); string(row.method, 'slide.method') }
  }
  if (data.evidence_blocks !== undefined) {
    const transcript = new Map<string, Obj>(data.transcript.map((row: Obj) => [row.id, row]))
    for (const raw of list(data.evidence_blocks, 'evidence_blocks')) {
      const block = object(raw, 'evidence_block')
      interval(block, duration, 'evidence_block')
      if (knownIds.has(block.id)) throw new Error(`Evidence ID Duplicate: ${block.id}`)
      knownIds.add(block.id)
      if (block.kind !== 'transcript') throw new Error(`Invalid retrieval block kind: ${block.id}`)
      string(block.text, 'evidence_block.text')
      const refs = ids(block.source_refs, 'evidence_block.source_refs')
      if (!refs.length) throw new Error(`Retrieval block is missing original transcript sources: ${block.id}`)
      const sources = refs.map(id => {
        const source = transcript.get(id)
        if (!source) throw new Error(`Retrieval block source is not in the original transcript: ${id}`)
        return source
      })
      if (Math.abs(block.start_sec - Math.min(...sources.map(source => source.start_sec))) > .01 || Math.abs(block.end_sec - Math.max(...sources.map(source => source.end_sec))) > .01) throw new Error(`Retrieval block times do not match the original transcript: ${block.id}`)
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
    if (nodeIds.has(node.id)) throw new Error(`Graph node ID Duplicate: ${node.id}`)
    nodeIds.add(node.id); ids(object(node.properties, 'node.properties').source_refs, 'node.source_refs')
  }
  for (const raw of list(graph.edges, 'graph.edges')) {
    const edge = object(raw, 'graph.edge')
    if (!nodeIds.has(edge.source) || !nodeIds.has(edge.target)) throw new Error('Graph relationship references an unknown node.')
    ids(object(edge.properties, 'edge.properties').source_refs, 'edge.source_refs')
  }
  const report = object(data.report, 'report')
  string(report.title, 'report.title')
  if (typeof report.summary !== 'string') throw new Error('Invalid report summary format.')
  for (const section of list(report.sections, 'report.sections')) {
    string(section.title, 'report.section.title')
    if (typeof section.text !== 'string') throw new Error('Invalid report section format.')
    ids(section.evidence_ids, 'report.evidence_ids')
  }
  ids(report.limitations, 'report.limitations'); list(data.snapshots, 'snapshots')
  return data as LessonPackage
}
function validateCorrectionComparison(data: Obj) {
  if (data.correction_comparison === undefined) return
  const comparison = object(data.correction_comparison, 'correction_comparison')
  const fail = (message: string): never => { throw new Error(`Invalid correction model comparison: ${message}`) }
  if (comparison.schema_version !== '1.0' || comparison.scope !== 'same_lesson_controlled_model_comparison' || comparison.reference_status !== 'no_human_verbatim_reference') fail('Version, Comparison scope or human reference status does not match.')
  if (comparison.source_video_sha256 !== data.lesson.source_video.sha256) fail('Video source does not match the current lesson.')
  if (typeof comparison.evaluated_at !== 'string' || !Number.isFinite(Date.parse(comparison.evaluated_at))) fail('Missing valid evaluation timestamp.')
  ids(comparison.limitations, 'correction_comparison.limitations')
  const transcript = new Map<string, Obj>(data.transcript.map((row: Obj) => [row.id, row]))
  const runIds = new Set<string>()
  for (const raw of list(comparison.runs, 'correction_comparison.runs')) {
    const run = object(raw, 'comparison.run')
    if (!['qwen', 'cec4'].includes(run.id) || runIds.has(run.id)) fail('Missing or duplicate model ID.')
    runIds.add(run.id)
    string(run.model_label, 'comparison.model_label'); string(run.model_sha256, 'comparison.model_sha256')
    for (const field of ['accepted_edit_count', 'model_proposal_count', 'guard_rejected_count', 'term_presence_count', 'terminology_group_count', 'model_accepted_edit_count', 'source_rule_accepted_count', 'additional_changed_spans_vs_rules']) {
      if (run[field] !== undefined && run[field] !== null && (!Number.isInteger(run[field]) || run[field] < 0)) fail(`${field} must be a nonnegative integer or unknown.`)
    }
    for (const field of ['status', 'native_status']) if (run[field] !== undefined) string(run[field], `comparison.${field}`)
    if (run.memory !== undefined && run.memory !== null) {
      const memory = object(run.memory, 'comparison.memory')
      if (memory.unit !== 'MiB' || memory.scope !== 'whole_gpu_sampled') fail('GPU memory unit or measurement scope does not match.')
      for (const field of ['baseline_mib', 'loaded_mib', 'peak_mib', 'sample_interval_ms']) {
        if (memory[field] !== undefined && memory[field] !== null && (typeof memory[field] !== 'number' || !Number.isFinite(memory[field]) || memory[field] < 0 || (field === 'sample_interval_ms' && memory[field] === 0))) fail(`${field} Invalid measurement.`)
      }
    }
    // Comparison TIDs deliberately stay local to each run: they are not new evidence.
    for (const field of ['segments', 'native_segments']) {
      if (field === 'native_segments' && run[field] === undefined) continue
      const seen = new Set<string>()
      for (const raw of list(run[field], `comparison.${field}`)) {
        const row = object(raw, 'comparison.segment'), source = transcript.get(row.transcript_id)
        if (!source || seen.has(row.transcript_id) || row.original_text !== source.text || row.start_sec !== source.start_sec || row.end_sec !== source.end_sec) fail('Segment, Original text or times do not match the current raw transcript.')
        if (typeof row.text !== 'string') fail('Segment is missing model output text.')
        list(row.uncertainties, 'comparison.uncertainties'); seen.add(row.transcript_id)
      }
    }
  }
  if (!runIds.size) fail('Missing model run records.')
}
function validateProofreading(data: Obj, duration: number, knownIds: Set<string>) {
  if (data.transcript_proofreading === undefined) {
    if (data.proofread_evidence_blocks?.length) throw new Error('Correction evidence is missing its source record.')
    return
  }
  const proof = object(data.transcript_proofreading, 'transcript_proofreading')
  string(proof.schema_version, 'proofreading.schema_version'); string(proof.status, 'proofreading.status')
  string(proof.method, 'proofreading.method'); string(proof.model, 'proofreading.model')
  if (proof.human_verified !== false || proof.source_video_sha256 !== data.lesson.source_video.sha256) throw new Error('Machine correction verification status or video source does not match.')
  string(proof.source_asr_sha256, 'proofreading.source_asr_sha256')
  if (proof.counts !== undefined) object(proof.counts, 'proofreading.counts')
  object(proof.provenance, 'proofreading.provenance')
  if (proof.provenance.model_label !== undefined && (typeof proof.provenance.model_label !== 'string' || !proof.provenance.model_label.trim() || proof.provenance.model_label.length > 160 || /[\u0000-\u001f\u007f]/.test(proof.provenance.model_label))) throw new Error('Invalid correction model display label.')
  const transcript = new Map<string, Obj>(data.transcript.map((row: Obj) => [row.id, row]))
  for (const raw of list(proof.alternate_evidence || [], 'proofreading.alternate_evidence')) {
    const row = object(raw, 'alternate_asr'); interval(row, duration, 'alternate_asr')
    if (knownIds.has(row.id)) throw new Error(`Evidence ID Duplicate: ${row.id}`)
    if (row.kind !== 'alternate_asr' || typeof row.text !== 'string') throw new Error('Invalid auxiliary ASR source format.')
    string(row.original_id, 'alternate_asr.original_id'); string(row.variant, 'alternate_asr.variant')
    knownIds.add(row.id)
  }
  const sources = new Set([...knownIds, ...(data.courseware_evidence || []).map((row: Obj) => row.id)])
  const corrected = new Map<string, Obj>()
  const usedTranscript = new Set<string>()
  for (const raw of list(proof.segments, 'proofreading.segments')) {
    const row = object(raw, 'proofreading.segment'); interval(row, duration, 'proofreading.segment')
    if (knownIds.has(row.id)) throw new Error(`Evidence ID Duplicate: ${row.id}`)
    const original = transcript.get(row.transcript_id)
    if (!original || usedTranscript.has(row.transcript_id) || row.original_text !== original.text || row.start_sec !== original.start_sec || row.end_sec !== original.end_sec) throw new Error(`Corrected segment does not match the original transcript: ${row.id}`)
    if (typeof row.text !== 'string') throw new Error(`Corrected segment is missing text: ${row.id}`)
    const refs = ids(row.source_refs, 'proofreading.source_refs')
    if (!refs.includes(original.id) || refs.some(id => !sources.has(id))) throw new Error(`Invalid corrected segment source: ${row.id}`)
    let reconstructed = original.text as string
    for (const raw of list(row.edits, 'proofreading.edits')) {
      const edit = object(raw, 'proofreading.edit'); string(edit.original, 'edit.original'); string(edit.replacement, 'edit.replacement'); string(edit.reason, 'edit.reason')
      if (ids(edit.evidence_ids, 'edit.evidence_ids').some(id => !sources.has(id))) throw new Error('Correction edit references unknown evidence.')
      if (!reconstructed.includes(edit.original)) throw new Error('Correction edit does not match the original text.')
      reconstructed = reconstructed.replace(edit.original, edit.replacement)
    }
    if (reconstructed !== row.text) throw new Error('Corrected text does not match the recorded edits.')
    list(row.uncertainties, 'proofreading.uncertainties')
    if (object(row.quality, 'proofreading.quality').status !== 'automatically_proofread_unreviewed') throw new Error('Machine corrections must remain marked as unverified.')
    knownIds.add(row.id); corrected.set(row.id, row); usedTranscript.add(row.transcript_id)
  }
  for (const raw of list(data.proofread_evidence_blocks || [], 'proofread_evidence_blocks')) {
    const row = object(raw, 'proofread_block'); interval(row, duration, 'proofread_block')
    if (knownIds.has(row.id)) throw new Error(`Evidence ID Duplicate: ${row.id}`)
    if (row.kind !== 'transcript_proofread') throw new Error('Invalid corrected retrieval block kind.')
    string(row.text, 'proofread_block.text')
    const refs = ids(row.source_refs, 'proofread_block.source_refs')
    const segments = ids(row.proofreading_ids, 'proofread_block.proofreading_ids').map(id => corrected.get(id))
    if (!segments.length || segments.some(row => !row) || refs.some(id => !sources.has(id) && !corrected.has(id))) throw new Error('Invalid corrected retrieval block source.')
    const present = segments as Obj[]
    if (Math.abs(row.start_sec - Math.min(...present.map(s => s.start_sec))) > .01 || Math.abs(row.end_sec - Math.max(...present.map(s => s.end_sec))) > .01) throw new Error('Corrected retrieval block times do not match their sources.')
    if (row.text !== present.map(segment => segment.text).join(' ')) throw new Error('Corrected retrieval block text does not match its sources.')
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
  for (const segment of data.alignment) entries.push({ ...segment, text: segment.transcript_ids.map(id => transcript.get(id) || '').filter(Boolean).join(' ') || 'Lesson alignment segment' })
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
  if (result.warnings !== undefined && (!Array.isArray(result.warnings) || result.warnings.length > 8 || result.warnings.some((warning: unknown) => typeof warning !== 'string' || warning.length > 500))) throw new Error('AI Invalid response source verification notice format.')
  if (!['local_llm', 'remote_llm'].includes(result.source)) throw new Error('AI Response does not declare the actual model source.')
  const index = evidenceIndex(data)
  const blockIds = new Set((data.evidence_blocks || []).map(block => block.id))
  const strictIds = new Set([...(data.courseware_evidence || []), ...(data.revised_evidence_blocks || []), ...(data.transcript_proofreading?.alternate_evidence || []), ...(data.transcript_proofreading?.segments || []), ...(data.proofread_evidence_blocks || []), ...(data.teacher_transcript || []), ...(data.teacher_evidence_blocks || []), ...(data.activity_analysis?.evidence || [])].map(row => row.id))
  for (const id of ids(result.evidence_ids, 'evidence_ids')) if (!index.has(id)) throw new Error(`AI Response contains evidence from another lesson: ${id}`)
  for (const raw of list(result.citations, 'citations')) {
    const citation = object(raw, 'citation'); interval(citation, data.lesson.duration_sec, 'citation')
    string(citation.kind, 'citation.kind')
    if (typeof citation.text !== 'string') throw new Error('AI Citation text is missing.')
    const source = index.get(citation.id)
    if (!source) throw new Error(`AI Citation does not belong to this lesson: ${citation.id}`)
    const tolerance = blockIds.has(citation.id) || strictIds.has(citation.id) ? .01 : 1
    if (Math.abs(citation.start_sec - source.start_sec) > tolerance || Math.abs(citation.end_sec - source.end_sec) > tolerance) throw new Error(`AI Citation times do not match the original evidence: ${citation.id}`)
    if (blockIds.has(citation.id) && (citation.kind !== 'transcript' || citation.text !== source.text)) throw new Error(`AI Retrieval block citation text or kind does not match the original evidence: ${citation.id}`)
    if (strictIds.has(citation.id) && (citation.kind !== source.kind || citation.text !== source.text)) throw new Error(`AI Additional citation text or kind does not match the original evidence: ${citation.id}`)
  }
  return result as ReviewChatResponse
}
export function buildReviewMarkdown(data: LessonPackage): string {
  const { lesson, report } = data
  const citationLines = (refs: string[]) => resolveEvidence(data, refs).map(e => `- [${e.id}] ${formatReviewTime(e.start_sec)}–${formatReviewTime(e.end_sec)} ${e.text}`).join('\n')
  return [
    `# ${report.title || lesson.title}`, `Lesson: ${lesson.title}(${formatReviewTime(lesson.duration_sec)})`,
    `source: ${lesson.source_video.filename}\nSHA-256: ${lesson.source_video.sha256}`, report.summary,
    ...report.sections.map(section => `## ${section.title}\n\n${section.text}\n\n${citationLines(section.evidence_ids)}`),
    ...activityMarkdown(data),
    ...(data.transcript_proofreading ? [`## Machine-corrected transcript(Awaiting human verification)\n\n${data.transcript_proofreading.segments.map(row => `- [${row.id}] ${formatReviewTime(row.start_sec)}–${formatReviewTime(row.end_sec)} ${row.text}\n  Original transcript [${row.transcript_id}]: ${row.original_text}\n  source: ${row.source_refs.join(', ')}`).join('\n')}`] : []),
    ...(data.courseware ? [`## Uploaded courseware provenance\n\n${data.courseware.filename}\nSHA-256: ${data.courseware.sha256}\n\n${data.courseware.matches.map(m => `- ${formatReviewTime(m.start_sec)}–${formatReviewTime(m.end_sec)} ${m.status} ${m.page_id || 'Unmatched'}`).join('\n')}`, `## Courseware-assisted revisions(Original transcript preserved)\n\n${(data.transcript_revisions || []).map(r => `- [${r.id}] ${formatReviewTime(r.start_sec)} Original text: ${r.original_text}\n  Suggestion: ${r.corrected_text}\n  Source: ${r.edits.flatMap(e => e.page_ids).join(', ')}`).join('\n') || 'No eligible revisions.'}`] : []),
    `## Limitations\n\n${report.limitations.map(line => `- ${line}`).join('\n')}`,
    `## Generation method\n\n\`\`\`json\n${JSON.stringify(report.generated_by, null, 2)}\n\`\`\``,
    `## Processing provenance\n\n\`\`\`json\n${JSON.stringify(lesson.provenance, null, 2)}\n\`\`\``,
    'Machine transcript, OCR and automatic graph links require checking against the original video; Behavioral cues cannot establish knowledge mastery or teaching effectiveness.',
  ].join('\n\n')
}
