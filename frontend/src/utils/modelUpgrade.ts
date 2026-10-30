import type { ModelUpgradeExperiment } from '../types/lessonReview'

type Obj = Record<string, any>
const fail = (message: string): never => { throw new Error(`Invalid model upgrade experiment: ${message}`) }
function object(value: unknown): Obj {
  if (!value || typeof value !== 'object' || Array.isArray(value)) fail('Missing object record.')
  return value as Obj
}
function list(value: unknown): any[] {
  if (!Array.isArray(value)) fail('Missing list record.')
  return value as any[]
}
function text(value: unknown): string {
  if (typeof value !== 'string' || !value.trim()) fail('Missing nonempty text or source ID.')
  return value as string
}
function sameSegment(row: Obj, source: Obj | undefined): boolean {
  return Boolean(source && row.start_sec === source.start_sec && row.end_sec === source.end_sec && row.text === source.text)
}

/** Run-local transcript IDs never become global evidence or imply one-to-one alignment. */
export function validateModelUpgradeExperiment(data: Obj): void {
  if (data.model_upgrade_experiment === undefined) return
  const experiment = object(data.model_upgrade_experiment)
  if (experiment.schema_version !== '1.0' || experiment.reference_status !== 'no_human_verbatim_reference') fail('Version or human reference status does not match.')
  if (experiment.source_video_sha256 !== data.lesson.source_video.sha256) fail('Video source does not match the current lesson.')
  const asr = object(data.lesson.provenance.asr)
  const audioHash = asr.audio_sha256 || asr.source_audio?.sha256
  if (text(experiment.source_audio_sha256) !== text(audioHash)) fail('Audio source does not match the current lesson.')
  if (typeof experiment.evaluated_at !== 'string' || !Number.isFinite(Date.parse(experiment.evaluated_at))) fail('Missing valid run timestamp.')
  list(experiment.limitations).forEach(text)
  const transcript = new Map<string, Obj>(data.transcript.map((row: Obj) => [row.id, row]))
  const runs = new Map<string, Obj>()
  for (const raw of list(experiment.asr_runs)) {
    const run = object(raw)
    if (!['baseline', 'qwen3_asr'].includes(run.id) || runs.has(run.id)) fail('Missing or duplicate ASR run ID.')
    text(run.model_label); text(run.source_asr_sha256)
    if (run.runtime !== undefined) object(run.runtime)
    const seen = new Set<string>()
    for (const rawSegment of list(run.segments)) {
      const row = object(rawSegment), id = text(row.id)
      if (seen.has(id)) fail('ASR segment in the same run ID Duplicate.')
      seen.add(id); text(row.text)
      if (![row.start_sec, row.end_sec].every(value => typeof value === 'number' && Number.isFinite(value) && value >= 0)
        || row.start_sec > row.end_sec || row.end_sec > data.lesson.duration_sec) fail('ASR segment times fall outside the lesson.')
      if (run.id === 'qwen3_asr' && !sameSegment(row, transcript.get(id))) fail('New ASR segment does not match the current raw transcript.')
    }
    if (!seen.size) fail('Missing actual ASR segments.')
    if (run.id === 'qwen3_asr' && seen.size !== transcript.size) fail('New ASR does not fully match the current raw transcript.')
    runs.set(run.id, run)
  }
  if (!runs.has('baseline') || !runs.has('qwen3_asr')) fail('Missing baseline or new ASR records.')
  const currentAsrHash = text(asr.source_asr_sha256)
  if (runs.get('qwen3_asr')!.source_asr_sha256 !== currentAsrHash) fail('New ASR file source does not match.')
  const correction = object(experiment.correction), proof = object(data.transcript_proofreading)
  text(correction.model_label); text(correction.model_sha256); text(correction.status)
  if (correction.source_asr_sha256 !== currentAsrHash || proof.source_asr_sha256 !== currentAsrHash) fail('Correction input does not match the new ASR file source.')
  if (proof.provenance.model_label !== undefined && correction.model_label !== proof.provenance.model_label) fail('Correction model label does not match the actual correction source.')
  if (proof.provenance.model_sha256 !== undefined && correction.model_sha256 !== proof.provenance.model_sha256) fail('Correction model file does not match the actual correction source.')
  if (correction.status !== proof.status) fail('Correction status does not match the actual correction record.')
  for (const field of ['accepted_edit_count', 'model_accepted_edit_count', 'source_rule_accepted_count']) {
    if (!Number.isInteger(correction[field]) || correction[field] < 0) fail('Correction counts must be nonnegative integers.')
  }
  const acceptedEdits: Obj[] = proof.segments.flatMap((row: Obj) => row.edits)
  const modelAccepted = acceptedEdits.filter(edit => edit.origin === 'model').length
  if (correction.accepted_edit_count !== acceptedEdits.length || correction.model_accepted_edit_count !== modelAccepted
    || correction.source_rule_accepted_count !== acceptedEdits.length - modelAccepted
    || (proof.edit_count !== undefined && proof.edit_count !== correction.accepted_edit_count)) fail('Correction counts do not match the actual accepted edits.')
  const corrected = new Map<string, Obj>(proof.segments.map((row: Obj) => [row.transcript_id, row]))
  const seen = new Set<string>()
  for (const raw of list(correction.segments)) {
    const row = object(raw), id = text(row.transcript_id), source = corrected.get(id)
    if (seen.has(id) || !sameSegment(row, source) || row.original_text !== source?.original_text) fail('Corrected segments do not match the actual correction record.')
    text(row.text)
    if (JSON.stringify(list(row.uncertainties)) !== JSON.stringify(source!.uncertainties)) fail('Unconfirmed correction content does not match the actual record.')
    seen.add(id)
  }
  if (seen.size !== corrected.size) fail('Corrected segments do not fully match the actual correction record.')
}

/** Keep each lane's own boundaries; no positional or transcript-ID zip across ASR runs. */
export function modelUpgradeLanes(experiment: ModelUpgradeExperiment, time: number, all = false) {
  const lanes = [
    ...(['baseline', 'qwen3_asr'] as const).map(id => {
      const run = experiment.asr_runs.find(run => run.id === id)!
      return { id, title: id === 'baseline' ? 'Baseline ASR' : 'New ASR', model: run.model_label, segments: run.segments }
    }),
    { id: 'correction', title: 'Conservative correction', model: experiment.correction.model_label,
      segments: experiment.correction.segments.map(row => ({ ...row, id: row.transcript_id })) },
  ]
  return lanes.map(lane => ({ ...lane, segments: all ? lane.segments : lane.segments.filter(row => row.start_sec <= time && time < row.end_sec) }))
}

/** Millisecond precision for independent predicted boundaries, without minute-floor truncation. */
export function modelUpgradeTime(value: number): string {
  return `${value.toLocaleString('en-US', { maximumFractionDigits: 3, useGrouping: false })} s`
}
