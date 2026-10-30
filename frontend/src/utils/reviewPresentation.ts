/** Display-only heading cleanup. Saved lesson and teacher text remain unchanged. */
export function headingText(value: string): string {
  return value.replace(/[\s。．.!！?？:：;；,，、…·—–-]+$/gu, '')
}

/** English display labels; raw source values remain in the exported package. */
export function transcriptSourceLabel(segment: { source: string; quality?: unknown }): string {
  if (segment.quality && typeof segment.quality === 'object' && 'status' in segment.quality
    && segment.quality.status === 'machine_transcript_ocr_vocabulary_prompted_unreviewed') return 'Transcript prompted with courseware vocabulary · Unverified'
  return reviewTerm(segment.source)
}

export function reviewTerm(value: string): string {
  const names: Record<string, string> = {
    pdf_native: 'Native PDF text', pptx_native: 'Native presentation text', pdf_ocr: 'Document OCR', ocr: 'Screen OCR',
    complete: 'Completed', running: 'Processing', failed: 'Failed', unavailable: 'Not provided',
    transcript: 'Audio evidence', transcript_revision: 'Transcript revision', slide: 'Slide image', courseware: 'Uploaded courseware', behavior: 'Activity evidence',
    transcript_proofread: 'Machine-corrected transcript', automatically_proofread_unreviewed: 'Machine correction · Unverified',
    faster_whisper: 'Automatic audio transcript', asr: 'Automatic audio transcript', video_ocr: 'Video screen OCR', no_slide_evidence: 'No projected slides observed',
    trained_context: 'Teaching context model prediction', pretrained_detector: 'Pretrained behavior detection', pose_rule: 'Pose rule',
    trained_classifier: 'Per-person state classification',
    pose_geometry_rule: 'Geometric pose rule', alternate_asr: 'Auxiliary ASR',
    clip_level_prediction: 'Clip-level context prediction', offline_research_only: 'Offline research only',
    source_extracted: 'Source-extracted, unverified', model_proposed: 'Model-proposed, unverified', teacher_confirmed: 'Teacher confirmed',
  }
  return names[value] || (/[\u3400-\u9fff]/.test(value) ? value : 'Source record')
}

export function reviewModelName(value: string): string {
  if (value === 'classroom-local') return 'Local review model'
  if (value.startsWith('ts-context-')) return 'Teaching context classifier'
  return /[\u3400-\u9fff]/.test(value) ? value : 'Review model'
}

/** Resolve configured corrector aliases without changing the saved provenance. */
export function proofreadingModelName(value: string, provenance?: Record<string, unknown>): string {
  const label = provenance?.model_label
  if (typeof label === 'string' && label.trim() && label.length <= 160 && !/[\u0000-\u001f\u007f]/.test(label)) return label
  if (value === 'classroom-corrector-qwen35') return 'Qwen3.5-4B'
  if (value === 'classroom-corrector') return 'ChineseErrorCorrector4-4B (CEC4)'
  if (value === 'classroom-local') return 'Qwen3-4B'
  return value
}

/** Model identity comes from the actual ASR provenance, never the proofreading alias. */
export function lessonAsrModelName(provenance: Record<string, unknown>): string {
  const asr = provenance.asr
  if (!asr || typeof asr !== 'object' || Array.isArray(asr)) return 'ASR model not provided'
  const row = asr as Record<string, unknown>, config = row.config as Record<string, unknown> | undefined
  for (const candidate of [row.model_label, config?.model_id, row.model_id, row.model]) {
    if (typeof candidate !== 'string' || !candidate.trim() || candidate.length > 240 || /[\u0000-\u001f\u007f]/.test(candidate)) continue
    if (/Qwen3-ASR-1\.7B/i.test(candidate)) return 'Qwen3-ASR-1.7B'
    if (/faster-whisper-small/i.test(candidate)) return 'faster-whisper-small'
    return candidate
  }
  return 'ASR model not provided'
}

export function readableChineseText(value: string): string {
  return value.split('\n').filter(line => (line.match(/[\u3400-\u9fff]/g) || []).length >= 4 && !/[A-Za-z]{2,}/.test(line)).join('\n').trim()
}

/** The legacy mixed-language token below is source metadata, not an interface label. */
export function reviewLimitation(value: string): string {
  return value.replace(/ASR\/OCR/g, 'Speech and screen text').replace(/uncertainty_sec/g, 'Boundary uncertainty')
    .replace(/known_people与coverage/g, 'Usable rule-score person count and coverage').replace(/ARIC/g, 'Auxiliary behavior data')
}
