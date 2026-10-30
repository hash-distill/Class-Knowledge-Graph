/** Display-only heading cleanup. Saved lesson and teacher text remain unchanged. */
export function headingText(value: string): string {
  return value.replace(/[\s。．.!！?？:：;；,，、…·—–-]+$/gu, '')
}

/** Chinese display labels; raw source values remain in the exported package. */
export function transcriptSourceLabel(segment: { source: string; quality?: unknown }): string {
  if (segment.quality && typeof segment.quality === 'object' && 'status' in segment.quality
    && segment.quality.status === 'machine_transcript_ocr_vocabulary_prompted_unreviewed') return '课件词汇提示转写 · 待核验'
  return reviewTerm(segment.source)
}

export function reviewTerm(value: string): string {
  const names: Record<string, string> = {
    pdf_native: '文档原生文字', pptx_native: '演示文稿原生文字', pdf_ocr: '文档图片文字识别', ocr: '画面文字识别',
    complete: '已完成', running: '处理中', failed: '处理失败', unavailable: '未提供',
    transcript: '语音证据', transcript_revision: '转写修订', slide: '课件画面', courseware: '独立课件', behavior: '行为证据',
    transcript_proofread: '机器校对转写', automatically_proofread_unreviewed: '机器校对 · 待核验',
    faster_whisper: '原声自动转写', asr: '原声自动转写', video_ocr: '录像画面文字识别', no_slide_evidence: '未观测到课件投影',
    trained_context: '教学情境模型预测', pretrained_detector: '预训练行为检测', pose_rule: '姿态规则',
    trained_classifier: '逐人状态分类',
    pose_geometry_rule: '姿态几何规则', alternate_asr: '辅助语音识别',
    clip_level_prediction: '整段情境预测', offline_research_only: '仅限离线研究',
    source_extracted: '来源提取，待核验', model_proposed: '模型提议，待核验', teacher_confirmed: '教师已确认',
  }
  return names[value] || (/[\u3400-\u9fff]/.test(value) ? value : '来源记录')
}

export function reviewModelName(value: string): string {
  if (value === 'classroom-local') return '本地课堂复盘模型'
  if (value.startsWith('ts-context-')) return '教学情境分类模型'
  return /[\u3400-\u9fff]/.test(value) ? value : '课堂复盘模型'
}

/** Resolve configured corrector aliases without changing the saved provenance. */
export function proofreadingModelName(value: string, provenance?: Record<string, unknown>): string {
  const label = provenance?.model_label
  if (typeof label === 'string' && label.trim() && label.length <= 160 && !/[\u0000-\u001f\u007f]/.test(label)) return label
  if (value === 'classroom-corrector-qwen35') return 'Qwen3.5-4B'
  if (value === 'classroom-corrector') return 'ChineseErrorCorrector4-4B（CEC4）'
  if (value === 'classroom-local') return 'Qwen3-4B'
  return value
}

/** Model identity comes from the actual ASR provenance, never the proofreading alias. */
export function lessonAsrModelName(provenance: Record<string, unknown>): string {
  const asr = provenance.asr
  if (!asr || typeof asr !== 'object' || Array.isArray(asr)) return '识别模型未提供'
  const row = asr as Record<string, unknown>, config = row.config as Record<string, unknown> | undefined
  for (const candidate of [row.model_label, config?.model_id, row.model_id, row.model]) {
    if (typeof candidate !== 'string' || !candidate.trim() || candidate.length > 240 || /[\u0000-\u001f\u007f]/.test(candidate)) continue
    if (/Qwen3-ASR-1\.7B/i.test(candidate)) return 'Qwen3-ASR-1.7B'
    if (/faster-whisper-small/i.test(candidate)) return 'faster-whisper-small'
    return candidate
  }
  return '识别模型未提供'
}

export function readableChineseText(value: string): string {
  return value.split('\n').filter(line => (line.match(/[\u3400-\u9fff]/g) || []).length >= 4 && !/[A-Za-z]{2,}/.test(line)).join('\n').trim()
}

export function reviewLimitation(value: string): string {
  return value.replace(/ASR\/OCR/g, '语音和画面文字').replace(/uncertainty_sec/g, '边界不确定度')
    .replace(/known_people与coverage/g, '规则分数可用对象数与覆盖率').replace(/ARIC/g, '辅助行为数据')
}
