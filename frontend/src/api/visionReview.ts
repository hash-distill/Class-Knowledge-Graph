import { reviewRequest, ReviewServiceError } from './reviewService'
import type { VisualAnnotation, VisualReviewDocument, VisualReviewHistory } from '../types/visionReview'

const base = (lesson: string) => `/lessons/${encodeURIComponent(lesson)}/vision-review`
const framePath = (lesson: string, frame: string) => `${base(lesson)}/frames/${encodeURIComponent(frame)}`
const post = (body: unknown) => ({ method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) })
function document(value: unknown, lesson: string, frame: string): VisualReviewDocument {
  const row = value as VisualReviewDocument
  if (!row || row.lesson_id !== lesson || row.frame_id !== frame || !Number.isInteger(row.version) || row.version < 0 ||
    !/^[a-f0-9]{64}$/.test(row.basis_fingerprint) || !Array.isArray(row.annotations) || !(row.width > 0 && row.height > 0)) {
    throw new Error('复核记录来源或格式无效，请重新加载')
  }
  return row
}
export const visionReview = {
  async get(lesson: string, frame: string) { return document(await reviewRequest(framePath(lesson, frame)), lesson, frame) },
  async save(doc: VisualReviewDocument, annotations: VisualAnnotation[]) {
    return document(await reviewRequest(framePath(doc.lesson_id, doc.frame_id), post({ expected_version: doc.version,
      basis_fingerprint: doc.basis_fingerprint, annotations })), doc.lesson_id, doc.frame_id)
  },
  async history(lesson: string, frame: string) {
    return (await reviewRequest(`${framePath(lesson, frame)}/history`) as { history: VisualReviewHistory[] }).history
  },
  async restore(doc: VisualReviewDocument, version: number) {
    return document(await reviewRequest(`${framePath(doc.lesson_id, doc.frame_id)}/restore`, post({ expected_version: doc.version,
      basis_fingerprint: doc.basis_fingerprint, restore_version: version })), doc.lesson_id, doc.frame_id)
  },
  async export(lesson: string) {
    const controller = new AbortController(), timer = setTimeout(() => controller.abort(), 60000)
    try {
      const response = await fetch(`/api/review${base(lesson)}/export`, { signal: controller.signal })
      if (!response.ok) {
        const data = await response.json()
        throw new ReviewServiceError(data?.error?.message || '样本导出失败', response.status, data?.error?.code || '')
      }
      const url = URL.createObjectURL(await response.blob()), link = documentElement('a')
      link.href = url; link.download = `vision-review-${lesson}.zip`; link.click()
      setTimeout(() => URL.revokeObjectURL(url), 30000)
    } finally { clearTimeout(timer) }
  },
}
function documentElement(tag: 'a') { return window.document.createElement(tag) }
