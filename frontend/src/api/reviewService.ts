import { parseLessonList, parseLessonPackage, parseChatResponse } from '../utils/lessonReview'
import type { LessonPackage, ReviewChatRequest } from '../types/lessonReview'
import { parseTeacherWorkspace, parseReviewTasks } from '../utils/teacherWorkspace'
export class ReviewServiceError extends Error {
  constructor(message: string, public status: number, public code: string) { super(message); this.name = 'ReviewServiceError' }
}
export async function reviewRequest(path: string, init?: RequestInit): Promise<unknown> {
  const controller = new AbortController()
  const timeout = setTimeout(() => controller.abort(), path === '/chat' ? 180000 : 30000)
  try {
    const response = await fetch(`/api/review${path}`, { ...init, signal: controller.signal })
    let data: any
    try { data = await response.json() } catch { throw new Error(`The service did not return valid JSON (HTTP ${response.status}). Check that the review service is running.`) }
    if (!response.ok) throw new ReviewServiceError(data?.error?.message || `Review service request failed (HTTP ${response.status})`, response.status, data?.error?.code || '')
    return data
  } catch (error) {
    if (error instanceof Error && error.name === 'AbortError') throw new Error('The service timed out. Please try again later.')
    if (error instanceof TypeError) throw new Error('Unable to connect to the backend service. Check the connection and try again.')
    throw error
  } finally { clearTimeout(timeout) }
}
export const reviewService = {
  async lessons() { return parseLessonList(await reviewRequest('/lessons')) },
  async lesson(id: string) { return parseLessonPackage(await reviewRequest(`/lessons/${encodeURIComponent(id)}`)) },
  async workspace(id: string) { return parseTeacherWorkspace(await reviewRequest(`/lessons/${encodeURIComponent(id)}/workspace`), id) },
  async saveWorkspace(id: string, path: string, body: Record<string, unknown>, method = 'POST') {
    return parseTeacherWorkspace(await reviewRequest(`/lessons/${encodeURIComponent(id)}/workspace${path}`, { method, headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) }), id)
  },
  async revertCorrection(id: string, transcriptId: string, version: number) {
    return parseTeacherWorkspace(await reviewRequest(`/lessons/${encodeURIComponent(id)}/workspace/corrections/${encodeURIComponent(transcriptId)}?expected_version=${version}`, { method: 'DELETE' }), id)
  },
  async tasks(id: string) { return parseReviewTasks(await reviewRequest(`/tasks?lesson_id=${encodeURIComponent(id)}`), id) },
  async task(id: string, action: 'cancel' | 'retry') { return reviewRequest(`/tasks/${encodeURIComponent(id)}/${action}`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: '{}' }) },
  async chat(request: ReviewChatRequest, data: LessonPackage) {
    return parseChatResponse(await reviewRequest('/chat', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(request) }), data)
  },
}
