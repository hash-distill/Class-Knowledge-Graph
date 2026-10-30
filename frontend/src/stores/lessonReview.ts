import { computed, ref } from 'vue'
import { defineStore } from 'pinia'
import { reviewService } from '@/api/reviewService'
import { activeAt, evidenceIndex, formatReviewTime, isObservedSlide } from '@/utils/lessonReview'
import { parseWorkbenchRoute, workbenchUrl, type WorkbenchView, type BehaviorView, type EvidenceView } from '@/utils/workbenchNavigation'
import type { ChatScope, LessonPackage, LessonSummary, ReviewMessage, TimedEvidence } from '@/types/lessonReview'
import { teacherDisplayTranscripts } from '@/utils/teacherWorkspace'
export const useLessonReviewStore = defineStore('lessonReview', () => {
  const lessons = ref<LessonSummary[]>([])
  const data = ref<LessonPackage | null>(null)
  const loading = ref(false), error = ref(''), currentTime = ref(0), seekVersion = ref(0), isPlaying = ref(false), selectedNodeId = ref('')
  const messages = ref<ReviewMessage[]>([])
  const scope = ref<ChatScope>('lesson')
  const activeView = ref<WorkbenchView>(parseWorkbenchRoute(typeof window === 'undefined' ? '' : window.location.search).view)
  const behaviorView = ref<BehaviorView>('overview'), evidenceView = ref<EvidenceView>('content'), assistantOpen = ref(false)
  let loadVersion = 0, messageId = 0, refreshVersion = 0
  const sources = computed(() => data.value ? evidenceIndex(data.value) : new Map<string, TimedEvidence>())
  const intervalsByNode = computed(() => {
    const index = new Map<string, TimedEvidence[]>()
    if (data.value) for (const node of data.value.graph.nodes) {
      const ids = [...new Set([...node.properties.source_refs, ...data.value.alignment.filter(row => row.concept_ids.includes(node.id)).map(row => row.id)])]
      index.set(node.id, ids.map(id => sources.value.get(id)).filter((row): row is TimedEvidence => Boolean(row)).sort((a,b) => a.start_sec - b.start_sec || a.end_sec - b.end_sec))
    }
    return index
  })
  const duration = computed(() => data.value?.lesson.duration_sec || 0)
  const durationText = computed(() => formatReviewTime(duration.value))
  const timeText = computed(() => formatReviewTime(currentTime.value))
  const transcripts = computed(() => data.value ? activeAt(data.value.transcript, currentTime.value) : [])
  const proofreadByTranscript = computed(() => new Map((data.value?.transcript_proofreading?.segments || []).map(row => [row.transcript_id, row])))
  const displayTranscripts = computed(() => data.value ? teacherDisplayTranscripts(data.value) : [])
  const currentDisplayTranscripts = computed(() => activeAt(displayTranscripts.value, currentTime.value))
  const observedSlides = computed(() => data.value?.slides.filter(isObservedSlide) || [])
  const slides = computed(() => activeAt(observedSlides.value, currentTime.value))
  const currentSlide = computed(() => slides.value[0] || null)
  const currentActivityFrame = computed(() => data.value?.activity_analysis?.frames.find(frame => frame.start_sec <= currentTime.value && currentTime.value < frame.end_sec) || null)
  const currentAlignments = computed(() => data.value ? activeAt(data.value.alignment, currentTime.value) : [])
  const currentConceptIds = computed(() => {
    const ids = new Set(currentAlignments.value.flatMap(a => a.concept_ids))
    for (const [nodeId, intervals] of intervalsByNode.value) if (activeAt(intervals, currentTime.value).length) ids.add(nodeId)
    return [...ids]
  })
  const selectedNode = computed(() => data.value?.graph.nodes.find(n => n.id === selectedNodeId.value) || null)
  const selectedIntervals = computed(() => intervalsByNode.value.get(selectedNodeId.value) || [])
  const isThinking = computed(() => messages.value.some(m => m.status === 'pending'))
  function updateTime(seconds: number) { currentTime.value = Math.max(0, Math.min(duration.value, Number.isFinite(seconds) ? seconds : 0)) }
  function seek(seconds: number) { updateTime(seconds); seekVersion.value++; isPlaying.value = false }
  function jumpToEvidence(id: string) { const evidence = sources.value.get(id); if (evidence) seek(evidence.start_sec) }
  function selectNode(id: string) { selectedNodeId.value = id; const evidence = selectedIntervals.value[0]; if (evidence) seek(evidence.start_sec) }
  function setView(view: WorkbenchView) {
    if (activeView.value === view) return
    activeView.value = view
    if (typeof window !== 'undefined') window.history.pushState({}, '', workbenchUrl(window.location.href, view, data.value?.lesson.id))
  }
  async function restoreRoute() {
    if (typeof window === 'undefined') return
    const route = parseWorkbenchRoute(window.location.search)
    activeView.value = route.view
    if (data.value && (!route.lesson || route.lesson === data.value.lesson.id)) {
      // Returning to the visible recording is a new navigation intent even without a fetch.
      // Invalidate a replacement already in flight before it can overwrite this route.
      loadVersion++; loading.value = false; error.value = ''
    } else if (route.lesson && lessons.value.some(lesson => lesson.id === route.lesson)) await selectLesson(route.lesson)
  }
  async function selectLesson(id: string) {
    const version = ++loadVersion
    loading.value = true; error.value = ''
    try {
      const result = await reviewService.lesson(id)
      if (version !== loadVersion) return
      const sameRecording = data.value?.lesson.id === result.lesson.id && data.value.lesson.video_url === result.lesson.video_url && data.value.lesson.source_video.sha256 === result.lesson.source_video.sha256
      data.value = result
      if (sameRecording) {
        updateTime(currentTime.value)
        if (!result.graph.nodes.some(node => node.id === selectedNodeId.value)) selectedNodeId.value = ''
      } else { currentTime.value = 0; seekVersion.value++; isPlaying.value = false; selectedNodeId.value = ''; messages.value = [] }
    } catch (reason) { if (version === loadVersion) error.value = reason instanceof Error ? reason.message : 'Unable to load the lesson.' }
    finally { if (version === loadVersion) loading.value = false }
  }
  async function loadLessons() {
    const version = ++loadVersion
    loading.value = true; error.value = ''
    try {
      const result = await reviewService.lessons()
      if (version !== loadVersion) return
      lessons.value = result
      if (!result.length) { data.value = null; currentTime.value = 0; messages.value = []; return }
      const requested = typeof window === 'undefined' ? null : parseWorkbenchRoute(window.location.search).lesson
      const chosen = result.find(l => l.id === (requested || data.value?.lesson.id)) || result[0]!
      await selectLesson(chosen.id)
    } catch (reason) { if (version === loadVersion) error.value = reason instanceof Error ? reason.message : 'Unable to load the lesson list.' }
    finally { if (version === loadVersion) loading.value = false }
  }
  async function refreshLesson(id: string): Promise<boolean> {
    if (loading.value || data.value?.lesson.id !== id) return false
    const navigation = loadVersion, request = ++refreshVersion
    const result = await reviewService.lesson(id)
    if (navigation !== loadVersion || request !== refreshVersion || loading.value || data.value?.lesson.id !== id) return false
    const sameRecording = data.value.lesson.video_url === result.lesson.video_url && data.value.lesson.source_video.sha256 === result.lesson.source_video.sha256
    data.value = result
    if (sameRecording) {
      updateTime(currentTime.value)
      if (!result.graph.nodes.some(node => node.id === selectedNodeId.value)) selectedNodeId.value = ''
    } else { currentTime.value = 0; seekVersion.value++; isPlaying.value = false; selectedNodeId.value = ''; messages.value = [] }
    return true
  }
  async function sendMessage(message: ReviewMessage) {
    if (!data.value) return
    const lesson = data.value
    message.status = 'pending'; message.error = undefined
    try {
      const response = await reviewService.chat({ lesson_id: lesson.lesson.id, query: message.query, at_sec: message.at_sec, scope: message.scope }, lesson)
      if (!messages.value.includes(message)) return
      message.response = response; message.status = 'answered'
    } catch (reason) {
      if (!messages.value.includes(message)) return
      message.error = reason instanceof Error ? reason.message : 'Model request failed.'; message.status = 'error'
    }
  }
  async function ask(query: string) {
    if (!query.trim() || !data.value || isThinking.value) return
    messages.value.push({ id: ++messageId, query: query.trim(), at_sec: currentTime.value, scope: scope.value, status: 'pending' })
    await sendMessage(messages.value[messages.value.length - 1]!)
  }
  async function retry(id: number) { const message = messages.value.find(m => m.id === id); if (message && !isThinking.value) await sendMessage(message) }
  function evidence(id: string): TimedEvidence | undefined { return sources.value.get(id) }
  return { lessons, data, loading, error, currentTime, seekVersion, isPlaying, selectedNodeId, messages, scope, activeView, behaviorView, evidenceView, assistantOpen, currentActivityFrame, setView, restoreRoute, duration, durationText, timeText, transcripts, displayTranscripts, currentDisplayTranscripts, proofreadByTranscript, observedSlides, slides, currentSlide, currentAlignments, currentConceptIds, selectedNode, selectedIntervals, isThinking, loadLessons, selectLesson, refreshLesson, updateTime, seek, jumpToEvidence, selectNode, ask, retry, evidence }
})
