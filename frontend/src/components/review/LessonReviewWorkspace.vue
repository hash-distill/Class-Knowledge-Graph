<script setup lang="ts">
import { computed, nextTick, onMounted, onUnmounted, ref, watch } from 'vue'
import { headingText, reviewTerm } from '@/utils/reviewPresentation'
import { workbenchUrl, type WorkbenchView } from '@/utils/workbenchNavigation'
import { useLessonReviewStore } from '@/stores/lessonReview'
import AppIcon from '@/components/common/AppIcon.vue'
import LessonPlayer from './LessonPlayer.vue'
import LessonOverview from './LessonOverview.vue'
import LessonEvidenceWorkspace from './LessonEvidenceWorkspace.vue'
import LessonGraph from './LessonGraph.vue'
import LessonAssistant from './LessonAssistant.vue'
import LessonReport from './LessonReport.vue'
import LessonBehavior from './LessonBehavior.vue'
import TeacherWorkspace from './TeacherWorkspace.vue'

const store = useLessonReviewStore()
const player = ref<HTMLElement | null>(null)
const assistantTrigger = ref<HTMLButtonElement | null>(null)
const assistantClose = ref<HTMLButtonElement | null>(null)
const navItems: { id: WorkbenchView; title: string; icon: string; description: string }[] = [
  { id: 'review', title: 'Lesson overview', icon: 'grid', description: 'Replay the original lesson and review student behavior alongside its audio.' },
  { id: 'evidence', title: 'Audio and courseware', icon: 'book', description: 'Replay the audio segment by segment and check video frames against uploaded courseware.' },
  { id: 'behavior', title: 'Student behavior', icon: 'chart', description: 'Observe behavioral cues from visible people while keeping unknown observations explicit.' },
  { id: 'knowledge', title: 'Knowledge graph', icon: 'graph', description: 'Explore lesson concepts and return to their original supporting evidence.' },
  { id: 'report', title: 'Review report', icon: 'film', description: 'Review the complete lesson record and export it with its sources and limitations.' },
  { id: 'teaching', title: 'Teaching improvement', icon: 'book', description: 'Verify lesson records, plan improvements, and save follow-up assessments and actual feedback.' },
]
const currentNav = computed(() => navItems.find(item => item.id === store.activeView)!)
const activeComponent = computed(() => ({ review: LessonOverview, evidence: LessonEvidenceWorkspace, behavior: LessonBehavior, knowledge: LessonGraph, report: LessonReport, teaching: TeacherWorkspace })[store.activeView])
const momentText = computed(() => store.currentDisplayTranscripts.map(row => row.text).join(' '))
const momentVerified = computed(() => store.currentDisplayTranscripts.length > 0 && store.currentDisplayTranscripts.every(row => row.teacher?.human_verified))
function openMoment() { store.evidenceView = 'content'; store.setView('evidence') }
function href(view: WorkbenchView) { const query = new URLSearchParams({ view }); if (store.data) query.set('lesson', store.data.lesson.id); return `?${query}` }
async function chooseLesson(event: Event) {
  await store.selectLesson((event.target as HTMLSelectElement).value)
  if (store.data) window.history.pushState({}, '', workbenchUrl(window.location.href, store.activeView, store.data.lesson.id))
}
async function toggleAssistant() {
  store.assistantOpen = !store.assistantOpen
  await nextTick()
  if (store.assistantOpen) assistantClose.value?.focus()
  else assistantTrigger.value?.focus()
}
async function showCitation() {
  store.assistantOpen = false
  await nextTick()
  player.value?.scrollIntoView({ block: 'nearest', behavior: 'auto' })
  player.value?.focus({ preventScroll: true })
}
function onKey(event: KeyboardEvent) { if (event.key === 'Escape' && store.assistantOpen) void toggleAssistant() }
function onRoute() { void store.restoreRoute() }
watch(() => store.activeView, () => { if (typeof window !== 'undefined') window.scrollTo({ top: 0, behavior: 'auto' }) })
onMounted(async () => {
  window.addEventListener('popstate', onRoute)
  window.addEventListener('keydown', onKey)
  await store.loadLessons()
  if (store.data) window.history.replaceState({}, '', workbenchUrl(window.location.href, store.activeView, store.data.lesson.id))
})
onUnmounted(() => { window.removeEventListener('popstate', onRoute); window.removeEventListener('keydown', onKey) })
</script>

<template>
  <div class="review-workbench" :data-active-view="store.activeView">
    <aside class="review-sidebar">
      <a :href="href('review')" class="review-brand" @click.prevent="store.setView('review')" aria-label="Jingxi Insight lesson overview"><span><AppIcon name="book" :size="25" /></span><div><strong>Jingxi Insight</strong><small>Lesson observation and review</small></div></a>
      <div class="review-space"><b>T</b><div><strong>Teaching research space</strong><small>Recorded lesson · Whole-lesson review</small></div></div>
      <p class="review-nav-label">Workspace <span>Lesson evidence</span></p>
      <nav aria-label="Workspace features"><a v-for="item in navItems" :key="item.id" :href="href(item.id)" :data-view="item.id" :class="{ active: store.activeView === item.id }" :aria-label="item.title" :aria-current="store.activeView === item.id ? 'page' : undefined" @click.prevent="store.setView(item.id)"><AppIcon :name="item.icon" :size="18" /><span>{{ item.title }}</span></a></nav>
      <div class="review-quote"><span>“</span><p>Turn the details of each lesson<br />into insights for teaching improvement.</p><small>Observe · Understand · Review</small></div>
      <div class="review-profile"><b>T</b><div>Teacher workspace<small>Local teaching space</small></div><i></i></div>
    </aside>
    <div class="review-body">
      <header class="review-header"><div>Teaching workspace <AppIcon name="chevron" :size="12" /><strong>{{ currentNav.title }}</strong></div><button ref="assistantTrigger" class="assistant-trigger" :class="{ active: store.assistantOpen }" :aria-expanded="store.assistantOpen" aria-controls="review-assistant-drawer" @click="toggleAssistant"><AppIcon name="spark" :size="16" /><span>Review assistant</span><i v-if="store.isThinking"></i></button></header>
      <main class="review-main">
        <div class="review-page-heading"><div><p class="eyebrow"><span class="heading-orbit" aria-hidden="true"></span>Observe lessons · Connect evidence · Reflect on teaching</p><h1>{{ headingText(currentNav.title) }}</h1><p>{{ currentNav.description }}</p></div><button class="btn refresh-lessons" :disabled="store.loading" @click="store.loadLessons"><AppIcon name="reset" :size="14" />Refresh lessons</button></div>
        <section class="review-source-bar" aria-label="Lesson source"><span class="tag">Recorded lesson</span><label for="lesson-select">Current lesson</label><select v-if="store.lessons.length" id="lesson-select" :value="store.data?.lesson.id || ''" :disabled="store.loading" @change="chooseLesson"><option v-for="lesson in store.lessons" :key="lesson.id" :value="lesson.id">{{ lesson.title }}</option></select><span v-else>No lesson loaded</span><small v-if="store.data">Original video · {{ store.durationText }} · {{ reviewTerm(store.data.lesson.status) }}</small></section>
        <div v-if="store.error" class="review-error" role="alert"><div><strong>Unable to load the lesson</strong><p>{{ store.error }}</p><small v-if="store.data">The last successfully loaded lesson remains below.</small></div><button class="btn" :disabled="store.loading" @click="store.loadLessons">Retry connection</button></div>
        <div v-if="store.loading" class="review-loading" role="status">Loading lesson records and evidence…</div>
        <section v-if="!store.data && !store.loading && !store.error" class="panel review-empty"><AppIcon name="book" :size="42" /><h2>No recorded lessons are ready for review</h2><p>Audio, courseware, and concept evidence will appear here once processing finishes.</p><button class="btn" @click="store.loadLessons">Reload lessons</button></section>
        <template v-if="store.data">
          <div class="workspace-stage" :class="{ 'is-overview': store.activeView === 'review' }">
            <!-- One stable parent and component: feature navigation never replaces the video. -->
            <div ref="player" class="shared-player" data-testid="shared-player" tabindex="-1">
              <LessonPlayer :compact="store.activeView !== 'review'" />
              <div v-if="store.activeView !== 'review'" class="rail-context">
                <section class="rail-clock" aria-label="Current source video time"><div><span class="eyebrow">This moment in the lesson</span><strong class="mono">{{ store.timeText }}</strong></div><span class="tag"><i :class="{ playing: store.isPlaying }"></i>{{ store.isPlaying ? 'Playing' : 'Reviewing' }}</span><div class="rail-progress" aria-hidden="true"><i :style="{ width: `${Math.min(100, store.currentTime / Math.max(1, store.duration) * 100)}%` }"></i></div></section>
                <section class="rail-excerpt"><div><h2>Audio evidence</h2><button @click="openMoment">View segments <AppIcon name="arrow" :size="13" /></button></div><p>{{ momentText || 'No transcript segment at this time. Play the original video to check.' }}</p><small>{{ momentText ? (momentVerified ? 'Teacher verified' : 'Machine context · Unverified') : 'Gaps remain unknown' }}</small></section>
                <div class="rail-shortcuts" aria-label="Lesson workspace shortcuts"><button :class="{ active: store.activeView === 'evidence' }" @click="openMoment"><AppIcon name="book" :size="17" /><span>Audio evidence</span></button><button :class="{ active: store.activeView === 'knowledge' }" @click="store.setView('knowledge')"><AppIcon name="graph" :size="17" /><span>Concept links</span></button><button :class="{ active: store.activeView === 'teaching' }" @click="store.setView('teaching')"><AppIcon name="check" :size="17" /><span>Improvement records</span></button></div>
              </div>
              <p v-if="store.activeView !== 'review'" class="shared-player-note"><AppIcon name="clock" :size="14" /><span>Shared source video timeline<small>Continue this lesson across workspaces</small></span><AppIcon name="check" :size="14" /></p>
            </div>
            <div class="active-feature" :data-view-panel="store.activeView"><Transition name="workspace-view" mode="out-in"><KeepAlive><component :is="activeComponent" :key="store.activeView" /></KeepAlive></Transition></div>
          </div>
          <section v-if="store.activeView === 'review'" class="review-lesson-strip" aria-label="Whole-lesson evidence summary"><div><span class="eyebrow">This lesson</span><h2>{{ headingText(store.data.lesson.title) }}</h2><p>Preserve the source video, machine recognition records and evidence sources</p></div><div><span>Audio transcript</span><strong>{{ store.data.transcript.length }}<small>segments</small></strong></div><div><span>Observed slide segments</span><strong>{{ store.observedSlides.length }}<small>segments</small></strong></div><div><span>Behavior samples</span><strong>{{ store.data.activity_analysis ? store.data.activity_analysis.frames.length : 'Not provided' }}<small v-if="store.data.activity_analysis">frames</small></strong></div><div><span>Concepts</span><strong>{{ store.data.graph.nodes.length }}<small>items</small></strong></div></section>
        </template>
        <footer class="review-footer"><span>Jingxi Insight · Teaching research workspace</span><span>Preserve sources to keep every review traceable.</span></footer>
      </main>
    </div>
    <aside v-if="store.assistantOpen" id="review-assistant-drawer" class="assistant-drawer" role="complementary" aria-label="Review assistant"><div class="drawer-bar"><span>Lesson Q&A <small>{{ store.timeText }}</small></span><button ref="assistantClose" class="icon-btn" aria-label="Close review assistant" @click="toggleAssistant"><AppIcon name="close" :size="19" /></button></div><LessonAssistant @citation="showCitation" /></aside>
  </div>
</template>

<style scoped src="./LessonReviewWorkspace.css"></style>
