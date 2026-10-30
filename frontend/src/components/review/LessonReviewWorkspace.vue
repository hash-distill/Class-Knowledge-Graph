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
  { id: 'review', title: '课堂总览', icon: 'grid', description: '回看原始课堂，关注学生表现与同刻语音。' },
  { id: 'evidence', title: '语音与课件', icon: 'book', description: '逐段回听原声，核对课堂画面与独立课件。' },
  { id: 'behavior', title: '学生行为', icon: 'chart', description: '观察可见对象的行为线索，保留每一处未知。' },
  { id: 'knowledge', title: '知识图谱', icon: 'graph', description: '从课堂概念出发，回到支撑它的原始证据。' },
  { id: 'report', title: '复盘报告', icon: 'film', description: '梳理整课记录，带着来源与边界导出复盘。' },
  { id: 'teaching', title: '教学改进', icon: 'book', description: '核对课堂记录，制定改进行动，保存课后练习与实际反馈。' },
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
      <a :href="href('review')" class="review-brand" @click.prevent="store.setView('review')" aria-label="精析智评课堂总览"><span><AppIcon name="book" :size="25" /></span><div><strong>精析智评</strong><small>课堂观察与复盘</small></div></a>
      <div class="review-space"><b>教</b><div><strong>教学研究空间</strong><small>真实课堂 · 整课复盘</small></div></div>
      <p class="review-nav-label">工作台 <span>同课证据</span></p>
      <nav aria-label="课堂复盘功能"><a v-for="item in navItems" :key="item.id" :href="href(item.id)" :data-view="item.id" :class="{ active: store.activeView === item.id }" :aria-label="item.title" :aria-current="store.activeView === item.id ? 'page' : undefined" @click.prevent="store.setView(item.id)"><AppIcon :name="item.icon" :size="18" /><span>{{ item.title }}</span></a></nav>
      <div class="review-quote"><span>“</span><p>让课堂中的每一个细节，<br />成为教学改进的线索。</p><small>观察 · 理解 · 复盘</small></div>
      <div class="review-profile"><b>教</b><div>教师工作台<small>本地教学空间</small></div><i></i></div>
    </aside>
    <div class="review-body">
      <header class="review-header"><div>教学工作台 <AppIcon name="chevron" :size="12" /><strong>{{ currentNav.title }}</strong></div><button ref="assistantTrigger" class="assistant-trigger" :class="{ active: store.assistantOpen }" :aria-expanded="store.assistantOpen" aria-controls="review-assistant-drawer" @click="toggleAssistant"><AppIcon name="spark" :size="16" /><span>复盘助手</span><i v-if="store.isThinking"></i></button></header>
      <main class="review-main">
        <div class="review-page-heading"><div><p class="eyebrow"><span class="heading-orbit" aria-hidden="true"></span>观察课堂 · 关联证据 · 回顾教学</p><h1>{{ headingText(currentNav.title) }}</h1><p>{{ currentNav.description }}</p></div><button class="btn refresh-lessons" :disabled="store.loading" @click="store.loadLessons"><AppIcon name="reset" :size="14" />刷新课程</button></div>
        <section class="review-source-bar" aria-label="课堂来源"><span class="tag">{{ store.data?.lesson.provenance.synthetic_demo === true ? '合成演示' : '真实课堂' }}</span><label for="lesson-select">当前课程</label><select v-if="store.lessons.length" id="lesson-select" :value="store.data?.lesson.id || ''" :disabled="store.loading" @change="chooseLesson"><option v-for="lesson in store.lessons" :key="lesson.id" :value="lesson.id">{{ lesson.title }}</option></select><span v-else>尚未加载课程</span><small v-if="store.data">原始录像 · {{ store.durationText }} · {{ reviewTerm(store.data.lesson.status) }}</small></section>
        <div v-if="store.error" class="review-error" role="alert"><div><strong>课程暂时无法加载</strong><p>{{ store.error }}</p><small v-if="store.data">下方保留上一次成功加载的课程。</small></div><button class="btn" :disabled="store.loading" @click="store.loadLessons">重试连接</button></div>
        <div v-if="store.loading" class="review-loading" role="status">正在读取课堂记录与证据…</div>
        <section v-if="!store.data && !store.loading && !store.error" class="panel review-empty"><AppIcon name="book" :size="42" /><h2>还没有可复盘的真实课堂</h2><p>课程处理完成后，语音、课件和知识证据会出现在这里。</p><button class="btn" @click="store.loadLessons">重新加载课程</button></section>
        <template v-if="store.data">
          <div class="workspace-stage" :class="{ 'is-overview': store.activeView === 'review' }">
            <!-- One stable parent and component: feature navigation never replaces the video. -->
            <div ref="player" class="shared-player" data-testid="shared-player" tabindex="-1">
              <LessonPlayer :compact="store.activeView !== 'review'" />
              <div v-if="store.activeView !== 'review'" class="rail-context">
                <section class="rail-clock" aria-label="当前原录像时间"><div><span class="eyebrow">此刻的课堂</span><strong class="mono">{{ store.timeText }}</strong></div><span class="tag"><i :class="{ playing: store.isPlaying }"></i>{{ store.isPlaying ? '播放中' : '回看中' }}</span><div class="rail-progress" aria-hidden="true"><i :style="{ width: `${Math.min(100, store.currentTime / Math.max(1, store.duration) * 100)}%` }"></i></div></section>
                <section class="rail-excerpt"><div><h2>原声线索</h2><button @click="openMoment">查阅段落 <AppIcon name="arrow" :size="13" /></button></div><p>{{ momentText || '当前时刻没有转写片段，可播放原录像核对' }}</p><small>{{ momentText ? (momentVerified ? '教师已核对' : '机器上下文 · 待核验') : '空白区间保留未知' }}</small></section>
                <div class="rail-shortcuts" aria-label="同课工作区快捷入口"><button :class="{ active: store.activeView === 'evidence' }" @click="openMoment"><AppIcon name="book" :size="17" /><span>语音证据</span></button><button :class="{ active: store.activeView === 'knowledge' }" @click="store.setView('knowledge')"><AppIcon name="graph" :size="17" /><span>知识关联</span></button><button :class="{ active: store.activeView === 'teaching' }" @click="store.setView('teaching')"><AppIcon name="check" :size="17" /><span>改进记录</span></button></div>
              </div>
              <p v-if="store.activeView !== 'review'" class="shared-player-note"><AppIcon name="clock" :size="14" /><span>共用原录像时间轴<small>切换工作区，继续当前课堂</small></span><AppIcon name="check" :size="14" /></p>
            </div>
            <div class="active-feature" :data-view-panel="store.activeView"><Transition name="workspace-view" mode="out-in"><KeepAlive><component :is="activeComponent" :key="store.activeView" /></KeepAlive></Transition></div>
          </div>
          <section v-if="store.activeView === 'review'" class="review-lesson-strip" aria-label="整课证据概况"><div><span class="eyebrow">本节课堂</span><h2>{{ headingText(store.data.lesson.title) }}</h2><p>保留原录像、机器识别记录与证据出处</p></div><div><span>原声转写</span><strong>{{ store.data.transcript.length }}<small>段</small></strong></div><div><span>观测课件</span><strong>{{ store.observedSlides.length }}<small>段</small></strong></div><div><span>行为采样</span><strong>{{ store.data.activity_analysis ? store.data.activity_analysis.frames.length : '未提供' }}<small v-if="store.data.activity_analysis">帧</small></strong></div><div><span>知识概念</span><strong>{{ store.data.graph.nodes.length }}<small>个</small></strong></div></section>
        </template>
        <footer class="review-footer"><span>精析智评 · 教学研究工作台</span><span>保留来源，让每一次复盘可追溯。</span></footer>
      </main>
    </div>
    <aside v-if="store.assistantOpen" id="review-assistant-drawer" class="assistant-drawer" role="complementary" aria-label="课堂复盘助手"><div class="drawer-bar"><span>本课问答 <small>{{ store.timeText }}</small></span><button ref="assistantClose" class="icon-btn" aria-label="关闭复盘助手" @click="toggleAssistant"><AppIcon name="close" :size="19" /></button></div><LessonAssistant @citation="showCitation" /></aside>
  </div>
</template>

<style scoped src="./LessonReviewWorkspace.css"></style>
