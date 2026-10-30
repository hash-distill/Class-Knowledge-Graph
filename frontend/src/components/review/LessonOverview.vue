<script setup lang="ts">
import { computed, ref } from 'vue'
import { useLessonReviewStore } from '@/stores/lessonReview'
import { transcriptSourceLabel, proofreadingModelName } from '@/utils/reviewPresentation'
import AppIcon from '@/components/common/AppIcon.vue'
import LessonStudentState from './LessonStudentState.vue'

const store = useLessonReviewStore()
const compareRaw = ref(false)
const frame = computed(() => store.currentActivityFrame)
function openEvidence() { store.evidenceView = 'content'; store.setView('evidence') }
</script>

<template>
  <section class="lesson-overview" aria-label="Student behavior and synchronized audio" data-testid="lesson-overview">
    <div class="overview-heading"><div><p class="eyebrow">Same lesson · Same time</p><h2>Focus on student behavior</h2></div><span class="mono">{{ store.timeText }}</span></div>
    <article class="moment-card behavior-card" data-testid="overview-behavior">
      <div class="moment-title"><span><AppIcon name="chart" :size="15" />Current student behavior<small class="student-count">{{ frame ? `${frame.detected_people} detected people` : 'Awaiting observation' }}</small></span><button @click="store.setView('behavior')">Analysis details <span aria-hidden="true">↗</span></button></div>
      <div class="overview-students"><LessonStudentState :frame="frame" compact /></div>
    </article>
    <article class="moment-card speech-card" data-testid="overview-speech">
      <div class="moment-title"><span><AppIcon name="book" :size="15" />{{ store.data?.transcript_proofreading ? 'Corrected transcript' : 'Audio transcript' }}</span><button @click="openEvidence">View segments <span aria-hidden="true">↗</span></button></div>
      <p v-if="store.currentDisplayTranscripts.length" class="moment-speech">{{ store.currentDisplayTranscripts.map(row => row.text).join(' ') }}</p>
      <p v-else class="moment-empty">No transcript is recorded at this time. Check against the original audio.</p>
      <button v-if="!store.transcripts.length && store.data?.transcript[0]" class="moment-jump" @click="store.seek(store.data.transcript[0].start_sec)">Go to a transcript segment in this lesson <span aria-hidden="true">→</span></button>
      <div v-if="store.data?.teacher_transcript && store.transcripts.length" class="overview-proofread" data-testid="transcript-teacher-default"><span><small v-for="row in store.currentDisplayTranscripts" :key="row.id">{{ row.id }} · {{ row.teacher?.human_verified ? 'Teacher verified' : 'Machine context · Unverified' }}</small></span><button data-testid="transcript-raw-toggle" :aria-pressed="compareRaw" @click="compareRaw = !compareRaw">{{ compareRaw ? 'Hide original text' : 'Compare original text' }}</button><p v-if="compareRaw" data-testid="transcript-raw-comparison">{{ store.transcripts.map(row => row.text).join(' ') }}</p></div>
      <div v-else-if="store.data?.transcript_proofreading && store.transcripts.length" class="overview-proofread" data-testid="transcript-proofread-default"><span>Machine correction · Unverified<small data-testid="active-correction-model">{{ proofreadingModelName(store.data.transcript_proofreading.model, store.data.transcript_proofreading.provenance) }}</small></span><button data-testid="transcript-raw-toggle" :aria-pressed="compareRaw" @click="compareRaw = !compareRaw">{{ compareRaw ? 'Hide original text' : 'Compare original text' }}</button><p v-if="compareRaw" data-testid="transcript-raw-comparison">{{ store.transcripts.map(row => row.text).join(' ') }}</p></div>
      <small v-else-if="store.transcripts.length">{{ store.transcripts.map(row => `${row.id} · ${transcriptSourceLabel(row)}`).join(' / ') }}</small>
    </article>
    <p class="overview-origin"><i></i>All records align to the source video clock; Gaps remain unknown.</p>
  </section>
</template>

<style scoped>
.lesson-overview{display:grid;gap:12px;min-width:0}
.overview-heading{display:flex;justify-content:space-between;align-items:center;gap:12px;padding:3px 2px}
.overview-heading h2{font-size:17px;font-weight:600;color:#2d538a;margin-top:8px}
.overview-heading>.mono{font-size:20px;color:#3475db}
.overview-heading .eyebrow{font-size:9px;letter-spacing:.08em}
.moment-card{background:var(--surface);border:1px solid var(--line);border-radius:10px;overflow:hidden;transition:border-color .22s,box-shadow .22s;animation:panel-reveal .45s var(--ease-out) both}
.moment-card:hover{border-color:#b8cff3;box-shadow:0 4px 16px #245edb0b}
.behavior-card{animation-delay:.03s}
.speech-card{animation-delay:.08s}
.moment-title{display:flex;flex-wrap:wrap;align-items:center;justify-content:space-between;gap:8px 10px;padding:14px 16px 10px;flex-shrink:0}
.moment-title>span{display:flex;flex-wrap:wrap;gap:7px;align-items:center;min-width:0;font-size:11px;font-weight:500;color:#3e6aab}
.moment-title button{margin-left:auto;padding:0;background:none;color:#6488bd;font-size:10px;white-space:nowrap}
.moment-title button:hover{color:var(--navy)}
.overview-students{min-height:0;padding:0 16px 13px}
.overview-students :deep(.compact .state-layout){flex-direction:column;gap:12px}
.student-count{font-size:9px;color:#7998c1;font-weight:400}
.overview-students :deep(.compact .state-heading){display:none}
.overview-students :deep(.compact .state-visual){display:block}
.overview-students :deep(.compact .state-reading){display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:7px 12px;align-items:start}
.overview-students :deep(.compact .participation-reading){grid-column:1;grid-row:1/span 2}
.overview-students :deep(.compact .participation-reading){flex-wrap:wrap;row-gap:6px;min-width:0}
.overview-students :deep(.compact .action-count){grid-column:2;margin:0;font-size:8px}
.overview-students :deep(.compact .participation-reference){grid-column:2;margin:0;padding:6px 8px;font-size:8px}
.overview-students :deep(.compact .state-coverage){grid-column:1/-1;margin-top:0}
.overview-students :deep(.compact .state-counts){grid-column:1/-1;grid-template-columns:repeat(3,minmax(0,1fr));max-height:none;margin-top:0;padding-top:8px;gap:6px 10px}
.overview-students :deep(.compact .state-counts b){flex-shrink:0}
.overview-students :deep(.compact .image-caption){margin:0;flex-wrap:wrap;row-gap:4px}
.overview-students :deep(.compact .locate-observation){grid-column:1/-1;margin-top:1px}
.overview-students :deep(.compact .state-boundary){margin-top:10px}
.moment-speech{font-size:12px;color:#47678f;line-height:1.9;padding:0 16px;max-height:110px;overflow:auto}
.moment-empty{font-size:11px;color:#667d9d;line-height:1.8;padding:0 16px 12px}
.moment-jump{display:flex;justify-content:space-between;gap:12px;margin:8px 16px 12px;padding:7px 9px;width:calc(100% - 32px);background:#e9f1fc;color:#4f709d;border-radius:4px;font-size:10px;line-height:1.6;flex-shrink:0}
.moment-jump:hover{background:#d8e7fc}
.moment-card>small{font-size:9px;color:#657d9f;display:block;padding:8px 16px 12px;overflow-wrap:anywhere;flex-shrink:0}
.overview-proofread{display:flex;flex-wrap:wrap;justify-content:space-between;gap:8px;font-size:9px;padding:8px 16px 11px;color:#577197;flex-shrink:0}
.overview-proofread>span{min-width:0;overflow-wrap:anywhere}
.overview-proofread>span>small{display:block;font-size:8px;margin-top:3px;line-height:1.7;color:#4e6a91}
.overview-proofread button{font-size:9px;color:#516e96;background:transparent;padding:0}
.overview-proofread>p{flex-basis:100%;border-left:2px solid #c8d4e4;padding-left:9px;color:#60799c;line-height:1.9}
.overview-origin{display:flex;align-items:center;gap:7px;font-size:9px;color:#8098b8;line-height:1.8}
.overview-origin i{height:4px;width:4px;border-radius:50%;background:#6199ee;flex-shrink:0}
@media(min-width:1001px){
  .lesson-overview{height:100%;min-height:0;grid-template-rows:auto minmax(0,1fr) minmax(110px,.24fr) auto}
  .overview-heading{padding:0 1px 3px}
  .overview-heading h2{font-size:16px}
  .moment-card{min-height:0;display:flex;flex-direction:column}
  .overview-students{flex:1;overflow:auto;overscroll-behavior-y:contain;scrollbar-gutter:stable}
  .overview-students :deep(.compact){height:auto;min-height:100%;display:flex;flex-direction:column}
  .overview-students :deep(.compact .candidate-notice),.overview-students :deep(.compact .state-boundary){flex-shrink:0}
  .overview-students :deep(.compact .state-layout){flex:1 0 auto;min-height:min-content;gap:14px}
  .overview-students :deep(.compact .state-visual){flex:1 0 auto;min-height:min-content;display:flex;flex-direction:column}
  .overview-students :deep(.compact .annotated-frame){flex:1;height:0;min-height:110px;aspect-ratio:auto!important}
  .overview-students :deep(.compact .image-caption){flex-shrink:0}
  .overview-students :deep(.compact .state-reading){flex-shrink:0}
  .overview-students :deep(.compact .locate-observation){display:none}
  .speech-card .moment-speech{min-height:0;flex:1;max-height:none}
  .overview-origin{font-size:8px}
}
@media(max-width:1000px){.lesson-overview{height:auto}.moment-speech{max-height:120px}}
@media(max-width:550px){.overview-heading h2{font-size:16px}.overview-students{padding:0 12px 13px}.moment-title{padding:13px 12px 10px}.overview-students :deep(.compact .state-counts){grid-template-columns:repeat(2,minmax(0,1fr))}}
</style>
