<script setup lang="ts">
import { useLessonReviewStore } from '@/stores/lessonReview'
import LessonEvidence from './LessonEvidence.vue'
import LessonCourseware from './LessonCourseware.vue'
const store = useLessonReviewStore()
</script>
<template>
  <div class="evidence-workspace">
    <div class="feature-tabs" role="tablist" aria-label="Content evidence workspace"><button id="evidence-content-tab" role="tab" :aria-selected="store.evidenceView === 'content'" aria-controls="evidence-content-panel" :class="{ active: store.evidenceView === 'content' }" @click="store.evidenceView = 'content'">Lesson content<small>Audio and recorded slides</small></button><button id="evidence-courseware-tab" role="tab" :aria-selected="store.evidenceView === 'courseware'" aria-controls="evidence-courseware-panel" :class="{ active: store.evidenceView === 'courseware' }" @click="store.evidenceView = 'courseware'">Uploaded courseware<small>Import, Match and verify</small></button></div>
    <div v-show="store.evidenceView === 'content'" id="evidence-content-panel" role="tabpanel" aria-labelledby="evidence-content-tab"><LessonEvidence /></div>
    <div v-show="store.evidenceView === 'courseware'" id="evidence-courseware-panel" role="tabpanel" aria-labelledby="evidence-courseware-tab"><LessonCourseware /></div>
  </div>
</template>
<style scoped>
.evidence-workspace{display:grid;gap:15px;min-width:0}.feature-tabs{display:flex;border:1px solid var(--line);background:#dee9f9;padding:4px;border-radius:8px;gap:4px}.feature-tabs button{background:none;padding:11px 16px;text-align:left;font-size:12px;color:#5c769b;border-radius:5px;flex:1}.feature-tabs button.active{background:var(--surface);color:var(--navy);box-shadow:0 1px 4px #1333610a}.feature-tabs small{font-size:9px;display:block;margin-top:5px;color:#627da2}.feature-tabs button.active small{color:#587296}@media(max-width:550px){.feature-tabs button{padding:10px 12px}}
</style>
