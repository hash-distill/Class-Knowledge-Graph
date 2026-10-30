<script setup lang="ts">
import { headingText, reviewLimitation } from '@/utils/reviewPresentation'
import { computed, ref } from 'vue'
import { engagementExport } from '@/utils/classEngagement'
import { useLessonReviewStore } from '@/stores/lessonReview'
import { buildReviewMarkdown, formatReviewTime } from '@/utils/lessonReview'
import AppIcon from '@/components/common/AppIcon.vue'
const store = useLessonReviewStore()
const exported = ref('')
const engagement = computed(() => store.data?.activity_analysis?.frames.some(frame => frame.visual_reference) ? engagementExport(store.data.activity_analysis.frames) : null)
function download(format: 'json' | 'md') {
  if (!store.data) return
  const content = format === 'json' ? JSON.stringify({ ...store.data, ...(engagement.value ? {
    derived_engagement: { ...engagement.value, source_sha256: store.data.activity_analysis!.source_sha256 },
  } : {}) }, null, 2) : buildReviewMarkdown(store.data)
  const url = URL.createObjectURL(new Blob([content], { type: format === 'json' ? 'application/json' : 'text/markdown;charset=utf-8' }))
  const anchor = document.createElement('a'); anchor.href = url; anchor.download = `${store.data.lesson.id}-整课复盘.${format}`; anchor.click()
  window.setTimeout(() => URL.revokeObjectURL(url), 1000); exported.value = format === 'json' ? '完整数据' : '复盘文稿'
}
</script>
<template>
  <section v-if="store.data" id="report" class="panel lesson-report">
    <div class="panel-heading"><div><h2>把整堂课，留成一份复盘</h2><p>报告、来源与使用边界一并保留</p></div><AppIcon name="book" :size="21" /></div>
    <div class="report-body"><span class="eyebrow">整课复盘</span><h3>{{ headingText(store.data.report.title) }}</h3><p class="report-summary">{{ store.data.report.summary }}</p><section v-if="engagement" class="report-section" data-testid="report-engagement"><h4>学生整体积极度</h4><p>本课平均积极度 {{ engagement.summary.average === null ? '暂无评分' : `${engagement.summary.average.toFixed(1)} / 100` }} · 有效评分 {{ engagement.summary.scored_frames }} / {{ engagement.summary.total_frames }} 帧</p><p>按有效画面时长加权，采用原项目七类行为权重、班级离散修正与连续平滑。低覆盖评分为辅助估计，未知观测不补零。下方保留生成时的原始报告与证据</p></section><section v-for="(section,index) in store.data.report.sections" :key="index" class="report-section"><h4><span>{{ (index + 1).toString().padStart(2, '0') }}</span>{{ headingText(section.title) }}</h4><p>{{ section.text }}</p><div class="report-references"><button v-for="id in section.evidence_ids" :key="id" :disabled="!store.evidence(id)" :title="store.evidence(id)?.text || '来源不可定位'" @click="store.jumpToEvidence(id)">{{ id }}<span v-if="store.evidence(id)"> · {{ formatReviewTime(store.evidence(id)!.start_sec) }}</span></button></div></section><div class="report-limits"><h4>使用边界</h4><ul><li v-for="limit in store.data.report.limitations" :key="limit">{{ reviewLimitation(limit) }}</li></ul></div><details class="report-provenance"><summary>查看生成方式与来源记录</summary><p>原始文件：{{ store.data.lesson.source_video.filename }}</p><p>SHA-256：{{ store.data.lesson.source_video.sha256 }}</p><h4>报告生成方式</h4><pre>{{ JSON.stringify(store.data.report.generated_by, null, 2) }}</pre><h4>课程处理来源</h4><pre>{{ JSON.stringify(store.data.lesson.provenance, null, 2) }}</pre></details></div>
    <div class="report-export"><span role="status">{{ exported ? `已导出 ${exported} · 含来源与边界` : '保留完整课程包，便于后续核验。' }}</span><button class="btn" @click="download('md')"><AppIcon name="download" :size="14" />复盘文稿</button><button class="btn primary" @click="download('json')"><AppIcon name="download" :size="14" />完整数据</button></div>
  </section>
</template>
<style scoped>
.lesson-report>.panel-heading>svg{color:#5f7ca6}.report-body{padding:3px 24px 18px}.report-body>.eyebrow{font-size:8px;color:#5f7ca5}.report-body h3{font-size:21px;font-weight:400;color:#385071;margin:12px 0 13px;line-height:1.6}.report-summary{font-size:12px;line-height:2;color:#546d90;white-space:pre-wrap}.report-section{border-top:1px solid #d5e3f7;margin-top:20px;padding-top:17px}.report-section h4{font-size:13px;font-weight:500;color:#466289;margin:0 0 10px;display:flex;align-items:center;gap:10px}.report-section h4>span{font-size:9px;color:#597bac;font-weight:400}.report-section>p{font-size:12px;line-height:2;color:#567095;white-space:pre-wrap;overflow-wrap:anywhere}.report-references{display:flex;gap:6px;flex-wrap:wrap;margin-top:10px}.report-references button{background:#e1ebf9;border-radius:4px;padding:5px 7px;font-size:8px;color:#4e6c97}.report-limits{padding:16px 18px;background:#e2ecf9;border:1px solid #cddbef;border-radius:7px;margin-top:23px}.report-limits h4{font-size:11px;font-weight:500;color:#466a9b;margin:0 0 9px}.report-limits ul{padding-left:16px;margin:0}.report-limits li{font-size:10px;line-height:1.9;color:#5176aa;margin-top:6px}.report-provenance{margin-top:20px;font-size:10px;color:#5776a2}.report-provenance summary{cursor:pointer;padding:7px 0}.report-provenance p{line-height:1.9;margin-top:8px;overflow-wrap:anywhere}.report-provenance h4{font-size:10px;font-weight:400}.report-provenance pre{font-size:9px;line-height:1.8;overflow:auto;max-height:260px;padding:12px;background:#ebf2fb;border-radius:5px}.report-export{border-top:1px solid var(--line);display:flex;align-items:center;gap:9px;padding:18px 22px}.report-export>span{font-size:9px;color:#5c7ca8;margin-right:auto}.report-export .btn{font-size:10px;padding:8px 10px}@media(max-width:550px){.report-body{padding-left:17px;padding-right:17px}.report-export{flex-wrap:wrap;padding:17px}.report-export>span{width:100%;margin-bottom:5px}}
</style>
