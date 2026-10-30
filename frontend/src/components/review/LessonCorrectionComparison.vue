<script setup lang="ts">
import { headingText } from '@/utils/reviewPresentation'
import { computed, ref } from 'vue'
import { useLessonReviewStore } from '@/stores/lessonReview'
import { formatReviewTime } from '@/utils/lessonReview'
import type { CorrectionComparisonRun } from '@/types/lessonReview'

const store = useLessonReviewStore()
const onlyChanged = ref(false), search = ref('')
const lane = ref<'guarded' | 'native'>('guarded')
const comparison = computed(() => store.data?.correction_comparison)
const runs = computed(() => ['qwen', 'cec4'].map(id => comparison.value?.runs.find(run => run.id === id)))
const segments = computed(() => (store.data?.transcript || []).map(source => ({
  ...source,
  outputs: runs.value.map(run => (lane.value === 'native' ? run?.native_segments : run?.segments)?.find(segment => segment.transcript_id === source.id)),
})).filter(segment => segment.outputs.some(Boolean))
  .filter(segment => !onlyChanged.value || segment.outputs.some(output => output && output.text !== segment.text))
  .filter(segment => !search.value || [segment.text, ...segment.outputs.map(output => output?.text || '')].some(text => text.includes(search.value))))
const metric = (value: number | null | undefined) => value === null || value === undefined ? '未知' : value.toLocaleString('zh-CN')
const memory = (value: number | null | undefined) => value === null || value === undefined ? '未知' : `${value.toLocaleString('zh-CN', { maximumFractionDigits: 1 })} MiB`
const evaluatedAt = computed(() => comparison.value ? new Date(comparison.value.evaluated_at).toLocaleString('zh-CN', { hour12: false }) : '')
const runLabel = (run: CorrectionComparisonRun | undefined, index: number) => run?.model_label || (index === 0 ? 'Qwen3-4B（无运行记录）' : 'CEC4（无运行记录）')
const uncertaintyText = (values: unknown[]) => values.map(value => typeof value === 'string' ? (/^[A-Z]+\d+(?:_\d+)?$/.test(value) ? '本段仍有未确认内容，请回听原声' : value) : JSON.stringify(value)).join('；')
const statuses: Record<string, string> = { complete: '已完成', partial: '未完整完成', running: '处理中', failed: '失败', pending: '尚未完成', unavailable: '未提供' }
const statusText = (value: string | undefined) => value ? (statuses[value] || value) : '未知'
</script>

<template>
  <details v-if="comparison" class="correction-comparison" data-testid="correction-comparison">
    <summary data-testid="correction-comparison-toggle"><span>校对模型对比<small>同一课堂 · 原文与两次实际运行</small></span><span class="comparison-expand"><span class="comparison-open-label">展开核对</span><span class="comparison-close-label">收起对比</span> <i aria-hidden="true">⌄</i></span></summary>
    <div class="comparison-body">
      <p class="comparison-boundary">本课没有人工逐字稿。改动数、术语出现数与拦截数仅描述运行结果，不能据此判断准确率或误改率。</p>
      <div class="comparison-runs">
        <article v-for="(run, index) in runs" :key="index" class="comparison-run" :data-testid="`correction-comparison-run-${index === 0 ? 'qwen' : 'cec4'}`">
          <h3><span>{{ index === 0 ? '01' : '02' }}</span>{{ headingText(runLabel(run, index)) }}</h3>
          <p class="comparison-status">受控校对：{{ statusText(run?.status) }}<span>直接输出：{{ statusText(run?.native_status) }}</span></p>
          <dl class="comparison-contributions"><div><dt>接受模型候选</dt><dd>{{ metric(run?.model_accepted_edit_count) }}<small>处</small></dd></div><div><dt>来源规则修订</dt><dd>{{ metric(run?.source_rule_accepted_count) }}<small>处</small></dd></div></dl>
          <p class="comparison-additional" data-testid="comparison-additional-spans">超出纯规则结果：{{ run?.additional_changed_spans_vs_rules == null ? '未知' : `${metric(run?.additional_changed_spans_vs_rules)}处（待核验）` }}</p>
          <dl class="comparison-metrics">
            <div><dt>最终接受合计</dt><dd>{{ metric(run?.accepted_edit_count) }}<small>处</small></dd></div>
            <div><dt>模型提议</dt><dd>{{ metric(run?.model_proposal_count) }}<small>条</small></dd></div>
            <div><dt>规则拦截</dt><dd>{{ metric(run?.guard_rejected_count) }}<small>条</small></dd></div>
            <div><dt>规范术语组出现</dt><dd>{{ metric(run?.term_presence_count) }}</dd></div>
            <div><dt>术语组数</dt><dd>{{ metric(run?.terminology_group_count) }}</dd></div>
          </dl>
          <div class="comparison-memory"><strong>受控校对 · 整卡采样显存</strong><dl><div><dt>加载前基线</dt><dd>{{ memory(run?.memory?.baseline_mib) }}</dd></div><div><dt>加载后</dt><dd>{{ memory(run?.memory?.loaded_mib) }}</dd></div><div><dt>运行峰值</dt><dd>{{ memory(run?.memory?.peak_mib) }}</dd></div></dl><small>采样间隔 {{ metric(run?.memory?.sample_interval_ms) }} ms · 含同卡其他进程占用</small></div>
          <details v-if="run" class="comparison-model-source"><summary>模型文件来源</summary><p>SHA-256：{{ run.model_sha256 }}</p></details>
        </article>
      </div>
      <div class="comparison-lane"><label for="correction-comparison-lane">逐段对照</label><select id="correction-comparison-lane" v-model="lane" data-testid="correction-comparison-lane" aria-label="选择校对对照输出"><option value="guarded">守卫后结果</option><option value="native">模型直接输出</option></select><small>{{ lane === 'native' ? '未经规则校验的模型假设' : '来源规则与模型候选经守卫后的最终结果' }}</small></div>
      <p v-if="lane === 'native'" class="comparison-boundary native-boundary" data-testid="correction-comparison-native-notice">模型直接输出未经规则校验，可能有补写、整句改写或误改。以下内容仅供回听对照，不进入知识图谱与问答引用。</p>
      <div class="comparison-tools"><input v-model="search" type="search" aria-label="搜索校对模型对比" placeholder="搜索原文或模型输出…" /><label><input v-model="onlyChanged" type="checkbox" />只看有改动</label></div>
      <div class="comparison-segments" aria-label="逐段校对模型输出">
        <article v-for="segment in segments" :key="segment.id" class="comparison-segment" :data-testid="`comparison-segment-${segment.id}`">
          <button class="comparison-seek" :aria-label="`${formatReviewTime(segment.start_sec)} 回听原声 ${segment.id}`" @click="store.seek(segment.start_sec)"><span class="mono">{{ formatReviewTime(segment.start_sec) }}–{{ formatReviewTime(segment.end_sec) }}</span><span>{{ segment.id }} · 回听原声 ↗</span></button>
          <div class="comparison-text original"><strong>原始转写</strong><p>{{ segment.text }}</p></div>
          <div v-for="(output, index) in segment.outputs" :key="index" class="comparison-text" :class="{ changed: output && output.text !== segment.text }"><strong>{{ index === 0 ? 'Qwen' : 'CEC4' }}<small v-if="output">{{ lane === 'native' ? '未经校验' : output.text === segment.text ? '保留原文' : '改动待核验' }}</small></strong><p>{{ output ? output.text : '未知：本段无模型运行记录' }}</p><p v-if="output?.uncertainties.length" class="comparison-uncertainty">待核验：{{ uncertaintyText(output.uncertainties) }}</p></div>
        </article>
        <p v-if="!segments.length" class="comparison-empty">{{ lane === 'native' && !runs.some(run => run?.native_segments?.length) ? '暂无模型直接输出记录。' : '没有符合筛选条件的对比段落。' }}</p>
      </div>
      <div class="comparison-source"><p>运行时间：{{ evaluatedAt }} · {{ comparison.runs.length }} 份模型记录</p><p>对比输出仅供逐段回听核验，不作为知识图谱或问答引用。</p><details><summary>查看对比来源与说明</summary><p>录像 SHA-256：{{ comparison.source_video_sha256 }}</p><p v-for="limit in comparison.limitations" :key="limit">{{ limit }}</p><p>完整数据导出保留两次运行、原文、时间与测量记录。</p></details></div>
    </div>
  </details>
</template>

<style scoped>
.comparison-additional{font-size:10px;line-height:1.8;color:#466ba0;margin:9px 0 0}
.comparison-status{font-size:10px;line-height:1.8;color:#43689c;margin-top:8px}.comparison-status>span{display:block;color:#576f91}.comparison-contributions{display:grid;grid-template-columns:1fr 1fr;gap:10px;background:#e2ecf9;border-radius:5px;padding:12px;margin:12px 0 0}.comparison-contributions dt{font-size:10px;color:#587297;line-height:1.8}.comparison-contributions dd{font-size:23px;line-height:1.5;color:#39557b;margin:3px 0 0;font-variant-numeric:tabular-nums}.comparison-contributions dd>small{font-size:9px;color:#607ca4;margin-left:5px}.comparison-lane{display:flex;align-items:center;flex-wrap:wrap;gap:9px 12px;margin:17px 0 0;font-size:11px;color:#4e698f}.comparison-lane select{min-width:0;max-width:100%;font-size:11px;color:#3f5a80;background:#fffefa;border:1px solid #cdd9ea;border-radius:5px;padding:8px 10px}.comparison-lane>small{font-size:10px;line-height:1.8;color:#5d789e}.native-boundary{margin-bottom:0}
.correction-comparison{margin:0 22px 20px;border:1px solid #d2dbe8;border-radius:8px;background:#f4f8fd;min-width:0}.correction-comparison>summary{display:flex;justify-content:space-between;align-items:center;gap:12px;list-style:none;cursor:pointer;padding:14px 15px;color:#3f587c;font-size:12px;font-weight:500}.correction-comparison>summary::-webkit-details-marker{display:none}.correction-comparison>summary small{display:block;color:#5e7493;font-size:10px;font-weight:400;margin-top:5px;line-height:1.6}.comparison-expand{font-size:10px;font-weight:400;white-space:nowrap;color:#546b8b}.comparison-close-label{display:none}.correction-comparison[open] .comparison-open-label{display:none}.correction-comparison[open] .comparison-close-label{display:inline}.comparison-expand i{font-style:normal;margin-left:5px;display:inline-block;transition:transform .2s}.correction-comparison[open] .comparison-expand i{transform:rotate(180deg)}.comparison-body{padding:0 14px 14px;border-top:1px solid #d0e0f6}.comparison-boundary{font-size:11px;color:#466ba0;line-height:1.9;background:#dee9f9;padding:11px 12px;margin:13px 0;border-radius:5px}.comparison-runs{display:grid;grid-template-columns:1fr 1fr;gap:10px}.comparison-run{min-width:0;padding:13px;background:#fffefa;border:1px solid #d1e0f6;border-radius:6px}.comparison-run h3{font-size:12px;font-weight:500;color:#364f73;line-height:1.8;overflow-wrap:anywhere}.comparison-run h3>span{font-size:9px;color:#607ca4;margin-right:8px}.comparison-metrics{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:13px 10px;margin:14px 0}.comparison-metrics dt,.comparison-memory dt{font-size:10px;color:#5a7192;line-height:1.8}.comparison-metrics dd{font-size:17px;font-variant-numeric:tabular-nums;color:#40597d;margin:3px 0 0}.comparison-metrics dd small{font-size:9px;color:#657d9f;margin-left:4px}.comparison-memory{border-top:1px solid #d6e4f7;padding-top:10px}.comparison-memory>strong{font-size:10px;font-weight:500;color:#4d678b}.comparison-memory dl{margin:8px 0}.comparison-memory dl>div{display:flex;justify-content:space-between;gap:10px;margin:5px 0}.comparison-memory dd{margin:0;font-size:10px;line-height:1.8;color:#4d678b;font-variant-numeric:tabular-nums}.comparison-memory>small{display:block;font-size:9px;line-height:1.8;color:#607a9e}.comparison-model-source{font-size:9px;color:#597293;margin-top:9px}.comparison-model-source summary{cursor:pointer}.comparison-model-source p{overflow-wrap:anywhere;line-height:1.8;margin-top:7px}.comparison-tools{display:flex;gap:12px;align-items:center;margin:17px 0 12px}.comparison-tools>input{min-width:0;flex:1;border:1px solid #d2dceb;border-radius:5px;background:#fffefa;padding:9px;font-size:11px;color:#425b7e}.comparison-tools label{font-size:10px;white-space:nowrap;color:#576d8d;display:flex;gap:4px;align-items:center}.comparison-tools input[type=checkbox]{accent-color:var(--blue)}.comparison-segments{max-height:490px;overflow:auto;overscroll-behavior:contain;padding-right:3px}.comparison-segment{padding:12px 0;border-top:1px solid #cddef5}.comparison-seek{display:flex;flex-wrap:wrap;justify-content:space-between;gap:7px 12px;background:transparent;color:#436491;font-size:10px;width:100%;padding:0 0 10px;text-align:left}.comparison-seek:hover{color:var(--navy)}.comparison-text{display:grid;grid-template-columns:75px minmax(0,1fr);gap:9px;padding:9px 10px;color:#506787;font-size:12px;line-height:1.9;border-radius:4px}.comparison-text strong{font-size:10px;font-weight:500;color:#567196}.comparison-text strong>small{display:block;font-size:9px;line-height:1.7;font-weight:400;color:#5f7ba2}.comparison-text p{white-space:pre-wrap;overflow-wrap:anywhere;min-width:0}.comparison-text.original{background:#dfeaf9;color:#5c7290}.comparison-text.original strong{color:#5c7291}.comparison-text.changed{background:#e3edfa}.comparison-uncertainty{grid-column:2;font-size:10px;color:#4472b2}.comparison-empty{padding:20px 8px;text-align:center;font-size:11px;color:#61799b}.comparison-source{border-top:1px solid #cddef6;margin-top:12px;padding-top:11px;font-size:10px;color:#5d7699;line-height:1.9}.comparison-source>p{margin:3px 0}.comparison-source details{margin-top:7px}.comparison-source summary{cursor:pointer}.comparison-source details p{overflow-wrap:anywhere;margin-top:7px}@media(max-width:700px){.comparison-runs{grid-template-columns:1fr}.comparison-metrics{grid-template-columns:repeat(3,minmax(0,1fr))}}@media(max-width:550px){.correction-comparison{margin:0 15px 17px}.correction-comparison>summary{padding:12px}.comparison-body{padding:0 10px 11px}.comparison-text{grid-template-columns:62px minmax(0,1fr);gap:7px;padding:9px 7px}.comparison-tools{gap:8px}.comparison-tools>input{font-size:10px}.comparison-expand{font-size:9px}.comparison-segments{max-height:440px}}
</style>
