<script setup lang="ts">
withDefaults(defineProps<{ name: string; size?: number }>(), { size: 20 })
const paths: Record<string, string[]> = {
  grid: ['M3 3h7v7H3zM14 3h7v7h-7zM3 14h7v7H3zM14 14h7v7h-7z'],
  graph: ['M6 7l6 5m0 0 6-5m-6 5v7', 'M8 5a2 2 0 1 1-4 0 2 2 0 0 1 4 0ZM20 5a2 2 0 1 1-4 0 2 2 0 0 1 4 0ZM14 12a2 2 0 1 1-4 0 2 2 0 0 1 4 0ZM14 21a2 2 0 1 1-4 0 2 2 0 0 1 4 0Z'],
  chart: ['M4 3v17h17M8 15v-4m5 4V6m5 9v-7'],
  spark: ['m12 3 2.5 6.5L21 12l-6.5 2.5L12 21l-2.5-6.5L3 12l6.5-2.5L12 3Z'],
  book: ['M12 6v15M3 4c4-1 6 0 9 2 3-2 5-3 9-2v14c-4-1-6 0-9 2-3-2-5-3-9-2V4Z'],
  arrow: ['M5 12h14m-6-6 6 6-6 6'],
  chevron: ['m9 5 7 7-7 7'],
  download: ['M12 3v12m-5-5 5 5 5-5M4 16v5h16v-5'],
  help: ['M9 9a3 3 0 1 1 5 2c-1 .7-2 1-2 3M12 17h.01', 'M22 12a10 10 0 1 1-20 0 10 10 0 0 1 20 0Z'],
  clock: ['M12 7v5l3 2', 'M21 12a9 9 0 1 1-18 0 9 9 0 0 1 18 0Z'],
  play: ['m9 5 11 7-11 7V5Z'],
  pause: ['M8 5v14M16 5v14'],
  expand: ['M8 3H3v5m13-5h5v5M3 16v5h5m13-5v5h-5'],
  reset: ['M3 10a9 9 0 1 1 1 7M3 4v6h6'],
  close: ['m6 6 12 12M6 18 18 6'],
  send: ['M12 19V5m-6 6 6-6 6 6'],
  film: ['M3 3h18v18H3zM7 3v18M17 3v18M3 8h4m-4 8h4M17 8h4m-4 8h4'],
  check: ['m5 12 4 4L19 6'],
}
</script>
<template>
  <svg :width="size" :height="size" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.65" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path v-for="(d, i) in paths[name] || paths.grid" :key="i" :d="d" /></svg>
</template>
