import { fileURLToPath, URL } from 'node:url'
import { copyFileSync, mkdirSync } from 'node:fs'
import { resolve } from 'node:path'

import { defineConfig, type Plugin } from 'vite'
import vue from '@vitejs/plugin-vue'

function brandAssets(): Plugin {
  let output = ''
  return {
    name: 'review-brand-assets',
    apply: 'build',
    configResolved(config) { output = resolve(config.root, config.build.outDir) },
    closeBundle() {
      mkdirSync(output, { recursive: true })
      for (const name of ['brand.svg', 'favicon.ico']) copyFileSync(fileURLToPath(new URL(`./public/${name}`, import.meta.url)), resolve(output, name))
    },
  }
}

// https://vite.dev/config/
export default defineConfig({
  plugins: [
    vue(),
    brandAssets(),
  ],
  // Course media is served by /api/review; only brand assets belong in this bundle.
  build: { copyPublicDir: false },
  server: {
    proxy: {
      '/api/review': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
    },
  },
  resolve: {
    alias: {
      '@': fileURLToPath(new URL('./src', import.meta.url))
    },
  },
})
