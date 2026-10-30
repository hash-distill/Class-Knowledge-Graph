import { mkdirSync } from 'node:fs'
import { spawnSync } from 'node:child_process'
import { join } from 'node:path'
import { fileURLToPath } from 'node:url'

const root = fileURLToPath(new URL('../', import.meta.url))
const env = { ...process.env }

// This machine denies esbuild's cleanup in the Windows system TEMP directory.
// Keep temporary build files in the project's ignored dependency cache instead.
// Only the child build process receives this environment; system settings stay intact.
if (process.platform === 'win32') {
  const buildTemp = join(root, 'node_modules', '.cache', 'vite-build-temp')
  mkdirSync(buildTemp, { recursive: true })
  env.TEMP = buildTemp
  env.TMP = buildTemp
}

const result = spawnSync(
  process.execPath,
  [join(root, 'node_modules', 'vite', 'bin', 'vite.js'), 'build', ...process.argv.slice(2)],
  { cwd: root, env, stdio: 'inherit' },
)

if (result.error) console.error(result.error.message)
process.exitCode = result.status ?? 1
