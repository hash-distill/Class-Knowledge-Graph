# Jingxi Insight: historical light workspace design notes

Date: 2026-10-25

This historical design used an ivory surface, dark blue navigation, and sage green charts. It retained the then-existing Vue / Pinia / ECharts stack and snapshot/graph data interfaces while reorganizing the lesson review workspace. These notes describe that design iteration rather than the current production interface.

## Pages and interactions

- The sidebar contained Lesson overview, Knowledge graph, Participation analysis, and Review assistant links that scrolled to their corresponding sections.
- The summary showed playback duration, tracked people, sample concept segments, and sample engagement values.
- The overview graph showed 24 nodes; the expanded graph showed 40. It supported dragging, zooming, reset, and linked segment navigation.
- The timeline, progress slider, video, and concept segments shared playback state.
- The current JSON snapshot could be exported with a demo-mode notice.
- Dialogs supported Escape, a keyboard focus trap, and focus restoration on close.
- The desktop sidebar became an icon rail on tablets and top navigation on phones. Charts resized with their containers.

## Data limitations

This iteration redesigned frontend visuals and interactions. It did not complete the previously discussed evidence-system reconstruction. Concept mappings, engagement curves, graph activity, and heatmaps included demo logic; AI replies were preset. The interface and exports disclosed these limitations. Preset replies were revised to explain operation without inferring student knowledge mastery. Playback was paused on initial load.

The accompanying `snapshot-export.json` remains explicitly labeled `mode: demo`. Its synthetic metrics are historical preview data and are not connected to current lesson inference or treated as evidence of teaching effectiveness.

## Running and checking

From the project directory, run `npm run dev -- --host 127.0.0.1 --port 5173`. `npm run build` performs type checking and a production build.

On Windows, the build script confines esbuild temporary files to `node_modules/.cache/vite-build-temp` to avoid recurring deletion-denied errors in the local system TEMP directory. This affects the build subprocess only and does not change system environment variables. Other platforms retain their original temporary directory.

The historical `verify-ui.py` used Python Playwright to check playback, segment navigation, graph dialogs, help, exports, demo Q&A, and responsive layouts. Its saved results are in `verification.json`; the referenced screenshots were local design artifacts. These saved results describe that historical interface, not fresh verification of the current release, and do not validate model effectiveness.

## Original backup

A local archive was created before this iteration on 2026-10-25 at 17:07:03. It contained the previous `src`, `index.html`, `package.json`, `package-lock.json`, and `vite.config.ts`. Classroom media and backend source were unchanged by that design pass. The local archive is not included in this release.
