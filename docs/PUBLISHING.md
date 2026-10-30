# Publish to GitHub

Target repository: [hash-distill/Class-Knowledge-Graph](https://github.com/hash-distill/Class-Knowledge-Graph). Keep the repository URL and use **Jingxi Insight** as the application title. Suggested description: "Jingxi Insight: a local classroom review assistant for teachers, combining transcripts, courseware alignment, visible behavior analysis, manual correction, knowledge graphs, and teaching improvement records with FastAPI, Vue 3, and Tauri."

This publication round includes the complete application source in the unified English-named layout. The interface defaults to English; original lesson text and evidence remain in their source language. Classroom media, personal workspaces, model weights, and generated lesson artifacts remain excluded.

## Generate unified source

The original workspace contains separate Git metadata in vision, frontend, and historical graph modules. Export from the application root instead of staging that workspace wholesale:

```powershell
python scripts/prepare_release.py --check
python scripts/prepare_release.py --output release/github-ready
```

The output directory must not already exist. `source/` is the unified source tree, the ZIP contains the same snapshot, and `manifest.json` records source paths, sizes, and SHA-256 hashes. Export preserves existing repositories, source video, environments, databases, and models.

Filters exclude environments, caches, builds, weights, lesson artifacts, private registries, and historical frontend media exports. Review the manifest and text for personal paths, SSH configuration, and material rights before committing; credential checks cover common patterns only.

## Preserve remote history

Use a separate clone and a release branch, then apply the clean source tree. The remote's historical root `Class_Detection/` becomes `vision/Class_Detection/`. Preserve Git history while moving vision files, requirements, training guides, and GPU inspection scripts into the module. The unified application directories are `backend/`, `frontend/`, `vision/`, and `desktop/`.

Historical `.idea` files and snapshot JSON are IDE/classroom outputs and are excluded from the new source tree; old commits retain them. In a prepared checkout, inspect `git status --short`, `git diff --stat`, and `git diff --cached --stat`. Do not clean or reset the original workspace.

Push the reviewed application source using the maintainer's Git identity and existing authentication. Preserve remote history and avoid force-pushing. Keep local data and model paths out of commits; never place access tokens in scripts. Review the branch's layout, validation results, and README before merging it into the default branch.

## Local acceptance

Install application dependencies in a fresh source directory. Run backend tests and frontend `npm ci`, `npm test`, and `npm run build`. After `scripts/workspace_setup.py`, the registry should be empty. Starting `scripts/run_workbench.py` should expose the health endpoint, an empty lesson list, and the interface.

CI verifies code and builds. It does not download classroom models, access private materials, or publish installers. Evaluate real-video recognition, model answers, and the Windows installation wizard separately. Until the overall license is established, do not add an unsupported open-source license badge.
