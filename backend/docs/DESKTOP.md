# Jingxi Insight desktop and runtimes

The Tauri 2 shell uses installed Windows WebView2. A separate Python backend binds only to `127.0.0.1`. Interface assets, lesson evidence, workspace state, and model runtimes are stored separately. Internal binary names, readiness markers, and application IDs retain their compatible historical identifiers.

## Run source

From the repository root:

```powershell
& .\desktop\start_source.ps1
```

This opens the system browser using `backend/.venv-media`. First install `backend/requirements-app.txt` and run `npm ci` and `npm run build` in `frontend/`. See the [application README](../../README.md) for a fresh setup.

The shortcut stores state in `desktop/workspace/source-state`. Override it with `-DataDir` or `REVIEW_DATA_DIR`. You can also specify paths directly:

```powershell
& .\backend\.venv-media\Scripts\python.exe .\backend\scripts\desktop_entry.py `
  --port 0 --registry .\backend\configs\lessons.json `
  --frontend-dir .\frontend\dist --reuse --open-browser
```

`--port 0` requests an available OS port. Startup prints `4C_DESKTOP_READY` and the address. The direct entry defaults to `%LOCALAPPDATA%/org.4c.workbench/state`; use `--data-dir` to change it. The source entry reads the existing `configs/lessons.json`. Close its terminal or press `Ctrl+C` to stop the backend it started.

`--reuse` requires the same workspace, matching startup fingerprint, and a private random-token handshake. It does not infer ownership from ports, names, or PIDs. The handshake disallows proxies and redirects. A reusing launcher leaves the original service running; unrelated services are untouched.

## Build Windows packages

Requirements: Node.js, Rust MSVC, Visual Studio desktop C++ tools, and WebView2. Administrator rights are not required. Prefer existing tools; otherwise follow [Tauri prerequisites](https://v2.tauri.app/start/prerequisites/). Optional project-local `CARGO_HOME` and `RUSTUP_HOME` paths are `desktop/.toolchains/cargo` and `desktop/.toolchains/rustup`; use `--no-modify-path` with the installer and set variables only in the current PowerShell session.

Create a separate packaging environment from the root:

```powershell
& .\backend\.venv-media\Scripts\python.exe -m venv .\desktop\.venv-build
& .\desktop\.venv-build\Scripts\python.exe -m pip install `
  -r .\desktop\requirements-build.txt `
  -r .\backend\requirements-review.txt `
  -r .\backend\requirements-courseware.txt
```

Build frontend and desktop:

```powershell
Push-Location .\frontend
npm ci
npm run build
Pop-Location
Push-Location .\desktop
npm ci
& .\scripts\build.ps1
Pop-Location
```

PyInstaller first creates `desktop/src-tauri/binaries/4c-review-backend-x86_64-pc-windows-msvc.exe`, then Tauri builds the shell. The backend includes only allowlisted public configuration, excluding `lessons.json`, lesson outputs, video, and weights. Tauri maps the current `frontend/dist` directly to packaged `frontend` resources to avoid stale staging copies. Use `build.ps1 -NoBundle` for a portable executable, adding `-SkipBackend` when the backend is already built.

Tauri outputs to `desktop/src-tauri/target/release`; NSIS packages are under `bundle/nsis`. Installation defaults to the current user. The package reuses system WebView2; install it first if absent. Preserve actual Rust, linker, or network errors from failed builds. Configuration files alone do not prove a runnable package exists.

## Run packaged applications

The shell starts its backend and waits for readiness before opening the workbench. On exit it requests graceful shutdown over stdin, then terminates only its own subprocess group on timeout. It does not stop external Python, model services, or same-name processes.

In a source checkout, `desktop/start_desktop.ps1` opens the built application with the current registry/runtime manifest. It uses separate writable state in `desktop/workspace/desktop-state`; override `-Registry`, `-DataDir`, or `-RuntimeRoot`. Use `start_source.ps1` for browser fallback.

A packaged application's first registry is empty. To use an existing registry, set it before launch:

```powershell
$env:REVIEW_LESSON_REGISTRY = (Resolve-Path .\backend\configs\lessons.json).Path
& .\desktop\src-tauri\target\release\four-c-desktop.exe
```

If AppData is restricted, set `REVIEW_DESKTOP_DATA_DIR` to a dedicated writable workspace. Databases, backend temporary files, and WebView user data go there; source lesson paths are unchanged.

The registry can still point to existing local evidence. Databases remain separate in local application data. Models go under `runtime/artifacts/models` in that workspace; public configuration is copied to `runtime/configs` on first run. `REVIEW_RUNTIME_ROOT` can select an existing root containing `configs` and `artifacts`. The app does not automatically download models.

## Model/runtime inspection

`review.runtime.inspect_runtime()` is read-only and exposed by the app at `GET /api/review/runtime`. It checks configuration and nonempty files without importing model libraries, connecting to the network, reading video, or initializing GPUs. Returned paths are relative to the runtime root. Absolute paths, traversal, and links outside that root are rejected.

`configs/runtime_profiles.json` covers ASR, OCR, embeddings, Q&A, and proofreading. Configuration, model files, and executables are reported separately. `installed_unverified` means file checks passed while inference remains unverified. Do not translate `service_status=not_checked`, `inference_status=not_verified`, or `available=false` into a claim that the model is available. Inspection does not start or call a model.

Qwen3-ASR-0.6B, PP-OCRv6 ONNX, and BGE-small are recommended installations. OCR/embedding ONNX names are reserved conventions; adapters are unfinished. Missing files report `missing`. Experiments configured on another computer still show missing when local weights are absent.

Install and start models and CPU/CUDA engines separately. Retain upstream model cards, licenses, versions, checksums, and download sources. Installers exclude these models and do not accept or replace their terms. The API process does not load GPU models.

## Workers and permissions

Frozen jobs use:

```text
4c-review-backend.exe --run-worker --db <jobs.sqlite3> --registry <lessons.json> --job <id> --token <token>
```

`--run-worker` must be first. The entry removes it and forwards remaining arguments unchanged to `review.job_worker.main()`, bypassing desktop startup. The queue generates paths, tokens, and lesson arguments; web pages do not build shell commands.

Courseware matching also uses a separate native-library process. The frozen entry accepts only the fixed `-m review.courseware_alignment` JSON protocol, not arbitrary modules. The bundled PowerPoint conversion script still requires local PowerPoint for PPT/PPTX; text PDFs do not.

Tauri capabilities grant web content no shell, filesystem, or custom native-command access and expose no global Tauri object. Native code runs a fixed sidecar. Navigation is limited to the built-in startup page and current backend origin. Backend CSP restricts pages to same-origin resources and prohibits embedded objects/external scripts. API path and version checks continue to protect lesson evidence.

## Verification

From the root:

```powershell
& .\backend\.venv-media\Scripts\python.exe -m pytest .\backend\tests\test_desktop_runtime.py -q
```

Tests cover missing/empty models, damaged configuration, path sanitization, uninstalled recommendations, startup arguments, frozen-worker arguments, and real HTTP start/reuse/shutdown. Windows symlink tests skip without permission. They do not measure model accuracy or speed.

To repeat lifecycle tests for an actual frozen backend, set `REVIEW_DESKTOP_BINARY` to its absolute path in the same PowerShell session. The shell puts PyInstaller extraction under its own `backend-temp` workspace to avoid restricted system temporary directories.

Historical Tauri reference checks used official [sidecar](https://v2.tauri.app/develop/sidecar/), [capabilities](https://v2.tauri.app/security/capabilities/), and [CSP](https://v2.tauri.app/security/csp/) documentation. Context7 resolution was unavailable in that recorded run, so official pages were the fallback.

## Historical build record — 2026-10-28

A Windows x64 executable, separate Python backend, and NSIS package were generated. The shell used Tauri 2.12.5 and installed WebView2. Rust 1.99.1 and build caches were project-local; system PATH was unchanged.

Tests against the actual frozen backend reported **13 passed, one skipped**. They exercised PDF upload, persistent jobs, frozen workers, separate image matching, publication, and the final recovery/cancellation boundary. The skip was the Windows symlink-permission test.

`start_desktop.ps1` was checked with the real registry: six visible lessons and seven runtime profiles. A native window loaded a lesson and advanced video time. After normal exit, registry/bundle digests were unchanged; source-video size and modification time were unchanged. The video was not rehashed in full for this check.

Local `desktop/build/release-manifest.json` records outputs, inputs, and SHA-256 hashes; `desktop/build/native-real-registry-check.json` records native acceptance. These generated records are excluded from source releases. The installer was built, but its wizard was not run. Model inference remained `not_verified` as reported by inspection.
