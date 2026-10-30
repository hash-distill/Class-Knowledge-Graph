#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::{
    sync::{atomic::{AtomicBool, Ordering}, Arc, Mutex},
    thread,
    time::Duration,
};
use tauri::{Manager, RunEvent, WebviewUrl, WebviewWindowBuilder};
use tauri_plugin_shell::{process::{CommandChild, CommandEvent}, ShellExt};

struct Backend {
    child: Mutex<Option<CommandChild>>,
    terminated: Arc<AtomicBool>,
}

fn show_error(app: &tauri::AppHandle, message: &str) {
    if let Some(window) = app.get_webview_window("main") {
        // JSON escaping prevents backend error text from becoming executable JS.
        let text = serde_json::to_string(message).unwrap_or_else(|_| "\"Startup failed\"".into());
        let _ = window.eval(format!("document.getElementById('status').textContent={text};"));
    }
}

fn main() {
    let terminated = Arc::new(AtomicBool::new(false));
    let managed = Backend { child: Mutex::new(None), terminated: terminated.clone() };
    let app = tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .manage(managed)
        .setup(move |app| {
            let data = match std::env::var_os("REVIEW_DESKTOP_DATA_DIR") {
                Some(path) => std::path::PathBuf::from(path),
                None => app.path().app_local_data_dir()?.join("state"),
            };
            std::fs::create_dir_all(&data)?;
            let temporary = data.join("backend-temp");
            std::fs::create_dir_all(&temporary)?;
            let permitted_origin = Arc::new(Mutex::new(None::<String>));
            let navigation_origin = permitted_origin.clone();
            WebviewWindowBuilder::new(app, "main", WebviewUrl::App("index.html".into()))
                .title("Jingxi Insight")
                .inner_size(1440.0, 920.0)
                .min_inner_size(1024.0, 700.0)
                .data_directory(data.join("webview"))
                .on_navigation(move |url| {
                    let bundled = matches!(url.scheme(), "tauri" | "asset") ||
                        matches!(url.host_str(), Some("tauri.localhost"));
                    let local = navigation_origin.lock().ok().and_then(|value| value.clone());
                    bundled || local.as_deref() == Some(url.origin().ascii_serialization().as_str())
                })
                .build()?;
            let handle = app.handle().clone();
            let frontend = app.path().resource_dir()?.join("frontend");
            let args = vec!["--port".into(), "0".into(), "--data-dir".into(), data.to_string_lossy().into_owned(),
                "--frontend-dir".into(), frontend.to_string_lossy().into_owned(), "--reuse".into(), "--watch-stdin".into()];
            match app.shell().sidecar("4c-review-backend")
                .map(|command| command.args(args).env("TEMP", &temporary).env("TMP", &temporary).set_process_group(true).spawn()) {
                Ok(Ok((mut rx, child))) => {
                    *app.state::<Backend>().child.lock().unwrap() = Some(child);
                    let exited = terminated.clone();
                    tauri::async_runtime::spawn(async move {
                        let mut ready = false;
                        while let Some(event) = rx.recv().await {
                            match event {
                                CommandEvent::Stdout(bytes) => {
                                    let line = String::from_utf8_lossy(&bytes);
                                    if let Some(payload) = line.trim().strip_prefix("4C_DESKTOP_READY ") {
                                        if let Ok(value) = serde_json::from_str::<serde_json::Value>(payload) {
                                            if let Some(url_text) = value.get("url").and_then(|url| url.as_str()) {
                                                if let Ok(url) = url_text.parse::<tauri::Url>() {
                                                    if url.scheme() == "http" && url.host_str() == Some("127.0.0.1") && url.port().is_some() {
                                                        *permitted_origin.lock().unwrap() = Some(url.origin().ascii_serialization());
                                                        if let Some(window) = handle.get_webview_window("main") {
                                                            if let Err(error) = window.navigate(url) {
                                                                show_error(&handle, &format!("Unable to open the workbench: {error}"));
                                                            } else { ready = true; }
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                                CommandEvent::Terminated(_) => {
                                    exited.store(true, Ordering::Release);
                                    if !ready { show_error(&handle, "The backend failed to start. Use the documented command-line entry point to inspect the error and check the frontend files."); }
                                    break;
                                }
                                CommandEvent::Error(_) => {
                                    if !ready { show_error(&handle, "Unable to read backend output. Check the installation directory and local permissions."); }
                                }
                                _ => {}
                            }
                        }
                    });
                }
                _ => show_error(&handle, "Unable to start the backend. Check that 4c-review-backend.exe was installed with the application."),
            }
            Ok(())
        })
        .build(tauri::generate_context!())
        .expect("Failed to initialize the Jingxi Insight desktop window");
    app.run(|app, event| {
        if matches!(event, RunEvent::ExitRequested { .. } | RunEvent::Exit) {
            let state = app.state::<Backend>();
            let owned_child = state.child.lock().unwrap().take();
            if let Some(mut child) = owned_child {
                // Ask the owned wrapper to shut down. A reused service is never
                // killed: its wrapper exits without stopping the original owner.
                let _ = child.write(b"stop\n");
                for _ in 0..30 {
                    if state.terminated.load(Ordering::Acquire) { break; }
                    thread::sleep(Duration::from_millis(100));
                }
                if !state.terminated.load(Ordering::Acquire) { let _ = child.kill(); }
            }
        }
    });
}
