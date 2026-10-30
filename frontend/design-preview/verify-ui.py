"""Manual-delivery browser smoke checks for the classroom workbench redesign."""
import json
from pathlib import Path
from playwright.sync_api import sync_playwright, expect, TimeoutError

out = Path(__file__).parent
results = {}
with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    context = browser.new_context(viewport={"width": 1512, "height": 982}, device_scale_factor=1, reduced_motion="reduce")
    page = context.new_page()
    errors = []
    page.on("pageerror", lambda error: errors.append(str(error)))
    page.goto("http://127.0.0.1:5173", wait_until="domcontentloaded", timeout=60000)
    expect(page.get_by_role("button", name="导出当前快照")).to_be_enabled(timeout=60000)
    try:
        page.wait_for_load_state("networkidle", timeout=5000)
    except TimeoutError:
        # Local video range requests may remain active; verify rendered state instead.
        pass
    expect(page.locator(".graph-count").first).to_contain_text("24")
    page.screenshot(path=str(out / "desktop.png"), full_page=True)
    page.screenshot(path=str(out / "desktop-viewport.png"))
    results["initial_video"] = page.locator("video").evaluate("v=>({ready:v.readyState,error:v.error?.message,duration:v.duration})")

    page.get_by_role("button", name="全屏查看知识图谱").click()
    expect(page.get_by_role("dialog", name="课程知识图谱")).to_be_visible()
    expect(page.get_by_role("dialog").locator(".graph-count")).to_contain_text("40")
    page.screenshot(path=str(out / "graph-expanded.png"))
    page.keyboard.press("Escape")
    expect(page.get_by_role("dialog")).to_have_count(0)
    expect(page.get_by_role("button", name="全屏查看知识图谱")).to_be_focused()
    results["graph_modal_and_focus_restore"] = "passed"

    page.get_by_role("button", name="查看使用说明", exact=True).click()
    expect(page.get_by_role("dialog")).to_contain_text("预设演示回复")
    page.get_by_role("button", name="关闭弹窗").click()
    assert page.evaluate("document.body.style.overflow") != "hidden"
    results["help_modal"] = "passed"

    page.get_by_role("button", name="查看全部 8 个片段").click()
    expect(page.locator(".chapter")).to_have_count(8)
    page.locator(".chapter").nth(5).click()
    expect(page.locator(".active-topic strong")).to_have_text("分数意义")
    expect(page.locator(".time-now")).not_to_have_text("00:00")
    results["chapter_seek_seconds"] = page.locator("video").evaluate("v=>v.currentTime")
    page.get_by_role("button", name="收起片段").click()
    page.get_by_role("slider", name="课堂回放进度").fill("50")
    expect(page.locator(".time-now")).to_have_text("02:27")
    results["slider_seek"] = "passed"

    page.get_by_role("button", name="播放课堂回放").click()
    expect(page.get_by_role("button", name="暂停回放")).to_be_visible()
    page.wait_for_function("document.querySelector('video').currentTime > 148.5")
    page.get_by_role("button", name="暂停回放").click()
    assert page.locator("video").evaluate("v=>v.paused")
    results["play_pause"] = "passed"

    with page.expect_download() as download_info:
        page.get_by_role("button", name="导出当前快照").click()
    download = download_info.value
    download.save_as(str(out / "snapshot-export.json"))
    payload = json.loads((out / "snapshot-export.json").read_text(encoding="utf-8"))
    assert payload["mode"] == "demo" and "snapshot" in payload
    results["snapshot_export"] = "passed"

    page.get_by_role("button", name="如何回看当前知识点？", exact=True).click()
    expect(page.locator(".message.user")).to_contain_text("如何回看当前知识点")
    expect(page.locator(".message.ai")).to_contain_text("不能直接推断学生是否掌握了知识。", timeout=20000)
    results["demo_chat"] = "passed"
    page.locator("#assistant").scroll_into_view_if_needed()
    page.screenshot(path=str(out / "assistant.png"))

    page.get_by_role("button", name="回到起点").click()
    for width, height, name in [(1280, 900, "laptop"), (1024, 768, "tablet-landscape"), (768, 1024, "tablet"), (390, 844, "mobile")]:
        page.set_viewport_size({"width": width, "height": height})
        page.get_by_role("link", name="课堂概览", exact=True).click()
        page.wait_for_function("window.scrollY < 130")
        size = page.evaluate("({viewport:innerWidth,document:document.documentElement.scrollWidth})")
        assert size["document"] <= size["viewport"], f"Horizontal overflow at {name}: {size}"
        results[name] = size
        page.screenshot(path=str(out / f"{name}.png"), full_page=True)
    page.get_by_role("link", name="复盘助手", exact=True).click()
    expect(page.locator("#assistant")).to_be_in_viewport()
    results["mobile_navigation"] = "passed"
    results["page_errors"] = errors
    assert not errors, errors
    (out / "verification.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(results, ensure_ascii=False, indent=2))
    context.close()
    browser.close()
