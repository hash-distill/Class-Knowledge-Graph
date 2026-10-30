"""Real published lesson: independent ASR clocks, conservative proof and export."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

from playwright.sync_api import expect, sync_playwright


p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--url', default='http://127.0.0.1:5175')
p.add_argument('--lesson', default='qwen35-chemistry-001-9')
p.add_argument('--output', type=Path, default=Path('output/playwright/qwen-upgrade-real'))
args = p.parse_args()
args.output.mkdir(parents=True, exist_ok=True)
backend = Path(__file__).resolve().parents[2]/'backend'


def fetch():
    with urllib.request.urlopen(args.url+'/api/review/lessons/'+args.lesson, timeout=10) as response:
        return json.load(response)


data = fetch()
experiment = data['model_upgrade_experiment']
proof = data['transcript_proofreading']
assert proof['status'] == 'complete' and proof['model'] == 'classroom-corrector-qwen35'
assert proof['provenance']['model_label'] == 'Qwen3.5-4B'
assert experiment['reference_status'] == 'no_human_verbatim_reference'
assert experiment['correction']['accepted_edit_count'] == proof['edit_count']
for filename, digest in json.loads((backend/'artifacts/verification/qwen-upgrade-20261028/source-before.json').read_text(encoding='utf-8')).items():
    assert hashlib.sha256((backend/'artifacts/unified-chemistry-001-9'/filename).read_bytes()).hexdigest() == digest
runs = {r['id']: r for r in experiment['asr_runs']}
errors, checks = [], []
with sync_playwright() as pw:
    browser = pw.chromium.launch()
    page = browser.new_page(viewport={'width': 1440, 'height': 1050}, device_scale_factor=1)
    page.on('pageerror', lambda error: errors.append(str(error)))
    page.goto(args.url+'/?lesson='+args.lesson+'&view=evidence', wait_until='networkidle')
    page.wait_for_function('document.querySelector("video").readyState>=2&&!document.querySelector("video").error')
    page.evaluate('window.__qwenSourceVideo=document.querySelector("video")')
    expect(page.get_by_test_id('active-correction-model').first).to_contain_text('Qwen3.5-4B')
    panel = page.get_by_test_id('model-upgrade-experiment')
    assert panel.evaluate('el=>!el.open')
    page.get_by_test_id('model-upgrade-toggle').click()
    expect(panel).to_contain_text('字错率未评估')
    for key in ('baseline', 'qwen3_asr', 'correction'):
        expect(page.get_by_test_id('model-upgrade-lane-'+key)).to_be_visible()
    page.get_by_test_id('model-upgrade-all-toggle').click()
    for key, run in runs.items():
        assert page.locator(f'[data-testid^="model-upgrade-{key}-T"]').count() == len(run['segments'])
    assert page.locator('[data-testid^="model-upgrade-correction-T"]').count() == len(proof['segments'])
    row = runs['qwen3_asr']['segments'][min(2, len(runs['qwen3_asr']['segments'])-1)]
    page.get_by_test_id('model-upgrade-qwen3_asr-'+row['id']).click()
    page.wait_for_function('(t)=>Math.abs(document.querySelector("video").currentTime-t)<.01', arg=row['start_sec'])
    assert page.evaluate('document.querySelector("video")===window.__qwenSourceVideo')
    page.get_by_test_id('model-upgrade-all-toggle').click()
    for key, run in runs.items():
        expected = [r for r in run['segments'] if r['start_sec'] <= row['start_sec'] < r['end_sec']]
        assert page.locator(f'[data-testid^="model-upgrade-{key}-T"]').count() == len(expected)
        for segment in expected:
            expect(page.get_by_test_id('model-upgrade-'+key+'-'+segment['id'])).to_contain_text(segment['text'])
    panel.screenshot(path=str(args.output/'same-clock-three-lanes.png'))
    checks.append('Actual old/new recognition use their own complete native segments and source clock; clicking new text seeks persistent video')
    page.set_viewport_size({'width': 390, 'height': 844})
    page.wait_for_function('document.documentElement.scrollWidth<=innerWidth+1')
    panel.screenshot(path=str(args.output/'model-upgrade-mobile.png'))
    page.locator('nav [data-view="report"]').click()
    with page.expect_download() as download:
        page.get_by_role('button', name='完整数据', exact=True).click()
    destination = args.output/'lesson-export.json'
    download.value.save_as(destination)
    assert json.loads(destination.read_text(encoding='utf-8')) == fetch()
    assert page.evaluate('document.querySelector("video")===window.__qwenSourceVideo')
    checks.append('Independent report view exports exact published experiment, ASR, correction and behavior; 390px layout fits')
    assert not errors, errors
    browser.close()
report = {'passed': True, 'scope': 'actual_models_api_media_and_browser_no_mocks', 'lesson_id': args.lesson,
          'checks': checks, 'page_errors': errors,
          'native_segment_counts': {key: len(run['segments']) for key, run in runs.items()},
          'accepted_edits': proof['edit_count']}
(args.output/'verification.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(report, ensure_ascii=False))
