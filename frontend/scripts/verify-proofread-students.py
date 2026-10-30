"""Acceptance against real lesson data, generated proofreading and actual video."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

from playwright.sync_api import sync_playwright, expect


p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--url', default='http://127.0.0.1:5175')
p.add_argument('--lesson', default='unified-chemistry-001-9')
p.add_argument('--output', type=Path, default=Path('output/playwright/proofread-students-real'))
args = p.parse_args()
args.output.mkdir(parents=True, exist_ok=True)


def get_data():
    with urllib.request.urlopen(args.url + '/api/review/lessons/' + args.lesson) as response:
        return json.load(response)


data = get_data()
source = data['lesson']['source_video']['sha256']
proof = data['transcript_proofreading']
assert proof['human_verified'] is False
assert proof['source_video_sha256'] == source
original = {r['id']: r for r in data['transcript']}
corrected = {r['transcript_id']: r for r in proof['segments']}
assert set(original) == set(corrected)
for key, raw in original.items():
    revised = corrected[key]
    assert revised['original_text'] == raw['text']
    assert revised['start_sec'] == raw['start_sec'] and revised['end_sec'] == raw['end_sec']
    assert revised['text'].strip()
changed = [r for r in proof['segments'] if r['text'] != r['original_text']]
assert changed, 'Generation must actually proofread the sample, rather than only flag it'
assert len(data['transcript_unguided']) == 36
activity = data['activity_analysis']
assert activity['source_sha256'] == source
states = set()
for frame in activity['frames']:
    parts = frame['participants']
    assert sum(frame['student_state_counts'].values()) == frame['detected_people']
    states.update(frame['student_state_counts'])
    participation = frame['participation']
    assert participation['total_people'] == frame['detected_people']
    if not frame['detected_people']:
        assert participation['index'] is None and participation['active_fraction'] is None
    # Pose alone does not establish a reading/writing/hand-raising action score.
    if participation['scored_people'] == 0:
        assert participation['index'] is None
assert len(states) > 4, f'Need observable states beyond three legacy actions and unknown: {states}'

backend = Path(__file__).resolve().parents[2] / 'backend'
baseline = json.loads((backend / 'artifacts/verification/proofread-students-20261027/source-before.json').read_text(encoding='utf-8'))
for relative, expected in baseline.items():
    assert hashlib.sha256((backend / relative).read_bytes()).hexdigest() == expected, relative

checks = ['Original guided/unguided ASR and audio bytes preserved; generated corrections retain original segment times and source hash',
          'Expanded state counts refer to actual same-video people; action-only scores and no-detection unknown are consistent']
errors = []
with sync_playwright() as pw:
    browser = pw.chromium.launch()
    page = browser.new_page(viewport={'width': 1500, 'height': 1040}, device_scale_factor=1)
    page.on('pageerror', lambda error: errors.append(str(error)))
    for old in ['snapshots=1', 'demo=1', 'data=legacy.json']:
        page.goto(args.url + '/?' + old + '&lesson=' + args.lesson, wait_until='networkidle')
        expect(page.locator('[data-active-view]')).to_have_attribute('data-active-view', 'review')
        assert page.locator('nav [data-view]').count() == 5
        assert page.get_by_text('Behavior snapshot workspace', exact=True).count() == 0
        assert page.locator('a[href*="snapshots="]').count() == 0
    checks.append('All former snapshot/demo/data query routes render the modern workspace; legacy workbench link is absent')
    page.wait_for_function('document.querySelector("video").readyState>=2')
    page.evaluate('window.__sourceVideo=document.querySelector("video")')

    def seek(at):
        page.get_by_label('Whole-lesson playback progress').evaluate('(el,t)=>{el.value=String(t);el.dispatchEvent(new Event("input",{bubbles:true}))}', at)
        page.wait_for_function('(t)=>Math.abs(document.querySelector("video").currentTime-t)<.1&&!document.querySelector("video").seeking', arg=at)

    def view(name):
        page.locator('nav [data-view="' + name + '"]').click()
        expect(page.locator('[data-active-view]')).to_have_attribute('data-active-view', name)
        assert page.locator('video').count() == 1
        assert page.evaluate('document.querySelector("video")===window.__sourceVideo')

    at = (changed[0]['start_sec'] + changed[0]['end_sec']) / 2
    seek(at)
    expect(page.get_by_test_id('overview-speech')).to_contain_text(changed[0]['text'])
    view('evidence')
    expect(page.get_by_test_id('transcript-proofread-default').first).to_be_visible()
    raw_toggle = page.get_by_test_id('transcript-raw-toggle').first
    raw_toggle.click()
    expect(page.get_by_test_id('transcript-raw-comparison').first).to_be_visible()
    page.screenshot(path=str(args.output / 'corrected-and-original-transcript.png'), full_page=True)
    checks.append('Overview and transcript default to generated proofreading; original ASR is available for comparison')

    view('review')
    seek(0)
    action_jump = page.get_by_test_id('locate-participation-actions')
    expect(action_jump).to_be_visible()
    action_jump.click()
    page.wait_for_function('document.querySelector("video").currentTime>0&&!document.querySelector("video").seeking')
    expect(page.get_by_test_id('visible-participation')).not_to_contain_text('Unknown')
    checks.append('An unknown opening posture links to an actual supported participation-action frame without inventing an opening score')
    for at in [0, 40, 44.5, 60]:
        seek(at)
        panel = page.get_by_test_id('current-student-state')
        expect(panel).to_be_visible()
        frame = next(f for f in activity['frames'] if f['start_sec'] <= at < f['end_sec'])
        image = panel.locator('.annotated-frame img')
        expect(image).to_be_visible()
        assert image.evaluate('im=>im.complete&&im.naturalWidth>0')
        expect(page.get_by_test_id('student-state-counts')).to_be_visible()
        if frame['participation']['index'] is None:
            expect(page.get_by_test_id('visible-participation')).to_contain_text('Unknown')
        else:
            expect(page.get_by_test_id('visible-participation')).to_contain_text(f"{frame['participation']['index']:.1f}")
        if frame['participants']:
            expect(page.get_by_test_id('student-state-boxes')).to_be_visible()
        page.screenshot(path=str(args.output / f'overview-current-students-{at}s.png'))
    checks.append('Current source-frame image, state boxes, counts and participation reference follow playback; projection-only view stays unknown')
    seek(40)
    view('behavior')
    expect(page.get_by_test_id('current-student-state')).to_be_visible()
    diagnostics = page.get_by_test_id('student-diagnostics')
    assert diagnostics.evaluate('el=>!el.open')
    diagnostics.locator(':scope > summary').click()
    assert diagnostics.evaluate('el=>el.open')
    page.screenshot(path=str(args.output / 'student-page-current-and-diagnostics.png'), full_page=True)
    diagnostics.locator(':scope > summary').click()
    page.set_viewport_size({'width': 390, 'height': 844})
    page.wait_for_function('document.documentElement.scrollWidth<=innerWidth+1')
    page.screenshot(path=str(args.output / 'student-mobile.png'), full_page=True)
    checks.append('Behavior page prioritizes current students; cumulative engineering counts are collapsed diagnostics; mobile has no horizontal overflow')
    view('report')
    with page.expect_download() as event:
        page.get_by_role('button', name='Full data', exact=True).click()
    destination = args.output / 'full-lesson.json'
    event.value.save_as(destination)
    assert json.loads(destination.read_text(encoding='utf-8')) == get_data()
    assert not errors, errors
    browser.close()

report = {'passed': True, 'scope': 'real_api_real_media_no_mocks', 'lesson_id': args.lesson,
          'source_sha256': source, 'corrected_segments': len(changed), 'observed_state_labels': sorted(states),
          'checks': checks, 'page_errors': errors}
(args.output / 'verification.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(report, ensure_ascii=False, indent=2))
