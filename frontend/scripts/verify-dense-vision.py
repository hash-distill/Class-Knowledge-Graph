"""Read-only browser acceptance of the actual dense classroom observation package."""
import argparse
import json
import re
from pathlib import Path
from urllib.request import urlopen
from playwright.sync_api import sync_playwright, expect


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='http://127.0.0.1:18094')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    lesson = 'unified-chemistry-001-9'
    with urlopen(f'{args.url}/api/review/lessons/{lesson}') as response:
        package = json.load(response)
    frames = package['activity_analysis']['frames']
    expected = {time: min(frames, key=lambda f: abs(f['timestamp_sec'] - time)) for time in [0, 40, 45, 75, 89.9]}
    errors = []
    checks = []
    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        page = browser.new_page(viewport={'width': 1440, 'height': 1080})
        page.on('pageerror', lambda error: errors.append(str(error)))
        page.goto(f'{args.url}/?view=behavior&lesson={lesson}', wait_until='networkidle')
        expect(page.get_by_role('heading', name='Student behavior analysis', exact=True)).to_be_visible()
        expect(page.get_by_test_id('state-candidate-notice')).to_contain_text('Not yet reviewed against human annotations for this lesson')
        expect(page.get_by_test_id('student-state-boxes').locator('g')).to_have_count(expected[0]['detected_people'])
        checks.append('Actual API package renders all retained current detection boxes')
        selected = next(i for i, person in enumerate(expected[0]['participants']) if person['source'] == 'trained_classifier')
        page.get_by_test_id('student-state-boxes').locator('g').nth(selected).click()
        expect(page.get_by_test_id('selected-observation')).to_contain_text('Per-person state classification')
        page.get_by_role('button', name='Boxes only', exact=True).click()
        expect(page.get_by_test_id('student-state-boxes').locator('text')).to_have_count(0)
        page.get_by_role('button', name='State labels', exact=True).click()
        expect(page.get_by_test_id('student-state-boxes').locator('text')).to_have_count(expected[0]['detected_people'])
        page.get_by_role('button', name='Fullscreen review ↗', exact=True).click()
        expect(page.locator('.annotated-frame:fullscreen')).to_be_visible()
        page.evaluate('document.exitFullscreen()')
        checks.append('Source labels, annotation modes and fullscreen work')
        for time, frame in expected.items():
            page.get_by_role('slider', name='Whole-lesson playback progress').evaluate('(el,time)=>{el.value=String(time);el.dispatchEvent(new Event("input",{bubbles:true}))}', time)
            expect(page.get_by_test_id('student-state-boxes').locator('g')).to_have_count(frame['detected_people'])
            score = page.get_by_test_id('overall-engagement-index')
            expect(score).to_have_text(re.compile(r'^\d+\.\d$') if frame['detected_people'] else 'No observations')
            checks.append({'source_time': time, 'detected_people': frame['detected_people'], 'overall_score': score.inner_text()})
        page.get_by_role('slider', name='Whole-lesson playback progress').evaluate('(el)=>{el.value="75";el.dispatchEvent(new Event("input",{bubbles:true}))}')
        expect(page.get_by_test_id('visible-state-timeline')).to_be_visible()
        expect(page.get_by_test_id('visible-state-timeline').locator('.state-timeline>button')).to_have_count(len(frames))
        page.screenshot(path=str(args.output/'behavior-desktop.png'), full_page=True)
        page.get_by_role('tab', name='Person heatmap', exact=True).click()
        expect(page.locator('.track-table')).to_be_visible()
        page.get_by_role('button', name='Temporal action score', exact=True).click()
        page.get_by_role('button', name='Visible states', exact=True).click()
        checks.append('Categorical state heatmap and legacy temporal score mode remain separate')
        page.get_by_role('tab', name='Models and limitations', exact=True).click()
        expect(page.locator('.behavior-provenance')).to_contain_text('3.15 MB')
        expect(page.locator('.behavior-provenance')).to_contain_text('same classrooms, Adjacent times')
        expect(page.locator('.behavior-provenance')).to_contain_text('0.8169403180754294')
        checks.append('Model size, actual crop metric and split-overlap boundary are visible')
        page.goto(f'{args.url}/?view=review&lesson={lesson}', wait_until='networkidle')
        expect(page.get_by_test_id('state-candidate-notice')).to_be_visible()
        checks.append('Overview also labels the state classifier as unvalidated candidate')
        page.goto(f'{args.url}/?view=behavior&lesson={lesson}', wait_until='networkidle')
        page.set_viewport_size({'width': 390, 'height': 844})
        expect(page.get_by_role('heading', name='Student behavior analysis', exact=True)).to_be_visible()
        assert page.evaluate('document.documentElement.scrollWidth <= innerWidth + 1'), 'Mobile page overflows horizontally'
        page.screenshot(path=str(args.output/'behavior-mobile.png'), full_page=True)
        assert not errors, errors
        browser.close()
    report = {'status': 'passed', 'url': args.url, 'read_only': True, 'checks': checks, 'page_errors': errors,
              'scope': 'Rendering and interactions for the actual model artifact; not recognition accuracy or independent classroom validation'}
    (args.output/'acceptance.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    main()
