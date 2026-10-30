"""Write workflow acceptance ONLY against an explicitly isolated preview state."""
import argparse
import io
import json
from pathlib import Path
import zipfile
from playwright.sync_api import sync_playwright, expect


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='http://127.0.0.1:18094')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.url.rstrip('/') != 'http://127.0.0.1:18094':
        raise SystemExit('This mutation test must not run against the real workbench database')
    args.output.mkdir(parents=True, exist_ok=True)
    lesson = 'unified-chemistry-001-9'
    endpoint = f'{args.url}/api/review/lessons/{lesson}/vision-review/frames/V0001'
    checks = []; errors = []
    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        page = browser.new_page(viewport={'width': 1600, 'height': 1100})
        page.on('pageerror', lambda reason: errors.append(str(reason)))
        page.goto(f'{args.url}/?view=behavior&lesson={lesson}', wait_until='networkidle')
        # Reset only our owned, isolated acceptance database so a failed run is repeatable.
        prior = page.request.get(endpoint).json()
        if prior['annotations']:
            cleared = page.request.post(endpoint, data={'expected_version': prior['version'],
                'basis_fingerprint': prior['basis_fingerprint'], 'annotations': []})
            assert cleared.ok
        panel = page.get_by_test_id('vision-review')
        panel.get_by_role('button', name='Start reviewing the current frame').click()
        canvas = panel.get_by_test_id('vision-review-canvas')
        expect(canvas).to_be_visible()
        model_count = canvas.locator('g').count()
        assert model_count >= 30
        # Explicit role/label edits are synthetic browser acceptance actions.
        panel.get_by_test_id('review-role').select_option('student')
        panel.get_by_test_id('save-vision-review').click()
        expect(panel.get_by_role('alert')).to_contain_text('not been confirmed')
        assert page.request.get(endpoint).json()['annotations'] == []
        panel.get_by_test_id('review-label').select_option('read')
        panel.locator('.object-list button').nth(1).click()
        panel.get_by_placeholder('Explain the reason for exclusion. For example, a mirror duplicate').fill('Automated verification sample. Exclusion flow test only')
        panel.get_by_role('button', name='Exclude this person', exact=True).click()

        def draw():
            panel.get_by_role('button', name='＋ Draw a missed person', exact=True).click()
            box = canvas.bounding_box()
            page.mouse.move(box['x']+box['width']*.80, box['y']+box['height']*.70)
            page.mouse.down()
            page.mouse.move(box['x']+box['width']*.90, box['y']+box['height']*.85, steps=8)
            page.mouse.up()

        draw()
        expect(canvas.locator('g')).to_have_count(model_count+1)
        panel.get_by_test_id('review-role').select_option('student')
        panel.get_by_test_id('review-label').select_option('raise_hand')
        panel.get_by_test_id('save-vision-review').click()
        expect(panel.get_by_role('status')).to_contain_text('Saved version')
        saved = page.request.get(endpoint).json()
        assert len(saved['annotations']) == 3
        assert sum(row['status'] == 'ignored' for row in saved['annotations']) == 1
        manual = next(row for row in saved['annotations'] if row['origin'] == 'manual')
        assert 2 <= manual['bbox'][2]-manual['bbox'][0] <= 130
        checks.append('Edit label and role, exclude object, draw missed object in source pixel coordinates, persist version')
        panel.get_by_role('button', name='Compare source image', exact=True).click()
        expect(canvas).to_have_count(0)
        panel.get_by_role('button', name='Show review annotations', exact=True).click()
        canvas = panel.get_by_test_id('vision-review-canvas')
        # A cancelled manual box must never be resurrected by the draft cache.
        draw()
        panel.get_by_role('button', name='Delete added box', exact=True).click()
        expect(panel.get_by_test_id('save-vision-review')).to_be_disabled()
        panel.get_by_role('button', name='Load current frame', exact=True).click()
        expect(canvas.locator('g')).to_have_count(model_count+1)
        panel.locator('.object-list button').nth(2).click()
        panel.get_by_role('button', name='Confirm this person', exact=True).click()
        panel.get_by_role('button', name='Restore unreviewed state', exact=True).click()
        expect(panel.get_by_test_id('save-vision-review')).to_be_disabled()
        panel.get_by_role('button', name='Load current frame', exact=True).click()
        expect(panel.locator('.object-list button').nth(2)).to_have_class('unreviewed')
        checks.append('Cancelled boxes and restored unreviewed candidates do not resurrect as confirmed drafts')
        # View switches preserve real unsaved draft edits, without changing model outputs.
        panel.locator('.object-list button').first.click()
        panel.get_by_test_id('review-label').select_option('write')
        page.get_by_role('tab', name='Models and limitations', exact=True).click()
        page.get_by_role('tab', name='Student states and engagement', exact=True).click()
        panel.get_by_role('button', name='Start reviewing the current frame').click()
        expect(panel.get_by_test_id('review-label')).to_have_value('write')
        expect(panel.get_by_role('status')).to_contain_text('Restored the unsaved draft for this frame')
        panel.get_by_test_id('save-vision-review').click()
        expect(panel.get_by_role('status')).to_contain_text('Saved version')
        checks.append('Unsaved drafts survive view switch')

        panel.get_by_role('button', name='Revoke this frame review', exact=True).click()
        expect(panel.get_by_role('status')).to_contain_text('Revoked')
        assert page.request.get(endpoint).json()['annotations'] == []
        panel.get_by_role('button', name='Version history', exact=True).click()
        panel.locator('.review-history>div').nth(1).get_by_role('button', name='Restore this version').click()
        expect(panel.get_by_role('status')).to_contain_text('Restored historical content')
        checks.append('Withdraw current frame review and restore historical version')

        other = browser.new_page(viewport={'width': 1440, 'height': 1080})
        other.goto(page.url, wait_until='networkidle')
        other_panel = other.get_by_test_id('vision-review')
        other_panel.get_by_role('button', name='Start reviewing the current frame').click()
        expect(other_panel.get_by_test_id('vision-review-canvas')).to_be_visible()
        panel.locator('.object-list button').first.click()
        panel.get_by_test_id('review-label').select_option('read')
        panel.get_by_test_id('save-vision-review').click()
        expect(panel.get_by_role('status')).to_contain_text('Saved version')
        other_panel.get_by_test_id('review-label').select_option('discuss')
        other_panel.get_by_test_id('save-vision-review').click()
        expect(other_panel.get_by_role('alert')).to_contain_text('updated in another window')
        expect(other_panel.get_by_test_id('review-label')).to_have_value('discuss')
        other_panel.get_by_role('button', name='Reload records', exact=True).click()
        expect(other_panel.get_by_role('status')).to_contain_text('Loaded the server version')
        expect(other_panel.get_by_test_id('review-label')).to_have_value('read')
        with other.expect_download() as draft_download:
            other_panel.get_by_role('button', name='Draft backup', exact=True).click()
        backup = json.loads(Path(draft_download.value.path()).read_text(encoding='utf-8'))
        assert backup['kind'] == 'unverified_draft_backup' and backup['rows'][0]['label'] == 'discuss'
        checks.append('Concurrent stale save rejected, edits retained, server reload keeps downloadable old draft')

        with page.expect_download() as download:
            panel.get_by_test_id('export-vision-samples').click()
        archive_path = args.output/'automated-fixture-only.zip'
        download.value.save_as(archive_path)
        with zipfile.ZipFile(archive_path) as archive:
            manifest = json.loads(archive.read('manifest.json'))
            assert manifest['classification_sample_count'] == 2
            assert len([path for path in archive.namelist() if path.startswith('crops/')]) == 2
            assert manifest['partial_annotations']
        checks.append('ZIP includes only two explicitly confirmed student classification crops, provenance, partial annotation boundary')
        panel.get_by_role('button', name='Enlarge for review', exact=True).click()
        expect(page.locator(':fullscreen')).to_have_attribute('data-testid', 'vision-review')
        page.screenshot(path=str(args.output/'review-fullscreen.png'), full_page=True)
        panel.get_by_role('button', name='Exit enlarged view', exact=True).click()
        page.screenshot(path=str(args.output/'desktop.png'), full_page=True)
        for width in [1024, 390]:
            page.set_viewport_size({'width': width, 'height': 1000})
            page.wait_for_timeout(150)
            assert page.evaluate('document.documentElement.scrollWidth <= window.innerWidth'), f'horizontal overflow {width}'
            assert panel.get_by_test_id('vision-review-canvas').bounding_box()['width'] > 150
        page.screenshot(path=str(args.output/'mobile.png'), full_page=True)
        checks.append('Desktop, tablet and mobile layouts fit viewport')
        assert errors == [], errors
        browser.close()
    (args.output/'acceptance.json').write_text(json.dumps({'url': args.url, 'isolated_automated_fixture': True,
        'checks': checks, 'page_errors': errors}, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'passed': len(checks), 'page_errors': errors}, ensure_ascii=False))


if __name__ == '__main__': main()
