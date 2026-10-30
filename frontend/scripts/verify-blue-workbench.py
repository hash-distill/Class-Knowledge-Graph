"""Read-only visual acceptance of six real classroom workspaces using Playwright."""
import argparse
import json
from pathlib import Path
from playwright.sync_api import sync_playwright, expect

VIEWS = ['review', 'evidence', 'behavior', 'knowledge', 'report', 'teaching']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='http://127.0.0.1:18093')
    parser.add_argument('--lesson', default='unified-chemistry-001-9')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    errors, samples = [], []
    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        page = browser.new_page(viewport={'width': 1440, 'height': 1000})
        page.on('pageerror', lambda error: errors.append(str(error)))
        page.goto(args.url.rstrip('/') + f'/?view=review&lesson={args.lesson}', wait_until='networkidle')
        page.wait_for_function('document.querySelector("video")?.readyState >= 2')
        page.evaluate('window.__blueVideo = document.querySelector("video"); window.__blueVideo.currentTime = 5')
        page.wait_for_function('Math.abs(document.querySelector("video").currentTime - 5) < .2')

        def navigate(view):
            page.locator(f'.review-sidebar [data-view="{view}"]').click()
            expect(page.locator(f'[data-view-panel="{view}"]')).to_be_visible()
            page.wait_for_function('!document.querySelector(".workspace-view-enter-active, .workspace-view-leave-active")')
            if view == 'teaching':
                expect(page.get_by_test_id('teacher-goals')).to_be_enabled()
            assert page.evaluate('document.querySelector("video") === window.__blueVideo'), view
            assert page.locator('video').count() == 1
            assert abs(page.locator('video').evaluate('(video) => video.currentTime') - 5) < .2

        for width, height in [(1440, 1000), (1280, 800), (1024, 800), (768, 1024), (390, 844)]:
            page.set_viewport_size({'width': width, 'height': height})
            for view in VIEWS:
                navigate(view)
                layout = page.evaluate('''() => {
                    const player = document.querySelector('.shared-player').getBoundingClientRect();
                    const feature = document.querySelector('.active-feature').getBoundingClientRect();
                    const visible = el => !!(el.offsetWidth || el.offsetHeight || el.getClientRects().length);
                    const headings = [...document.querySelectorAll('h1,h2,h3,h4,h5,h6')].filter(visible).map(el => el.textContent.trim());
                    const green = [];
                    for (const el of document.querySelectorAll('.review-workbench *')) {
                        if (!visible(el)) continue;
                        const style = getComputedStyle(el);
                        for (const prop of ['color','backgroundColor','borderTopColor','fill','stroke']) {
                            const match = style[prop]?.match(/^rgba?\\((\\d+), (\\d+), (\\d+)(?:, ([\\d.]+))?\\)$/);
                            if (!match || (match[4] !== undefined && Number(match[4]) < .05)) continue;
                            const [r,g,b] = match.slice(1,4).map(Number);
                            if (g > r + 10 && g > b + 5) green.push({tag:el.tagName,cls:el.className.baseVal ?? el.className,prop,color:style[prop]});
                        }
                    }
                    return {topDelta:Math.abs(player.top-feature.top),bottomDelta:Math.abs(player.bottom-feature.bottom),
                        stacked:player.bottom<=feature.top, documentWidth:document.documentElement.scrollWidth,
                        viewportWidth:innerWidth, headings,green, playerHeight:player.height,featureHeight:feature.height};
                }''')
                assert layout['documentWidth'] <= width + 1, (width, view, 'horizontal overflow', layout)
                if width > 1000:
                    assert layout['topDelta'] <= 1 and layout['bottomDelta'] <= 1, (width, view, 'columns unaligned', layout)
                else:
                    assert layout['stacked'], (width, view, 'responsive ordering', layout)
                assert not layout['green'], (width, view, layout['green'][:5])
                assert page.locator('h1').inner_text()[-1] not in '。．.!！?？:：;；,，、…·—–-'
                for heading in layout['headings']:
                    assert not heading or heading[-1] not in '。．.!！?？:：;；,，、…·—–-', (width, view, heading)
                samples.append({'width': width, 'view': view, **layout})
                if width in (1440, 390):
                    page.screenshot(path=str(args.output / f'{view}-{width}.png'), full_page=True)

        page.set_viewport_size({'width': 1440, 'height': 1000})
        navigate('teaching')
        original_goal = page.get_by_test_id('teacher-goals').input_value()
        page.get_by_test_id('teacher-goals').fill('界面验收未保存草稿')
        page.locator('[data-teaching-tab="actions"]').click()
        expect(page.get_by_test_id('teacher-action-title')).to_be_visible()
        expect(page.get_by_test_id('teacher-goals')).not_to_be_visible()
        page.locator('[data-teaching-tab="practice"]').click()
        expect(page.get_by_test_id('teacher-assessment-question')).to_be_visible()
        page.locator('[data-teaching-tab="records"]').click()
        expect(page.get_by_test_id('teacher-task-list')).to_be_visible()
        page.locator('[data-teaching-tab="review"]').click()
        expect(page.get_by_test_id('teacher-goals')).to_have_value('界面验收未保存草稿')
        navigate('knowledge')
        navigate('teaching')
        expect(page.get_by_test_id('teacher-goals')).to_have_value('界面验收未保存草稿')
        page.get_by_test_id('teacher-goals').fill(original_goal)

        page.get_by_role('button', name='复盘助手', exact=True).click()
        expect(page.get_by_role('complementary', name='课堂复盘助手')).to_be_visible()
        assert page.get_by_role('heading', name='从一个课堂问题开始', exact=True).count() == 1
        page.get_by_role('button', name='关闭复盘助手').click()
        page.emulate_media(reduced_motion='reduce')
        motion = page.locator('.review-page-heading').evaluate('(el) => getComputedStyle(el).animationDuration')
        assert all(float(value.strip().removesuffix('s')) <= .001 for value in motion.split(',')), motion
        browser.close()

    assert not errors, errors
    report = {'status': 'passed', 'samples': samples, 'page_errors': errors,
              'checks': ['30 real view/viewport combinations', 'equal desktop column tops and bottoms',
                         'responsive stacking', 'no horizontal overflow', 'no green interface colors',
                         'no trailing heading punctuation', 'one stable video and playback time',
                         'four teacher sections reachable with drafts retained', 'assistant and reduced motion'],
              'scope': 'Read-only browser acceptance; no model evaluation or teacher API mutations'}
    (args.output / 'visual-acceptance.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'status': report['status'], 'samples': len(samples), 'checks': report['checks']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
