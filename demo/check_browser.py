"""Exercise the project page and guide in Chromium against a static server."""
from pathlib import Path
from urllib.parse import urljoin
import argparse
import json
from playwright.sync_api import sync_playwright

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("url")
parser.add_argument("--screenshots", type=Path)
args = parser.parse_args()
root_url = args.url.rstrip("/") + "/"
guide_url = urljoin(root_url, "guide/")
if args.screenshots:
    args.screenshots.mkdir(parents=True, exist_ok=True)

expected = {"supported": [True, True, True], "site_confounded": [False, False, False],
            "nonreplicating": [True, False, False]}
with sync_playwright() as p:
    browser = p.chromium.launch(headless=True, args=["--no-sandbox"])
    context = browser.new_context(viewport={"width": 1440, "height": 1000}, device_scale_factor=1)
    page = context.new_page()
    errors = []
    page.on("pageerror", lambda error: errors.append(str(error)))
    page.on("response", lambda response: errors.append(f"HTTP {response.status}: {response.url}")
            if response.url.startswith(root_url) and response.status >= 400 else None)
    response = page.goto(root_url)
    assert response.status == 200
    assert page.locator('h1').inner_text().replace('\n', ' ').startswith('CONFIRM: Governing Scientific Reporting')
    assert page.locator('.authors > span').count() == 5
    assert page.locator('figure img').count() == 4
    for figure in page.locator('figure img').all():
        figure.scroll_into_view_if_needed()
        figure.evaluate("img => img.decode()")
        assert figure.get_attribute('alt')
    results = context.request.get(urljoin(root_url, 'paper-data/results.json')).json()
    rows = page.locator('#followup-results tbody tr').all()
    for row, record in zip(rows, results['followup'], strict=True):
        assert row.locator('th,td').all_inner_texts() == [record['method'], f"{record['llm_calls']:,}",
            f"{record['candidates_evaluated']:,}", str(record['supported_on_search']), f"{record['holdout_percent']}%"]
    for width in [1440, 1024, 768, 390, 360]:
        page.set_viewport_size({"width": width, "height": 1000})
        assert page.evaluate('document.documentElement.scrollWidth <= innerWidth'), ('project', width)
        if args.screenshots and width in (1440, 390):
            page.screenshot(path=str(args.screenshots / f'project-{width}.png'), full_page=True)
    page.set_viewport_size({"width": 1440, "height": 1000})
    page.locator('a.button[href="guide/"]').first.click()
    page.wait_for_url(guide_url)
    page.locator('nav a[href="../"]').click()
    page.wait_for_url(root_url)
    page.goto(root_url + '?example=supported&step=4&standard=full#explore')
    page.wait_for_url(guide_url + '?example=supported&step=4&standard=full#explore')
    page.locator('.decision-card h3').wait_for()
    assert page.locator('.decision-card h3').inner_text() == 'Supported'
    assert page.locator('[data-step="4"]').get_attribute('aria-current') == 'step'
    page.goto(guide_url)
    page.locator('[data-case="nonreplicating"]').wait_for()
    if args.screenshots:
        page.screenshot(path=str(args.screenshots / "guide-desktop.png"), full_page=True)
    for case, decisions in expected.items():
        page.locator(f'[data-case="{case}"]').click()
        for step in range(5):
            page.locator(f'[data-step="{step}"]').click()
            assert page.locator('#stage-content').inner_text().strip()
        for standard, supported in zip(["screen", "replicate", "full"], decisions):
            page.locator(f'[data-standard="{standard}"]').click()
            assert page.locator('.decision-card h3').inner_text() == ("Supported" if supported else "Withheld")
        with page.expect_download() as download:
            page.locator('#record-link').click()
        record = json.loads(Path(download.value.path()).read_text())
        assert record["example"] == case
        page.locator('[data-step="3"]').click()
        assert page.locator('.gate-row').count() == 5
        page.locator('.gate-row summary').first.click()
    for step in range(4):
        page.locator(f'[data-replay="{step}"]').click()
        assert page.locator('#replay-content').inner_text().strip()
    assert page.locator('#replay-content .status').all_inner_texts() == ["Withheld", "Withheld"]
    page.locator('#replay-next').click()
    assert page.locator('[data-replay="0"]').get_attribute('aria-current') == 'step'
    for width in [1440, 1024, 768, 390, 360]:
        page.set_viewport_size({"width": width, "height": 900})
        for step in range(5):
            page.locator(f'[data-step="{step}"]').click()
            assert page.evaluate('document.documentElement.scrollWidth <= innerWidth'), (width, step)
        if args.screenshots and width == 390:
            page.locator('[data-step="3"]').click()
            page.screenshot(path=str(args.screenshots / "guide-mobile.png"), full_page=True)
    page.set_viewport_size({"width": 1440, "height": 1000})
    page.locator('[data-step="2"]').click()
    if args.screenshots:
        page.locator('#workspace').screenshot(path=str(args.screenshots / "results.png"))
    page.locator('[data-step="3"]').click()
    if args.screenshots:
        page.locator('#workspace').screenshot(path=str(args.screenshots / "checks.png"))
    page.reload()
    page.locator('[data-case="nonreplicating"]').wait_for()
    assert page.locator('[data-step="3"]').get_attribute('aria-current') == 'step'
    page.locator('[data-case="supported"]').focus()
    page.keyboard.press('Space')
    assert page.locator('[data-case="supported"]').get_attribute('aria-pressed') == 'true'
    page.locator('[data-step="4"]').focus()
    page.keyboard.press('Enter')
    assert page.locator('.decision-card h3').inner_text() == 'Supported'
    page.locator('#copy-command').focus()
    page.keyboard.press('Enter')
    page.wait_for_function("document.querySelector('#copy-status').textContent.startsWith('Command')")
    assert not errors, errors
    context.close()
    plain = browser.new_context(java_script_enabled=False)
    plain_page = plain.new_page()
    plain_page.goto(root_url)
    assert plain_page.locator('#paper-title').is_visible()
    assert plain_page.locator('#followup-results tbody tr').count() == 3
    assert plain_page.locator('.result-number strong').all_inner_texts() == ['0', '15']
    assert plain_page.locator('figure img').count() == 4
    plain.close()
    browser.close()
print('Browser checks passed: project figures and results, page navigation, legacy links, no-JavaScript homepage, all guide stages, nine decisions, downloads, follow-up playback, keyboard controls, and five viewport widths.')
