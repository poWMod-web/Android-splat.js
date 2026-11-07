// The reported flow, end to end: train a scene, take a link WITHOUT publishing
// it, then come back to it. Two things must hold — the choice is never final
// (the switch is on the card that just made the link), and a scene that was
// shared still knows it after a reload (it offers its link, not a Download).
import { test, expect } from '@playwright/test';
import { seedCapture, startCaptureRun } from './helpers.mjs';

const SPACE = '42485456_7777';

function stubPlatform(page, { mine = [] } = {}) {
  const puts = [];
  page.route(/api-live\.arrival\.space/, async (route) => {
    const url = route.request().url();
    const method = route.request().method();
    const json = (o) => route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(o) });
    if (/\/user\/me$/.test(url)) return json({ status: 'ok', data: { id: '42485456', name: 'Tester' } });
    if (/\/splatjs\/mine$/.test(url)) return json({ status: 'success', data: { items: mine } });
    if (/\/splatjs\/gallery/.test(url)) return json({ status: 'success', data: { items: [], nextBefore: null } });
    if (/\/files\/upload$/.test(url)) return json({ status: 'success', data: { params: { method: 'PUT', url: 'https://cdn.test/f?sig=1', headers: {} } } });
    if (/\/files\/upload-complete$/.test(url)) return json({ status: 'success', data: { resource_key: 'rk', url: 'https://cdn.test/f.bin' } });
    if (/\/user\/create-space$/.test(url)) return json({ status: 'success', data: { space_url: `https://arrival.space/${SPACE}` } });
    if (/\/api\/v1\/spaces\//.test(url) && method === 'PUT') {
      puts.push(JSON.parse(route.request().postData() || '{}'));
      return json({ status: 'success', data: {} });
    }
    return json({ status: 'success', data: {} });
  });
  page.route(/cdn\.test/, (route) => route.fulfill({ status: 200, body: '' }));
  return puts;
}

test('a link-only share can still be published, and survives a reload', async ({ page }) => {
  const puts = stubPlatform(page);

  await seedCapture(page, '?iters=1200');
  await page.evaluate(() => localStorage.setItem('arrival_token:https://api-live.arrival.space', 'test-token'));
  await startCaptureRun(page);
  await page.waitForFunction(() => window.__splat.state === 'done', null, { timeout: 240_000 });

  await page.click('.exportwrap button');
  await page.waitForSelector('#sh-go');
  await page.uncheck('#sh-priv');
  await page.click('#sh-go');

  // the result card carries the switch, off — the choice is not final
  await expect(page.locator('#sh-done')).toBeVisible({ timeout: 120_000 });
  await expect(page.locator('#sh-done-manage')).toBeVisible();
  expect(await page.isChecked('#sh-done-listing')).toBe(false);

  // and publishing from right there works
  await page.check('#sh-done-listing');
  await expect.poll(() => puts.filter((p) => p.privacy === 'Open').length, { timeout: 15_000 }).toBe(1);
  expect(await page.evaluate(() => window.__splat.share.privacy)).toBe('Open');

  // reload, reopen it from This device: a shared scene keeps its link
  await page.reload();
  await page.waitForFunction(() =>
    [...document.querySelectorAll('[data-pane="device"] .galtile')].some((t) => t.textContent.includes('splats')),
    null, { timeout: 30_000 });
  await page.evaluate(() => {
    const t = [...document.querySelectorAll('[data-pane="device"] .galtile')].find((x) => x.textContent.includes('splats'));
    t.click();
  });
  await page.waitForFunction(() => window.__splat.state === 'done' && window.__splat.preset?.id === '__restored',
    null, { timeout: 60_000 });

  expect(await page.evaluate(() => window.__splat.share && window.__splat.share.id)).toBe(SPACE);
  await expect(page.locator('.exportwrap button')).toHaveText(/Share/);
  await page.click('.exportwrap button');
  await expect(page.locator('#sh-url')).toHaveValue(new RegExp(`\\?space=${SPACE}`));
});
