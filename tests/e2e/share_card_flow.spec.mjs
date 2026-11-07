// The share card is where the press happened, so it is where the work and the
// link belong: it stays open, swaps the form for a progress line, and ends on
// the link itself — not in a note in the corner.
//
// create-space is held open here on purpose, to catch the working state.
import { test, expect } from '@playwright/test';
import { seedCapture, startCaptureRun } from './helpers.mjs';

test('the card keeps the work and the link', async ({ page }) => {
  let releaseSpace;
  const spaceHeld = new Promise((r) => { releaseSpace = r; });

  await page.route(/api-live\.arrival\.space/, async (route) => {
    const url = route.request().url();
    const json = (o) => route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(o) });
    if (/\/splatjs\/(mine|gallery)/.test(url)) return json({ status: 'success', data: { items: [] } });
    if (/\/user\/me$/.test(url)) return json({ status: 'ok', data: { id: '42485456', name: 'Tester' } });
    if (/\/files\/upload$/.test(url)) return json({ status: 'success', data: { params: { method: 'PUT', url: 'https://cdn.test/f?sig=1', headers: {} } } });
    if (/\/files\/upload-complete$/.test(url)) return json({ status: 'success', data: { resource_key: 'rk', url: 'https://cdn.test/f.bin' } });
    if (/\/user\/create-space$/.test(url)) {
      await spaceHeld;   // hold it here so the working state can be observed
      return json({ status: 'success', data: { space_url: 'https://arrival.space/42485456_4242' } });
    }
    return json({ status: 'success', data: {} });
  });
  await page.route(/cdn\.test/, (route) => route.fulfill({ status: 200, body: '' }));

  await seedCapture(page, '?iters=1200');
  await page.evaluate(() => localStorage.setItem('arrival_token:https://api-live.arrival.space', 'test-token'));
  await startCaptureRun(page);
  await page.waitForFunction(() => window.__splat.state === 'done', null, { timeout: 240_000 });

  await page.click('.exportwrap button');
  await page.waitForSelector('#sh-go');
  await page.uncheck('#sh-priv');                       // the Get link path
  await page.click('#sh-go');

  // the card is still there, working, with the form put away
  await expect(page.locator('#upcard')).toBeVisible();
  await expect(page.locator('#sh-work')).toBeVisible();
  await expect(page.locator('.sh-form')).toBeHidden();
  await expect(page.locator('#upcard > b')).toHaveText('Making your link …');
  await expect(page.locator('#sh-status')).not.toHaveText('');

  releaseSpace();

  // and it ends on the link, in the card
  await expect(page.locator('#sh-done')).toBeVisible({ timeout: 120_000 });
  await expect(page.locator('#upcard > b')).toHaveText('Your link is ready');
  await expect(page.locator('#sh-work')).toBeHidden();
  const url = await page.inputValue('#sh-done-url');
  expect(url).toContain('?space=42485456_4242');
  await expect(page.locator('#sh-done-enter')).toHaveAttribute('href', 'https://arrival.space/42485456_4242');
  await expect(page.locator('#sh-done-copy')).toBeVisible();
});
