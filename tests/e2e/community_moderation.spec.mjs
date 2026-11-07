// Who may clear a scene off the Community wall. The menu is the whole control
// here, so the guard is that a plain visitor never gets one on a scene that is
// not theirs — and that a moderator's press only fires after the confirm.
//
// Stubbed API, no GPU: the wall is the only thing under test.
import { test, expect } from '@playwright/test';

const OTHERS = {
  id: '99999999_1234', title: "Someone else's scene", privacy: 'Open',
  createdDate: '2026-09-15T10:00:00.000Z', splatjs: { splats: 12345, psnrTrain: 27.1 },
};

function stub(page, { isAdmin }) {
  const puts = [];
  page.route(/api-live\.arrival\.space/, async (route) => {
    const url = route.request().url();
    const method = route.request().method();
    const json = (o) => route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(o) });
    if (/\/user\/me$/.test(url)) return json({ status: 'ok', data: { id: '42485456', name: 'Tester', isAdmin } });
    if (/\/splatjs\/mine$/.test(url)) return json({ status: 'success', data: { items: [] } });   // owns nothing
    if (/\/splatjs\/gallery/.test(url)) return json({ status: 'success', data: { items: [OTHERS], nextBefore: null } });
    if (/\/api\/v1\/spaces\//.test(url) && method === 'PUT') {
      puts.push(JSON.parse(route.request().postData() || '{}'));
      return json({ status: 'success', data: {} });
    }
    return json({ status: 'success', data: {} });
  });
  return puts;
}

async function openWall(page) {
  await page.goto('/app/');
  await page.evaluate(() => localStorage.setItem('arrival_token:https://api-live.arrival.space', 'test-token'));
  await page.reload();
  await page.waitForSelector('[data-pane="community"] .galtile', { state: 'attached', timeout: 30_000 });
  await page.click('#walltabs [data-tab="community"]');   // the pane starts hidden behind its tab
  return page.locator('[data-pane="community"] .galtile').first();
}

test('a plain user gets no menu on someone else\'s scene', async ({ page }) => {
  stub(page, { isAdmin: false });
  const tile = await openWall(page);
  await expect(tile.locator('.run-menu')).toHaveCount(0);
});

test('a moderator can remove any scene from Community', async ({ page }) => {
  const puts = stub(page, { isAdmin: true });
  const tile = await openWall(page);
  await expect(tile.locator('.run-menu')).toHaveCount(1);

  await tile.locator('.run-menu').click();
  const labels = await tile.locator('.tilemenu button').allTextContents();
  expect(labels).toEqual(['Copy link', 'Remove from Community']);

  const remove = tile.locator('.tilemenu button.danger');   // stable: the label changes when armed
  await remove.click();                                    // first press arms
  expect(puts.length).toBe(0);
  await expect(remove).toHaveText('Delete?');
  await remove.click();                                    // second press does it
  await expect.poll(() => puts.length, { timeout: 15_000 }).toBe(1);
  expect(puts[0]).toEqual({ privacy: 'Link Only' });
});
