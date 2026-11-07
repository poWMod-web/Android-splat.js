// The viewer's tour and the space's intro cutscene are built from the same
// smoothing (viewport.js: quatFromR / qslerp, and the ±3 position pass in each
// caller). This guards the shared half: a finished run flies its path, with
// smoothed quaternions and an arc-length table behind it.
import { test, expect } from '@playwright/test';
import { seedCapture, startCaptureRun } from './helpers.mjs';

test('the viewer tour still flies after the helpers moved', async ({ page }) => {
  const errs = [];
  page.on('pageerror', (e) => errs.push('pageerror: ' + e.message));
  await seedCapture(page, '?iters=1200');
  await startCaptureRun(page);
  await page.waitForFunction(() => window.__splat.state === 'done', null, { timeout: 240_000 });
  // the finished run flies its own path on its own — that IS startTour
  await page.waitForFunction(() => !!window.__splat.tour, null, { timeout: 20_000 });
  const a = await page.evaluate(() => ({ s: window.__splat.tour.s, keys: window.__splat.tour.sq.length, samples: window.__splat.tour.samples.length }));
  await page.waitForTimeout(1500);
  const b = await page.evaluate(() => window.__splat.tour.s);
  expect(a.keys).toBeGreaterThan(1);
  expect(a.samples).toBeGreaterThan(a.keys);
  expect(b).toBeGreaterThan(a.s);
  expect(errs).toEqual([]);
});
