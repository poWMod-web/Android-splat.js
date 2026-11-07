// Video intake: a 6 s clip built from 180 consecutive truck photographs
// (tests/e2e/fixtures/truck_walk.mp4, 640 px, H.264) goes through the v2
// extractor — every frame decoded via WebCodecs, sharp-frames scoring,
// motion-window selection — and the winners train like a photo set.
import { test, expect } from '@playwright/test';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const here = dirname(fileURLToPath(import.meta.url));

test('own-video run: extract frames, solve, train', async ({ page }) => {
  const logs = [];
  page.on('console', (m) => { if (m.text().startsWith('[video]')) logs.push(m.text()); });
  await page.goto('/app/?iters=500');
  await page.setInputFiles('#file-input', join(here, 'fixtures', 'truck_walk.mp4'));

  // the scan ends in the review card (timeline + filmstrip); accept the automatic picks
  await page.waitForSelector('#vid-use', { timeout: 120_000 });
  const review = await page.evaluate(() => ({ picks: window.__splat.videoReview.picks.length, scored: window.__splat.videoReview.frames.length }));
  expect(review.scored).toBeGreaterThan(100);
  expect(review.picks).toBeGreaterThanOrEqual(12);
  await page.click('#vid-use');

  // extraction done: the scene opened from the extracted frames, ready to start
  await page.waitForFunction(() => {
    const b = document.getElementById('btn-go');
    return b && !b.disabled && window.__splat && window.__splat.state === 'ready' &&
      Array.isArray(window.__splat.ownUrls) && window.__splat.ownUrls.length >= 12;
  }, null, { timeout: 120_000 });
  const extracted = await page.evaluate(() => window.__splat.ownUrls.length);
  expect(extracted).toBeGreaterThanOrEqual(12);
  expect(extracted).toBeLessThanOrEqual(180);
  // the WebCodecs engine is expected here (Chromium); the element path is the fallback
  expect(logs.some((l) => /frames @|scored \d+ frames/.test(l))).toBeTruthy();

  await page.evaluate(() => document.getElementById('btn-go').click());
  await page.waitForFunction(() => window.__splat.state === 'done', null, { timeout: 240_000 });
  const done = await page.evaluate(() => ({ iter: window.__splat.iter, splats: window.__splat.splats, psnr: window.__splat.psnrTrain }));
  expect(done.iter).toBeGreaterThanOrEqual(500);
  expect(done.splats).toBeGreaterThan(5_000);
  expect(done.psnr).not.toBeNull();
  expect(done.psnr).toBeGreaterThan(15);
});
