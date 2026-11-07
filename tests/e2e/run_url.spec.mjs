// The address bar names what is on screen. Training from a shared scene used
// to leave ?space=<the old id> sitting there — confusing to read, and a
// refresh reopened that scene instead of the run.
import { test, expect } from '@playwright/test';
import { seedCapture, startCaptureRun } from './helpers.mjs';

test('a new run stops naming the scene it came from', async ({ page }) => {
  // a ?space= that cannot resolve: the wall still mounts, and the param stays
  // in the address bar exactly as it would after Train on a shared scene
  await seedCapture(page, '?space=42485456_0000&iters=1200');
  expect(new URL(page.url()).searchParams.get('space')).toBe('42485456_0000');

  await startCaptureRun(page);
  await page.waitForFunction(() => ['prep', 'train'].includes(window.__splat.state), null, { timeout: 60_000 });

  const u = new URL(page.url());
  expect(u.searchParams.get('space')).toBeNull();
  expect(u.searchParams.get('iters')).toBe('1200');   // the run's own flags stay
});
