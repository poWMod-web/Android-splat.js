// GPU preflight: fail LOUD when the rig can't do real WebGPU, instead of
// letting later specs time out or — worse — pass against a software or
// half-broken backend. (Dawn's d3d11 fallback runs every pipeline but its
// stats atomics read zero: models train to garbage while looking alive.)
import { test, expect } from '@playwright/test';

test('hardware WebGPU adapter creates the app device', async ({ page }) => {
  test.setTimeout(60_000);
  await page.goto('/app/');
  const probe = await page.evaluate(async () => {
    if (!navigator.gpu) return { err: 'navigator.gpu missing' };
    const a = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
    if (!a) return { err: 'no adapter' };
    const info = a.info || {};
    try {
      // the exact request the app makes (src/gpu/context.js)
      const want = 4 * (1 << 30);
      const d = await a.requestDevice({
        requiredFeatures: a.features.has('subgroups') ? ['subgroups'] : [],
        requiredLimits: {
          maxStorageBufferBindingSize: Math.min(a.limits.maxStorageBufferBindingSize, want),
          maxBufferSize: Math.min(a.limits.maxBufferSize, want),
        },
      });
      const limits = { maxBufferSize: d.limits.maxBufferSize };
      d.destroy();
      return { vendor: info.vendor || '', architecture: info.architecture || '', limits };
    } catch (e) {
      return { err: String((e && e.message) || e), vendor: info.vendor || '' };
    }
  });
  expect(probe.err, `device creation failed (${JSON.stringify(probe)}) — ` +
    'if this is DXGI_ERROR_DEVICE_REMOVED, check launch flags: --use-angle=d3d11 ' +
    'is known to break headless D3D12 (docs/lab-log.md 2026-08-31)').toBeUndefined();
  // software rasterizers "work" but train at unusable speed and have lied
  // to us before — a hardware vendor string is required
  expect(probe.vendor.toLowerCase()).not.toContain('swiftshader');
  expect(probe.vendor.toLowerCase()).not.toContain('llvmpipe');
  expect(probe.limits.maxBufferSize).toBeGreaterThanOrEqual(1 << 30);
});
