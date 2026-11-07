// Playwright config for the client E2E suite.
//
// These tests drive the REAL app (app/) in headless Chrome with real WebGPU
// training on the local GPU — they cannot run on GPU-less CI runners. The
// suite is meant for the dev box: nightly, and as a gate before deploys.
//
//   npm run test:e2e
//
// Uses the installed system Chrome (channel: 'chrome') so the browser under
// test is the one users run. Do NOT add --use-angle=d3d11: it breaks
// headless D3D12 device creation on some driver states (see docs/lab-log.md
// 2026-08-31), and Dawn's d3d11 fallback trains garbage silently.
import { defineConfig } from '@playwright/test';
import { fileURLToPath } from 'url';

const root = fileURLToPath(new URL('../..', import.meta.url));

export default defineConfig({
  testDir: fileURLToPath(new URL('.', import.meta.url)),
  outputDir: fileURLToPath(new URL('./test-results', import.meta.url)),
  // one GPU, stateful runs — never parallelize
  workers: 1,
  fullyParallel: false,
  timeout: 300_000,
  expect: { timeout: 30_000 },
  retries: 0,
  reporter: 'list',
  use: {
    baseURL: 'http://localhost:8734',
    channel: 'chrome',
    headless: true,
    launchOptions: { args: ['--enable-unsafe-webgpu'] },
    viewport: { width: 1600, height: 1000 },
    screenshot: 'only-on-failure',
  },
  webServer: {
    command: 'node serve.mjs 8734',
    url: 'http://localhost:8734/app/index.html',
    cwd: root,
    reuseExistingServer: true,
  },
});
