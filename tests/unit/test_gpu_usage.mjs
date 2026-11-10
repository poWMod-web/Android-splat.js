import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { gpuDuty, gpuIdleMs } from '../../src/session.js';

const sessionSrc = readFileSync(new URL('../../src/session.js', import.meta.url), 'utf8');
const app = readFileSync(new URL('../../app/js/app.js', import.meta.url), 'utf8');
const indexHtml = readFileSync(new URL('../../app/index.html', import.meta.url), 'utf8');

// the throttle math: duty-cycle idle between full batches, never smaller batches
{
  assert.equal(gpuDuty(undefined), 1);
  assert.equal(gpuDuty(null), 1);
  assert.equal(gpuDuty(NaN), 1);
  assert.equal(gpuDuty(1), 1);
  assert.equal(gpuDuty(2), 1);
  assert.equal(gpuDuty(0.5), 0.5);
  assert.equal(gpuDuty(0), 0.001);
  assert.equal(gpuDuty(-1), 0.001);

  assert.equal(gpuIdleMs(120, undefined), 0);   // unset: full speed, legacy behavior
  assert.equal(gpuIdleMs(120, 1), 0);
  assert.equal(gpuIdleMs(120, 0.5), 120);       // half the wall time idle
  assert.equal(gpuIdleMs(100, 0.25), 300);
  assert.equal(gpuIdleMs(120, 0.1), 1080);
  assert.equal(gpuIdleMs(5000, 0.5), 2000);     // one slow frame never parks the loop
  assert.equal(gpuIdleMs(120, 0), 2000);        // minimum speed still progresses
  // the duty promise: busy / (busy + idle) === u whenever the cap is not binding
  for (const u of [0.1, 0.25, 0.5, 0.75, 0.9]) {
    const idle = gpuIdleMs(120, u);
    assert.ok(Math.abs(120 / (120 + idle) - u) < 1e-9, `duty ${u} should reproduce itself`);
  }
}

// the frame loop: a live read, a compensated cap, the sleep before the reschedule check
{
  assert.match(sessionSrc, /gpuIdleMs\(dt, this\.opts\.gpuUsage\)/,
    'the loop should read the GPU usage live every batch');
  assert.match(sessionSrc, /Math\.round\(this\._ips \* chunk \/ gpuDuty\(this\.opts\.gpuUsage\)\)/,
    'the throughput cap should divide the duty back out so batches stay full');
  const sleepAt = sessionSrc.indexOf('await new Promise((r) => setTimeout(r, idleMs))');
  assert.ok(sleepAt > 0, 'the loop should sleep the duty-cycle idle between batches');
  const reschedAt = sessionSrc.indexOf(
    'if (this.training || (this.view.enabled && this.view._dirty)) this._scheduleFrame();', sleepAt);
  assert.ok(reschedAt > sleepAt,
    'the idle should come before the reschedule check so a pause during the sleep stops the loop');
}

// the app: a persisted slider feeding both run types, never a quality flip
{
  assert.equal((app.match(/gpu: 100/g) || []).length, 2,
    'phone and desktop defaults should both start at full GPU usage');
  assert.match(app, /const QKEYS = \['res', 'buf', 'sh', 'iters', 'splats', 'solve'\]/,
    'GPU usage should stay out of the quality macros (never flips Custom)');
  assert.match(app, /\$\('set-gpu'\)\.value = String\(st\.gpu \?\? 100\)/,
    'showSettings should drive the slider from the persisted value');
  assert.match(app, /st\.gpu = Math\.min\(100, Math\.max\(10/,
    'readSettings should persist the slider, clamped to its range');
  assert.match(app, /'set-gpu', \.\.\.AV_KEYS/,
    'the slider should commit on change like every other row');
  assert.match(app, /S\.session\.opts\.gpuUsage = /,
    'dragging the slider should reach a running training live');
  assert.match(app, /gpuUsage: \(st\.gpu \?\? 100\) \/ 100/,
    'fresh runs should start at the slider’s duty cycle');
  assert.match(app, /gpuUsage: \(S\.settings\.gpu \?\? 100\) \/ 100/,
    'continuations should start at the slider’s duty cycle');
  assert.match(indexHtml, /<input id="set-gpu" type="range" min="10" max="100" step="5" value="100">/,
    'the settings panel should offer a 10–100% GPU slider');
}

console.log('GPU usage throttle: passed');
