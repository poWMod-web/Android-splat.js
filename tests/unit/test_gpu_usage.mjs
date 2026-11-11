import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { gpuDuty, gpuIdleMs, gpuBurstMs } from '../../src/session.js';

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
  assert.match(sessionSrc, /gpuIdleMs\(dt, this\._effUsage\(\)\)/,
    'the loop should read the effective (governor x slider) GPU usage live every batch');
  assert.match(sessionSrc, /Math\.round\(this\._ips \* chunk \/ gpuDuty\(this\._effUsage\(\)\)\)/,
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
  assert.match(app, /<input id="t-gpu" type="range" min="10" max="100" step="5" value="\$\{S\.settings\.gpu \?\? 100\}">/,
    'the training dock should carry the GPU slider mid-run');
  assert.match(app, /tGpu\.addEventListener\('input', \(\) => \{[\s\S]*?S\.session\.opts\.gpuUsage = v \/ 100;/,
    'the dock slider should drive the live session directly');
  assert.match(app, /if \(\$\('t-gpu'\)\) \{ \$\('t-gpu'\)\.value = String\(st\.gpu \?\? 100\);/,
    'showSettings should mirror the persisted value into the dock');
  assert.match(app, /gpuUsage: \(st\.gpu \?\? 100\) \/ 100/,
    'fresh runs should start at the slider’s duty cycle');
  assert.match(app, /gpuUsage: \(S\.settings\.gpu \?\? 100\) \/ 100/,
    'continuations should start at the slider’s duty cycle');
  assert.match(indexHtml, /<input id="set-gpu" type="range" min="10" max="100" step="5" value="100">/,
    'the settings panel should offer a 10–100% GPU slider');
}

// touch-hold with a quiet window: training yields the GPU from the first
// model touch until 2s after the last, then comes back at full speed
{
  assert.match(app, /const onModel = \(e\) => !e\.target\.closest\('#controls, #hud, #start, \.chartwrap, button'\);/,
    'only model touches should hold the GPU — graph and hub touches must not');
  assert.match(app, /st\.addEventListener\('pointerdown', \(e\) => \{\s*if \(!onModel\(e\)\) return;\s*held\.add\(e\.pointerId\);\s*noteModelTouch\(\);/,
    'a model touch should engage the hold (orbit, pinch, compare drag, tap)');
  assert.match(app, /window\.addEventListener\('pointerup', \(e\) => \{\s*if \(held\.delete\(e\.pointerId\) && !held\.size && S\._dragPaused\) armResume\(\);/,
    'releases should ride the window so lifting outside the stage still counts');
  assert.match(app, /window\.addEventListener\('pointercancel', \(e\) => \{\s*if \(held\.delete\(e\.pointerId\) && !held\.size && S\._dragPaused\) armResume\(\);/,
    'cancelled touches should count toward the quiet window too');
  assert.match(app, /st\.addEventListener\('wheel', \(e\) => \{\s*if \(!onModel\(e\)\) return;/,
    'wheel ticks adjust the model without a touch — same hold, same window');
  assert.match(app, /S\.state === 'train' && S\.trainVisuals && S\.session\.training && !S\._dragPaused/,
    'the hold should engage only while training with visuals shown');
  assert.match(app, /S\._dragPaused = true;\s*S\.session\.pause\(\);/,
    'the hold should pause a running training');
  assert.match(app, /\}, 500\);/,
    'training should resume after 500ms of no interaction');
  assert.match(app, /clearTimeout\(resumeT\); resumeT = 0;/,
    'every new touch should refresh the quiet window');
  assert.match(app, /S\._dragPaused = false;\s*if \(S\.state === 'train'\) S\.session\.start\(\);/,
    'the quiet window should resume a hold-paused training');
  assert.match(app, /if \(!on\) \{ S\._dragPaused = false; checkpointRun\('pause'\)/,
    'a user-pressed pause should take ownership so the window never resumes over it');
  assert.match(sessionSrc, /this\.training = true;\n(\s*\/\/.*\n)+\s*this\._thrT = null;/,
    'resuming should restart the throughput window instead of reading the pause gap as GPU slowness');
}

// duty honesty: bursts shrink with the slider and the sleep drains first,
// so 10% is genuinely GPU-free instead of backlog-hot
{
  assert.equal(gpuBurstMs(400, undefined), 400);
  assert.equal(gpuBurstMs(400, 1), 400);       // full duty: exactly the old bound
  assert.equal(gpuBurstMs(400, 0.5), 200);
  assert.equal(gpuBurstMs(400, 0.1), 40);
  assert.equal(gpuBurstMs(120, 0.1), 12);
  assert.match(sessionSrc, /const chunk = gpuBurstMs\(this\.opts\.gpuChunkMs \?\? 400, this\._effUsage\(\)\) \/ 1000;/,
    'the batch cap should bound duty-scaled bursts, not full-size ones');
  assert.match(sessionSrc, /if \(duty < 1 && this\._fences\.length\) \{/,
    'the honesty drain should run below full duty and never at it');
  assert.match(sessionSrc, /gpuIdleMs\(dt, this\._effUsage\(\)\) \* 0\.25/,
    'the drain should be bounded by a fraction of the coming idle');
  assert.match(sessionSrc, /if \(drained\) this\._fences\.length = 0;/,
    'a successful drain should empty the ring so the sleep starts GPU-free');
  const drainAt = sessionSrc.indexOf('const drained = await Promise.race([');
  assert.ok(drainAt > 0 && drainAt < sessionSrc.indexOf('this._batchIdeal(batch, paceMs, gpuMs)', drainAt),
    'the drain should land before the measurement the adapters and idle are computed from');
}

// latest-wins view renders: orbiting must not queue stale frames behind the GPU
{
  assert.match(sessionSrc, /renderNow\(force = false\) \{/,
    'renderNow should take a force flag for synchronous snapshots');
  assert.match(sessionSrc, /if \(this\._renderInFlight && !force\) \{ this\._dirty = true; return; \}/,
    'a render in flight should defer to the newest camera instead of queueing');
  assert.match(sessionSrc, /this\._renderInFlight = t;/,
    'each submitted render should arm the gate');
  assert.equal((app.match(/\.renderNow\(true\)/g) || []).length, 2,
    'both canvas-readback snapshots should force their renders');
}

console.log('GPU usage throttle: passed');
