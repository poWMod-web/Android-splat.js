// Plan items (6) yields, (7) hidden-tab skips, (10) pooled step temporaries,
// plus the duty governor (auto rAF/event-loop health, slider as ceiling):
// the responsiveness half of the loop, layered on the session shell and the
// trainer (both local and worker — policy lives shell-side, yields and pools
// inside trainer methods).
import assert from 'node:assert/strict';
import { installWebGpuShims, strictMockGpu } from './mock_gpu.mjs';

installWebGpuShims();
const { GSTrainer, UNI_RING, MERGE_BUDGET_MS, UNMEASURED_MERGE } = await import('../../src/gs/trainer.js');
const { initGaussians } = await import('../../src/gs/init.js');
const { createSession, governDuty, GOV_DEFAULTS } = await import('../../src/session.js');

function synthPoints(np) {
  const pts = [];
  for (let i = 0; i < np; i++) {
    pts.push({
      X: [Math.sin(i * 1.7) * 2, Math.cos(i * 2.3), (i % 17) * 0.1 - 0.8],
      rgb: [0.5 + 0.1 * Math.sin(i), 0.5, 0.5],
    });
  }
  return pts;
}

function synthCams(n, w = 64, h = 48) {
  const cams = [];
  for (let i = 0; i < n; i++) {
    cams.push({
      imgIdx: i, R: [1, 0, 0, 0, 1, 0, 0, 0, 1], t: [i * 0.5, 0, 0],
      f: 50, cx: w / 2, cy: h / 2,
    });
  }
  return cams;
}

function synthFrames(n, w = 64, h = 48) {
  const frames = [];
  for (let i = 0; i < n; i++) {
    const rgb = new Float32Array(w * h * 3);
    for (let p = 0; p < w * h; p++) {
      rgb[p * 3] = 0.4 + 0.2 * ((p + i) % 5) / 5;
      rgb[p * 3 + 1] = 0.5; rgb[p * 3 + 2] = 0.6;
    }
    frames.push({ tw: w, th: h, fw: w, fh: h, rgb, alpha: null, sharpness: 100 + i, emptyFrac: 0 });
  }
  return frames;
}

async function smallTrainer(trainerOpts = {}) {
  const gpu = strictMockGpu({});
  const trainer = await GSTrainer.create({ gpu, maxIters: 10000, ...trainerOpts });
  const model = initGaussians(synthPoints(300), 2);
  trainer.setup(model, synthCams(4), synthFrames(4), 256, 144, model.radius);
  return { gpu, trainer };
}

// ---- the governor law (pure): derate fast, recover slow, hold between ----
{
  assert.deepEqual(GOV_DEFAULTS,
    { lagJank: 50, lagOk: 20, rafJank: 34, rafOk: 20,
      gpuJank: 1.25, gpuOk: 1.1, down: 0.7, up: 0.1, floor: 0.1 });
  let r = governDuty(1, { lagMs: 100, rafMs: 16 });
  assert.equal(r.auto, 0.7); assert.equal(r.moved, -1);
  r = governDuty(1, { lagMs: 5, rafMs: 50 });
  assert.equal(r.auto, 0.7); assert.equal(r.moved, -1);
  r = governDuty(0.11, { lagMs: 500, rafMs: 250 });
  assert.equal(r.auto, 0.1); assert.equal(r.moved, -1);   // the floor holds
  r = governDuty(0.5, { lagMs: 5, rafMs: 16 });
  assert.equal(r.auto, 0.6); assert.equal(r.moved, 1);
  r = governDuty(0.95, { lagMs: 0, rafMs: 16.7 });
  assert.equal(r.auto, 1); assert.equal(r.moved, 1);      // recovery caps at full
  r = governDuty(1, { lagMs: 0, rafMs: 16.7 });
  assert.equal(r.auto, 1); assert.equal(r.moved, 0);      // healthy at full: nothing to do
  r = governDuty(0.5, { lagMs: 30, rafMs: 16 });
  assert.equal(r.auto, 0.5); assert.equal(r.moved, 0);    // between thresholds: hold
  r = governDuty(0.5, { lagMs: 5, rafMs: 25 });
  assert.equal(r.auto, 0.5); assert.equal(r.moved, 0);    // one sick signal vetoes recovery
  r = governDuty(1, { lagMs: 0, rafMs: 16.7, gpuRatio: 1.5 });
  assert.equal(r.auto, 0.7); assert.equal(r.moved, -1);   // (3) GPU sag derates alone
  r = governDuty(0.5, { lagMs: 5, rafMs: 16, gpuRatio: 1.2 });
  assert.equal(r.auto, 0.5); assert.equal(r.moved, 0);    // warm GPU vetoes recovery
  r = governDuty(0.5, { lagMs: 5, rafMs: 16, gpuRatio: 1.05 });
  assert.equal(r.auto, 0.6); assert.equal(r.moved, 1);    // cool GPU lets it grow back
  r = governDuty(1, { lagMs: 100 }, { down: 0.5, floor: 0.2 });
  assert.equal(r.auto, 0.5);                              // opts tune the law
  r = governDuty(0.3, { lagMs: 100 }, { down: 0.5, floor: 0.2 });
  assert.equal(r.auto, 0.2);
}

// ---- governor wiring: slider ceiling, hidden/disabled bypass, sampler ----
{
  const s = createSession({ gpuUsage: 0.5 });
  assert.equal(s._effUsage(), 0.5);
  s._govAuto = 0.7;
  assert.equal(s._effUsage(), 0.5);                       // slider still the ceiling
  const s2 = createSession({});
  s2._govAuto = 0.7;
  assert.equal(s2._effUsage(), 0.7);                      // unset slider: auto rules
  const s3 = createSession({ gpuUsage: 0.5, autoDuty: false });
  s3._govAuto = 0.1;
  assert.equal(s3._effUsage(), 0.5);                      // disabled: slider exactly

  // EWMA samplers
  s._govNoteLag(100);
  assert.equal(s._govLag, 30);
  s._govNoteRaf(1000);
  assert.equal(s._govRaf, 16.7);                          // first tick only anchors
  s._govNoteRaf(1032);
  assert.ok(Math.abs(s._govRaf - 19.76) < 1e-9);

  // (3) the GPU sampler: anchors, min-filters, re-anchors on growth
  const g = createSession({});
  assert.equal(g._govGpuRatio(), 1);                      // unanchored: healthy
  g._govNoteGpu(null, 10, 100);
  assert.equal(g._govN, -1);                              // timerless: untouched
  g._govNoteGpu(30, 10, 100);
  assert.equal(g._govMsStep, 3); assert.equal(g._govMsBase, 3);
  assert.equal(g._govN, 100); assert.equal(g._govGpuRatio(), 1);
  g._govNoteGpu(15, 10, 100);                             // faster: floor follows down
  assert.ok(Math.abs(g._govMsStep - 2.55) < 1e-9);
  assert.equal(g._govMsBase, g._govMsStep);
  for (let k = 0; k < 10; k++) g._govNoteGpu(60, 10, 100); // sustained sag
  assert.equal(g._govMsBase, 2.55);                       // ...never lifts the floor
  assert.ok(g._govGpuRatio() > 2);
  g._govNoteGpu(60, 10, 200);                             // refine grew n: re-anchor
  assert.equal(g._govMsStep, 6); assert.equal(g._govMsBase, 6);
  assert.equal(g._govGpuRatio(), 1);
  g._govNoteGpu(0, 10, 200); g._govNoteGpu(NaN, 10, 200); g._govNoteGpu(60, 0, 200);
  assert.equal(g._govMsBase, 6);                           // garbage in, state kept

  // controller step: derate + bucketed logging, then recovery
  const logs = [];
  s.on('log', (m) => logs.push(m));
  s._govLag = 100; s._govRaf = 16.7; s._govAuto = 1; s._govBucket = 1;
  s._govUpdate();
  assert.equal(s._govAuto, 0.7);
  assert.equal(logs.length, 1);
  assert.match(logs[0], /auto-derated to 70% of the slider/);
  s._govUpdate();
  assert.ok(Math.abs(s._govAuto - 0.49) < 1e-9);
  assert.equal(logs.length, 2);                           // crossed into the 0.25 bucket
  s._govLag = 0;
  for (let k = 0; k < 6; k++) s._govUpdate();
  assert.equal(s._govAuto, 1);
  assert.match(logs[logs.length - 1], /system healthy — GPU duty back to the slider/);

  // (3) GPU-only sag names its cause; a mixed trip stays "display" w/ the clause
  const t = createSession({});
  const tlogs = [];
  t.on('log', (m) => tlogs.push(m));
  t._govLag = 3; t._govRaf = 17; t._govMsStep = 1.5; t._govMsBase = 1;
  t._govAuto = 1; t._govBucket = 1;
  t._govUpdate();
  assert.equal(t._govAuto, 0.7);
  assert.match(tlogs[0], /GPU throttling \(event-loop lag 3ms, frame 17ms, GPU step cost 1\.50x its best\)/);
  const u = createSession({});
  const ulogs = [];
  u.on('log', (m) => ulogs.push(m));
  u._govLag = 100; u._govRaf = 17; u._govMsStep = 1.5; u._govMsBase = 1;
  u._govAuto = 1; u._govBucket = 1;
  u._govUpdate();
  assert.match(ulogs[0], /display struggling \(event-loop lag 100ms, frame 17ms, GPU step cost 1\.50x its best\)/);

  // frozen while hidden or disabled
  globalThis.document = { hidden: true };
  try {
    assert.equal(s._effUsage(), 0.5);                     // hidden: slider as-is
    s._govAuto = 0.49; s._govLag = 500;
    s._govUpdate();
    assert.equal(s._govAuto, 0.49);                       // frozen, not derated
    const s4 = createSession({ autoDuty: false });
    s4._govAuto = 0.5; s4._govLag = 500;
    s4._govUpdate();
    assert.equal(s4._govAuto, 0.5);
  } finally {
    delete globalThis.document;
  }

  // the rAF sampler loop schedules while training, parks on pause
  const s5 = createSession({});
  s5.training = true;
  const scheduled = [];
  s5._sched = { raf: (fn) => scheduled.push(fn) };
  s5._govRafRunning = true;
  s5._govRafLoop();
  assert.equal(scheduled.length, 1);
  assert.ok(s5._govRafLast > 0);
  s5.training = false;
  scheduled[0]();
  assert.equal(s5._govRafRunning, false);
  assert.equal(scheduled.length, 1);                      // parked: no reschedule

  // the lag probe delivers through a real timer
  const s6 = createSession({});
  let noted = null;
  s6._govNoteLag = (ms) => { noted = ms; };
  s6._govProbe();
  await new Promise((r) => setTimeout(r, 25));
  assert.ok(typeof noted === 'number' && noted >= 0);
}

// ---- (7) hidden tabs: no renders, no holdout evals, training continues ----
{
  const s = createSession({});
  const view = s.view;
  view.enabled = true;
  view.camera = {};
  view.ctx = {};
  let renders = 0;
  view.renderNow = () => { renders++; view._dirty = false; };
  globalThis.document = { hidden: true };
  try {
    view._dirty = true;
    view._tick(10, true);
    assert.equal(renders, 0);                             // dirty render skipped
    assert.equal(view._dirty, true);                      // ...but not lost
    view._dirty = false;
    view._lastAuto = -10000;
    view._tick(11, true);
    assert.equal(renders, 0);                             // auto-refresh skipped
  } finally {
    delete globalThis.document;
  }
  view._dirty = true;
  view._tick(12, true);
  assert.equal(renders, 1);                               // visible: dirty renders
  assert.equal(view._dirty, false);
  view._lastAuto = -10000;
  view._tick(13, true);
  assert.equal(renders, 2);                               // visible: auto-refresh runs
}

// ---- (7) the holdout eval follows the same rule (loss is kept; final always evals) ----
{
  const mk = () => {
    const s = createSession({ holdout: -1 });
    let evals = 0;
    s.trainer = {
      iter: 5000, n: 100,
      readLoss: async () => 0.01,
      evalCamPsnr: async () => { evals++; return 25; },
    };
    s.holdout = 0;
    s._lastHoldEval = 0;
    s._lastStats = performance.now() - 3000;
    s._itersAtStats = 4900;
    return { s, evals: () => evals };
  };
  globalThis.document = { hidden: true };
  try {
    const { s, evals } = mk();
    const m = await s._emitMetricsInner(false);
    assert.equal(evals(), 0);                             // holdout skipped
    assert.equal(s._lastHoldEval, 0);                     // stale: fires on return
    assert.ok(m.psnrTrain > 0);                           // ...but the loss is kept
    assert.equal(m.psnrHold, undefined);
    assert.equal(typeof m.duty, 'number');                // duty rides the metrics
    assert.equal(typeof m.autoDuty, 'number');
    const mf = await s._emitMetricsInner(true);
    assert.equal(evals(), 1);                             // final always evaluates
    assert.ok(mf.psnrHold > 0);
  } finally {
    delete globalThis.document;
  }
  const { s, evals } = mk();
  const m = await s._emitMetricsInner(false);
  assert.equal(evals(), 1);                               // visible: holdout runs
  assert.ok(m.psnrHold > 0);
  assert.equal(s._lastHoldEval, 5000);
}

// ---- (6) yields: one event-loop turn per queue drain, every 25 refined passes ----
{
  const realSetTimeout = globalThis.setTimeout;
  const countYields = async (fn) => {
    let n = 0;
    globalThis.setTimeout = ((prev) => (cb, ms, ...a) => {
      if (ms === 0 || ms == null) n++;
      return prev(cb, ms, ...a);
    })(realSetTimeout);
    try {
      await fn();
      return n;
    } finally {
      globalThis.setTimeout = realSetTimeout;
    }
  };
  // legacy refine: params drain + gather drain
  {
    const { trainer } = await smallTrainer({ maxSplats: 6000, capMult: 2, shDeg: 0 });
    const yields = await countYields(() => trainer.refine());
    assert.equal(yields, 2);
  }
  // v2 refine: gather drain (death-heavy, so planning runs too)
  {
    const gpu = strictMockGpu({});
    const trainer = await GSTrainer.create({
      gpu, maxIters: 10000, maxSplats: 6000, capMult: 2, shDeg: 0,
      refineV2: true, growRate: 0.5, moveCap: 1.0,
    });
    const model = initGaussians(synthPoints(1500), 1);
    trainer.setup(model, synthCams(4), synthFrames(4), 256, 144, model.radius);
    gpu.device.mapFillers['refine-gatherRead'] = (ab) => {
      const g = new Float32Array(ab);
      for (let i = 0; i < 3000; i++) {
        const dead = i < 2500;
        g[i * 4] = dead ? -5 : 0;
        g[i * 4 + 1] = dead ? 0 : 1;
        g[i * 4 + 2] = dead ? 0 : 1;
        g[i * 4 + 3] = 0;
      }
    };
    const yields = await countYields(() => trainer.refine());
    assert.equal(yields, 1);
  }
  // refined eval: 50 passes breathe twice
  {
    const { trainer } = await smallTrainer({ maxSplats: 20000, capMult: 4, shDeg: 0 });
    const yields = await countYields(() => trainer.evalCamPsnrRefined(0, 50));
    assert.equal(yields, 2);
  }
}

// ---- (10) pooled step temporaries: SSAA twin + warmup scalings ----
{
  const { trainer } = await smallTrainer({ maxSplats: 20000, capMult: 4, shDeg: 0, ssaa: 2 });
  trainer.stepOnce();
  trainer.stepOnce();
  const u = trainer.camUniforms[trainer.lastCam];
  assert.ok(Math.abs(trainer._u2[16] - u[16] * 2) < 1e-6,
    'the supersampled twin comes from the pool');
  assert.ok(Math.abs(trainer._u2[20] - u[20] * 2) < 1e-6);

  const { trainer: w } = await smallTrainer({ maxSplats: 20000, capMult: 4, shDeg: 0, lrWarmup: 100 });
  w.iter = 0; w.adamT0 = 0;
  const out = w._warmed(w.adamData, 0, 14, w._warmAdam);
  assert.equal(out, w._warmAdam);                          // identity: the pool, no copy
  assert.ok(Math.abs(out[0] - w.adamData[0] * 0.01) < 1e-9);
  assert.equal(out[20], w.adamData[20]);                   // outside [lo, hi): untouched
  const fresh = w._warmed(w.adamData, 0, 14);
  assert.notEqual(fresh, w.adamData);                      // no scratch: copy, as before
  assert.notEqual(fresh, w._warmAdam);
  w.iter = 200;
  assert.equal(w._warmed(w.adamData, 0, 14, w._warmAdam), w.adamData); // warm: the array itself
}

// ---- (4) the submit cut: ms-budgeted queue depth, wall-drained ----
{
  const c = createSession({});
  assert.equal(c._submitCutMs(10), 0);                 // unanchored: submit, pred 0
  c._govNoteGpu(100, 10, 50);                          // frame 1's steps anchor msStep 10
  c._lastGpuMs = 100;
  assert.equal(c._submitCutMs(10), 0);                 // 0 + correct 100 + pred 100
  assert.equal(c._cutMs, 200);
  c._lastGpuMs = 100;
  assert.equal(c._submitCutMs(10), 0);
  assert.ok(c._cutMs > 299 && c._cutMs <= 300);
  c._lastGpuMs = 100;
  assert.equal(c._submitCutMs(10), 0);                 // ~400: still within budget
  c._lastGpuMs = 100;
  const cut = c._submitCutMs(10);                      // ~500: over, idle a beat
  assert.ok(cut > 50 && cut <= 100);
  assert.equal(c._cutPending, false);                  // a cut submits nothing to correct
  c._cutT -= 10000;                                    // the wall drain re-opens the gate
  assert.equal(c._submitCutMs(10), 0);
  c._lastGpuMs = 20;                                   // over-predicted: correct downward
  c._submitCutMs(10);
  assert.ok(c._cutMs > 115 && c._cutMs < 125, `cutMs ${c._cutMs}`);

  // the budget knob binds; timerless loops never cut
  const b = createSession({ gpuChunkMs: 150 });
  b._govNoteGpu(100, 10, 50);
  assert.equal(b._submitCutMs(10), 0);
  b._lastGpuMs = 100;
  b._cutMs = 10000;                                    // deep over: capped at one beat
  assert.equal(b._submitCutMs(10), 100);
  const n = createSession({});
  for (let k = 0; k < 5; k++) assert.equal(n._submitCutMs(10), 0);
  assert.equal(n._cutMs, 0);

  // a cut frame runs no steps, ticks the view, idles, and reschedules
  const f = createSession({});
  const scheduled = [];
  f._sched = { raf: (fn) => scheduled.push(fn), runFrame: () => {} };
  let steps = 0, ticks = 0;
  f.trainer = {
    camMeta: [{ w: 8, h: 8 }], iter: 0, n: 50,
    stepOnce() { steps++; },
    device: { queue: { async onSubmittedWorkDone() {} } },
  };
  f.view._tick = () => { ticks++; };
  f.training = true;
  f._frameCount = 0;
  f._lastStats = performance.now();
  f._govNoteGpu(100, 10, 50);
  f._cutMs = 10000;
  await f._frameLoop();
  assert.equal(steps, 0);
  assert.equal(ticks, 1);
  assert.equal(scheduled.length, 1);
  assert.equal(f.perf.marks.length, 1);
  assert.equal(f.perf.marks[0].kind, 'cut');
  f._cutT -= 20000;                                    // drained: the next frame submits
  await f._frameLoop();
  assert.equal(steps, 11);                             // ...at the decayed batch (15→10.5→round 11)
  assert.equal(f._cutStreak, 0, 'resumed work resets the streak');
  assert.equal(scheduled.length, 2);

  // the floor batch always submits: a minimum batch over budget must run
  // (slowly, honestly) — cutting it idles forever, since the cut branch
  // never re-estimates (2026-11: large-res phones hung exactly this way)
  const g = createSession({});
  g._govNoteGpu(100, 10, 50);
  g._govMsStep = 1000;                                 // a 1s/step estimate
  assert.equal(g._submitCutMs(4), 0, 'floor submits despite pred 4000 >> budget');
  assert.ok(g._submitCutMs(32) > 0, 'above the floor still cuts');

  // ...and above the floor the cut ESCAPES: each cut frame decays the
  // batch toward the floor instead of freezing it over budget forever
  const e = createSession({});
  const elogs = [];
  e.on('log', (m) => elogs.push(m));
  e._sched = { raf: () => {}, runFrame: () => {} };
  let esteps = 0;
  e.trainer = {
    camMeta: [{ w: 8, h: 8 }], iter: 0, n: 50,
    stepOnce() { esteps++; },
    device: { queue: { async onSubmittedWorkDone() {} } },
  };
  e.view._tick = () => {};
  e.training = true;
  e._frameCount = 0;
  e._lastStats = performance.now();
  e._govNoteGpu(100, 10, 50);
  e._govMsStep = 1000;
  e._batch = 32;
  let frames = 0;
  while (esteps === 0 && frames < 12) { await e._frameLoop(); frames++; }
  assert.equal(esteps, 4, `decays 32→floor then submits (took ${frames} frames)`);
  assert.ok(frames <= 8, `escapes fast, not eventually (${frames})`);
  assert.equal(e._cutStreak, 0, 'work resumes reset the streak');
  assert.ok(elogs.some((m) => /submit cut: holding \d+ms idle \(batch 32→22, streak 1\)/.test(m)),
    'the first cut of a streak says so');
}

// ---- (1) uniform ring + merged submits: one submit per batch ----
{
  const { gpu, trainer } = await smallTrainer({ maxSplats: 6000, capMult: 2, shDeg: 0 });
  const dev = gpu.device;
  const projOf = (p) => p.bgs.find((bg) => trainer.bgProjectTrainR.includes(bg));

  // three bracketed steps merge into a single submit
  const s0 = dev.queue.submits, p0 = dev.passes.length, w0 = dev.queue.writes.length;
  trainer.beginGpuBatch();
  trainer.stepOnce(); trainer.stepOnce(); trainer.stepOnce();
  assert.equal(dev.queue.submits, s0, 'no submit until the batch closes');
  await trainer.endGpuBatch();
  assert.equal(dev.queue.submits, s0 + 1, 'three steps, one submit');
  const passes = dev.passes.slice(p0);
  assert.equal(passes.length, 3);
  assert.ok(passes[0].enc === passes[1].enc && passes[1].enc === passes[2].enc,
    'all three passes share the merged encoder');
  assert.equal(dev.queue.submitted[dev.queue.submitted.length - 1].enc, passes[0].enc,
    'the merged encoder is what gets submitted');
  assert.deepEqual(
    passes.map((p) => [p.ts.beginningOfPassWriteIndex, p.ts.endOfPassWriteIndex]),
    [[0, 1], [2, 3], [4, 5]], 'timestamp slots advance per step');
  for (const p of passes) {
    assert.equal(p.pipes[0], trainer.pipeTileZero, 'each pass opens with the counter reset');
    assert.equal(p.bgs[0], trainer.bgTileZero);
  }
  // write-slot == bind-slot, per step
  const uniWrites = dev.queue.writes.slice(w0).filter((w) => w.label === 'uniTrain');
  assert.equal(uniWrites.length, 3);
  for (let k = 0; k < 3; k++) {
    assert.equal(projOf(passes[k]).entries[0].resource.buffer, trainer.uniTrainR[k],
      `step ${k} binds slot ${k}`);
    assert.equal(uniWrites[k].buffer, trainer.uniTrainR[k], `step ${k} writes slot ${k}`);
  }
  assert.ok(!dev.queue.writes.slice(w0).some((w) => w.label === 'tileCnt'),
    'no queue-write counter resets survive (dispatches replaced them)');

  // unbracketed steps keep the legacy immediate submit on slot 0
  const s1 = dev.queue.submits;
  trainer.stepOnce(); trainer.stepOnce();
  assert.equal(dev.queue.submits, s1 + 2);
  assert.equal(trainer._mEnc, null);
  assert.ok(dev.passes.slice(-2).every((p) =>
    projOf(p).entries[0].resource.buffer === trainer.uniTrainR[0]));

  // the full ring flushes mid-batch; slots reuse after the submit separates them
  const { gpu: gpu2, trainer: t2 } = await smallTrainer({ maxSplats: 6000, capMult: 2, shDeg: 0 });
  const dev2 = gpu2.device;
  const s2 = dev2.queue.submits, p2 = dev2.passes.length;
  t2._msStepEst = 0.001;                                   // measured fast: the ring binds
  t2.beginGpuBatch();
  for (let i = 0; i < UNI_RING + 2; i++) t2.stepOnce();
  await t2.endGpuBatch();
  assert.equal(dev2.queue.submits, s2 + 2, 'flush + close');
  const ps = dev2.passes.slice(p2);
  assert.equal(ps.length, UNI_RING + 2);
  const proj2 = (p) => p.bgs.find((bg) => t2.bgProjectTrainR.includes(bg));
  assert.equal(proj2(ps[0]), proj2(ps[UNI_RING]), 'slot 0 reused after the flush');
  assert.equal(proj2(ps[1]), proj2(ps[UNI_RING + 1]));
  assert.ok(ps[0].enc !== ps[UNI_RING].enc, 'the flush opened a second encoder');

  // the SSAA twin uniforms ring too
  const { gpu: gpu3, trainer: t3 } =
    await smallTrainer({ maxSplats: 20000, capMult: 4, shDeg: 0, ssaa: 2 });
  const s3 = gpu3.device.queue.submits;
  t3.beginGpuBatch();
  t3.stepOnce(); t3.stepOnce();
  await t3.endGpuBatch();
  assert.equal(gpu3.device.queue.submits, s3 + 1);
  assert.equal(t3.uniTrain2fR.length, UNI_RING);
  assert.notEqual(t3.uniTrain2fR[0], t3.uniTrain2fR[1]);

  // legacy immediate paths ride the current slot: profile + eval + capture
  const prof = await trainer.profileSteps(2);
  assert.equal(prof.steps, 2);
  assert.ok(Number.isFinite(prof.msPerStep));
  const psnr = await trainer.evalCamPsnr(0);
  assert.equal(typeof psnr, 'number');                   // zero stats read Infinity: runs clean
  const cap = await trainer.renderCapture(
    { R: [1, 0, 0, 0, 1, 0, 0, 0, 1], t: [0, 0, 0], f: 50, w: 16, h: 12 });
  assert.equal(cap.w, 16);
}

// ---- (1) merged-submit cap: predicted GPU time bounds each submit ----
{
  // slow measured step: the budget binds before the ring
  const { gpu, trainer } = await smallTrainer({ maxSplats: 6000, capMult: 2, shDeg: 0 });
  trainer._msStepEst = 50;                                 // 250ms / 50 = 5 steps/submit
  assert.equal(trainer._mergeCap(), 5);
  const s0 = gpu.device.queue.submits;
  trainer.beginGpuBatch();
  for (let i = 0; i < 12; i++) trainer.stepOnce();
  await trainer.endGpuBatch();
  assert.equal(gpu.device.queue.submits, s0 + 3, '5 + 5 + 2');

  // unmeasured: the conservative cap binds (no multi-second first submits)
  const { gpu: gpu2, trainer: t2 } = await smallTrainer({ maxSplats: 6000, capMult: 2, shDeg: 0 });
  assert.equal(t2._mergeCap(), UNMEASURED_MERGE);
  const s1 = gpu2.device.queue.submits;
  t2.beginGpuBatch();
  for (let i = 0; i < UNMEASURED_MERGE + 2; i++) t2.stepOnce();
  await t2.endGpuBatch();
  assert.equal(gpu2.device.queue.submits, s1 + 2, '8 + 2');

  // fast measured step: the full ring merges
  t2._msStepEst = MERGE_BUDGET_MS / UNI_RING / 2;
  assert.equal(t2._mergeCap(), UNI_RING);
  const s2 = gpu2.device.queue.submits;
  t2.beginGpuBatch();
  for (let i = 0; i < 20; i++) t2.stepOnce();
  await t2.endGpuBatch();
  assert.equal(gpu2.device.queue.submits, s2 + 1);

  // refine moves the cost basis: the estimate resets for one careful batch
  t2._msStepEst = 50;
  await t2.refine();
  assert.equal(t2._msStepEst, null);
  assert.equal(t2._mergeCap(), UNMEASURED_MERGE);
}

// ---- (2b) batch growth needs the honest timer ----
{
  const s = createSession({});
  assert.equal(s._batchIdeal(15, 60, 30), 30);             // measured: grows
  const g = createSession({});
  assert.equal(g._batchIdeal(15, 5, null), 15);            // 1st null: stale speed held
  assert.ok(Math.abs(g._batchIdeal(15, 500, null) - 3.6) < 1e-9); // 2nd null: slowness shrinks
  assert.equal(g._batchIdeal(15, 5, null), 360);           // timerless: legacy law resumes
  assert.equal(g._batchIdeal(15, 5, 30), 360);              // measured: honest growth, misses reset…
  assert.equal(g._gpuMsMisses, 0);
  assert.equal(g._batchIdeal(15, 5, null), 15);            // …so one null holds again
}

// ---- lowMem re-seed says plainly that the pixels are gone ----
{
  const mk = (rgb) => {
    const s = createSession({ lowMem: true });
    s.recon = {
      points: synthPoints(20),
      cams: [{ imgIdx: 0, R: [1, 0, 0, 0, 1, 0, 0, 0, 1], t: [0, 0, 0], f: 50 }],
    };
    s.frames = [{ tw: 64, th: 48, fw: 64, fh: 48, rgb, alpha: null, emptyFrac: 0 }];
    return s;
  };
  await assert.rejects(mk(null).seed({ hull: false, maskPoints: false }), /reload to train again/);
  // positive control: pixels present sails past the guard into boot
  // (which then dies on the shim GPU — the guard is what we pin)
  await assert.rejects(mk(new Float32Array(64 * 48 * 3).fill(0.5)).seed({ hull: false, maskPoints: false }),
    /requestAdapter|WebGPU not available/);
}

// ---- opts.timestamps === false kills the timer, keeps the merge ----
{
  const { gpu, trainer } =
    await smallTrainer({ maxSplats: 6000, capMult: 2, shDeg: 0, timestamps: false });
  assert.equal(trainer._tsSet, null);
  const s0 = gpu.device.queue.submits;
  trainer.beginGpuBatch();
  trainer.stepOnce(); trainer.stepOnce();
  assert.equal(await trainer.endGpuBatch(), null);
  assert.equal(gpu.device.queue.submits, s0 + 1, 'merging is independent of the timer');
  const { trainer: timed } = await smallTrainer({ maxSplats: 6000, capMult: 2, shDeg: 0 });
  assert.ok(timed._tsSet, 'the feature present means the timer builds by default');
}

console.log('responsive loop (yields, hidden skips, pooled step scratch, duty governor) verified');
