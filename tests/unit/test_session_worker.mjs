import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { createSession, litSample } from '../../src/session.js';
import { RemoteTrainer } from '../../src/gs/remote-trainer.js';
import { mockWorkerFactory } from './mock_train_worker.mjs';

const sessionSource = readFileSync(new URL('../../src/session.js', import.meta.url), 'utf8');
const remoteSource = readFileSync(new URL('../../src/gs/remote-trainer.js', import.meta.url), 'utf8');
const tick = (ms = 25) => new Promise((r) => setTimeout(r, ms));

const frame = (imgIdx) => ({
  tw: 8, th: 8, fw: 8, fh: 8, name: `f${imgIdx}`,
  rgb: new Float32Array(8 * 8 * 3).fill(0.5),
  alpha: new Uint8Array(8 * 8).fill(255),
  sharpness: 1, sampleColor: () => [0.5, 0.5, 0.5], // functions must never reach the wire
});
const hull = { lo: [0, 0, 0], cell: 1, dim: [2, 2, 2], alive: new Uint8Array(8).fill(1) };
const boot = (over = {}) => ({
  gaussians: { data: new Float32Array(160), n: 10, dc: 'sigmoid' },
  cams: [{ imgIdx: 0 }, { imgIdx: 1 }],
  frames: [frame(0), frame(1)],
  maxW: 64, maxH: 64, radius: 5,
  hull, hullOpts: { sigma: 2 }, hullKillFn: () => false,
  protect: { from: 1, to: 3, until: 50 },
  freezePos: { from: 0, to: 2 },
  holdout: 1, excluded: [0], iter: 5,
  ...over,
});

// ---- _shouldOffload matrix ----
{
  assert.equal(createSession()._shouldOffload(), false, 'Node has no Worker: auto stays local');
  assert.equal(createSession({ trainWorker: false })._shouldOffload(), false);
  assert.throws(() => createSession({ trainWorker: true })._shouldOffload(), /Web Workers are unavailable/);
  assert.equal(createSession({ device: {} })._shouldOffload(), false,
    'a host-owned device can never cross threads');
  assert.throws(() => createSession({ device: {}, trainWorker: true })._shouldOffload(), /cannot move to a worker/);
  globalThis.Worker = class MockWorker {};
  try {
    assert.equal(createSession()._shouldOffload(), true, 'auto offloads when workers exist');
    assert.equal(createSession({ trainWorker: false })._shouldOffload(), false);
    assert.equal(createSession({ trainWorker: true })._shouldOffload(), true);
  } finally {
    delete globalThis.Worker;
  }
}

// ---- remote boot: payload projection, seed-time sets, mirror, shim ----
const host = {};
const logs = [];
const s = createSession({ trainWorker: true, maxIters: 100, createTrainWorker: mockWorkerFactory(host) });
s.on('log', (m) => logs.push(m));
await s._bootRemoteTrainer({ maxIters: 100, camOpt: true }, boot());
{
  const tr = host.trainer;
  assert.ok(s.trainer instanceof RemoteTrainer);
  assert.equal(s._remote, true);
  assert.deepEqual(tr.calls[0], ['setup', 10, 2, 2], 'seed + 2 cams + 2 images reach worker setup');
  assert.equal(typeof tr.hullKill, 'function', 'the hull test is rebuilt worker-side from data');
  assert.equal(tr.hullKill(100, 100, 100, 0.1), true, '... and behaves (far outside => kill)');
  assert.deepEqual(tr.protect, { from: 1, to: 3, until: 50 });
  assert.deepEqual(tr.calls.find((c) => c[0] === 'freezePos'), ['freezePos', 0, 2]);
  assert.equal(tr.holdout, 1);
  assert.ok(tr.excluded instanceof Set && tr.excluded.has(0));
  assert.equal(tr.iter, 5);
  assert.equal(tr.adamT0, 5);
  assert.equal(s.trainer.n, 10);
  assert.equal(s.trainer.cap, 40);
  assert.equal(s.trainer.iter, 5);
  assert.equal(s.trainer.camMeta.length, 2);
  assert.equal(s.trainer.dilate, 0.1, 'export-time scalars ride the mirror');
  assert.equal(s.trainer.mipComp, true);
  assert.deepEqual(s.gpu, { info: { vendor: 'mock', architecture: 'mock-gpu', device: '', description: '' }, owned: false, dispose: s.gpu.dispose },
    'adapter info arrives as the plain 4-field snapshot (raw GPUAdapterInfo cannot clone)');
  assert.ok(logs.some((m) => m.includes('(worker)')), 'the GPU line names the worker path');
}

// worker setup is idempotent across boots (fresh worker per boot)
{
  const host2 = {};
  const s2 = createSession({ trainWorker: true, createTrainWorker: mockWorkerFactory(host2) });
  await s2._bootRemoteTrainer({}, boot({ hull: null, hullKillFn: null, protect: null, freezePos: null }));
  assert.equal(host2.trainer.hullKill, null);
  assert.equal(host2.trainer.protect, null);
  s2.dispose();
  assert.ok(host2.terminated, 'dispose terminates the worker');
  assert.equal(s2.trainer, null);
}

// maskTraining:false hides alphas remotely, like _setupTrainer locally
{
  const host4 = {};
  let gotImages = null;
  const factory4 = mockWorkerFactory(host4, { wrapTrainer: (t) => {
    const orig = t.setup.bind(t);
    t.setup = function (g, c, im) { gotImages = im; return orig(g, c, im); };
  } });
  const s4 = createSession({ trainWorker: true, maskTraining: false, createTrainWorker: factory4 });
  await s4._bootRemoteTrainer({}, boot());
  assert.ok(gotImages && gotImages.length === 2);
  assert.ok(gotImages.every((im) => im.alpha === null), 'alphas hidden before upload');
  assert.ok(gotImages.every((im) => im.rgb.length === 192), 'pixels intact');
  assert.ok(!('sampleColor' in gotImages[0]), 'frame functions never reach the wire');
  s4.dispose();
}

// lowMem transfers ownership instead of copying
{
  const host5 = {};
  const s5 = createSession({ trainWorker: true, lowMem: true, createTrainWorker: mockWorkerFactory(host5) });
  const fr = [frame(0)];
  await s5._bootRemoteTrainer({}, boot({ frames: fr }));
  const initReq = host5.ports.mainPort.sent.find((m) => m.msg.m === 'init');
  assert.ok(initReq.xfer.length >= 2, `ownership moves buffers, got ${initReq.xfer.length} transferred`);
  assert.ok(initReq.xfer.includes(fr[0].rgb.buffer), 'frame pixels ride transferred under lowMem');
  s5.dispose();
}

// worker device loss reaches the session shell (own session: loss is sticky)
{
  const hostL = {};
  const sL = createSession({ trainWorker: true, createTrainWorker: mockWorkerFactory(hostL) });
  await sL._bootRemoteTrainer({}, boot());
  assert.equal(typeof hostL.gpu.onLost, 'function', 'remote boot arms the lost handler');
  hostL.gpu.onLost({ reason: 'unit-test', message: 'reclaimed' });
  await tick();
  assert.equal(sL.deviceLost, true);
  assert.equal(sL.training, false);
  sL.dispose();
}

// ---- _commitSplit both modes ----
{
  s.holdout = 0;
  await s._commitSplit(new Set([1]));
  assert.ok(host.trainer.excluded.has(1) && !host.trainer.excluded.has(0));
  assert.equal(host.trainer.holdout, 0);
  assert.ok(s.trainer.excluded.has(1), 'the mirror follows the commit');
  const local = createSession({ trainWorker: false });
  local.trainer = {};
  local.holdout = 1;
  await local._commitSplit(new Set([0]));
  assert.ok(local.trainer.excluded.has(0));
  assert.equal(local.trainer.holdout, 1);
}

// ---- _camDriftDone: remote poses cross back, local path preserved ----
const fullReport = { n: 2, rotDegMedian: 0.5, rotDegMax: 1.0, trnMedian: 0.01, trnMax: 0.02 };
{
  host.trainer.camDrift = () => fullReport;
  s.recon = { cams: [{ R: new Array(9).fill(0), t: [0, 0, 0] }, { R: new Array(9).fill(0), t: [0, 0, 0] }] };
  const nLogs = logs.length;
  await s._camDriftDone();
  assert.deepEqual(s.recon.cams[1].t, [1, 0, 0], 'moved poses write back into the reconstruction');
  assert.deepEqual(s.recon.cams[0].R, [1, 0, 0, 0, 1, 0, 0, 0, 1]);
  assert.ok(logs.slice(nLogs).some((m) => m.includes('pose optimisation: 2 cameras moved')),
    'the drift report logs identically');
  // camOpt off: silent early return
  const quiet = createSession({ trainWorker: true, createTrainWorker: mockWorkerFactory({}) });
  quiet.trainer = { opts: {} };
  quiet.recon = { cams: [] };
  await quiet._camDriftDone();
  // local: sync reads, same write-back
  const local = createSession({ trainWorker: false });
  const llogs = [];
  local.on('log', (m) => llogs.push(m));
  local.trainer = { opts: { camOpt: true }, camDrift: () => fullReport,
    camMeta: [{ R: [1, 0, 0, 0, 1, 0, 0, 0, 1], t: [5, 0, 0] }] };
  local.recon = { cams: [{ R: new Array(9).fill(0), t: [0, 0, 0] }] };
  await local._camDriftDone();
  assert.deepEqual(local.recon.cams[0].t, [5, 0, 0]);
  assert.ok(llogs.some((m) => m.includes('pose optimisation: 2 cameras moved')));
}

// ---- continueFor / evalTestPsnr on the remote path ----
{
  const max = await s.continueFor(50);
  assert.equal(max, 150);
  assert.equal(s.opts.maxIters, 150);
  assert.equal(host.trainer.horizon, 150, 'the worker horizon stretches with the resume');
  assert.equal(host.trainer.opts.maxIters, 150);
  assert.equal(s.training, true);
  s.pause();
  await tick(60);
  assert.equal(s.training, false);
  s.testCams = [0, 1];
  const t = await s.evalTestPsnr();
  assert.equal(t.psnr, 20.5);
  assert.deepEqual(t.frames.map((f) => f.imgIdx), [0, 1]);
  s.testCams = [];
  assert.equal(await s.evalTestPsnr(), null);
}

// ---- fallback: auto degrades (then fails locally), true throws the worker error ----
{
  const failing = () => { throw new Error('no-worker'); };
  const sAuto = createSession({ createTrainWorker: failing });
  globalThis.Worker = class MockWorker {};
  try {
    await assert.rejects(sAuto._bootTrainer({}, boot({ frames: [] })), /WebGPU not available/,
      'auto swallows the worker failure and attempts local training');
    assert.equal(sAuto._remote, false);
    const sStrict = createSession({ trainWorker: true, createTrainWorker: failing });
    await assert.rejects(sStrict._bootTrainer({}, boot({ frames: [] })), /no-worker/,
      'true surfaces the worker failure');
  } finally {
    delete globalThis.Worker;
  }
}

// ---- remote view transport: keyed attach, ordered commands, capture ----
{
  const offs = [];
  const canvasEl = (w, h) => ({
    width: w, height: h,
    transferControlToOffscreen() {
      const off = { width: 0, height: 0, getContext: () => ({ configure(c) { off._cfg = c; } }) };
      offs.push(off);
      return off;
    },
  });
  const cv = canvasEl(640, 480);
  const viewSends = () => host.ports.mainPort.sent.filter((m) => m.msg.m === 'view');
  await s.view.attach(cv);
  assert.equal(offs.length, 1, 'the element transfers once');
  assert.equal(offs[0].width, 640, 'attach carries the live size');
  const firstAttach = viewSends().find((m) => m.msg.p.cmd === 'attach');
  assert.equal(firstAttach.xfer.length, 1, 'first attach moves the offscreen');
  cv.width = 320; cv.height = 240;
  await s.view.attach(cv);
  assert.equal(offs.length, 1, 're-attach moves nothing');
  assert.equal(offs[0].width, 320, '... but resizes by key');
  assert.equal(viewSends().filter((m) => m.msg.p.cmd === 'attach').at(-1).xfer.length, 0);
  const meta = await s.view.lookThrough(1);
  assert.equal(meta.imgIdx, 1, 'lookThrough still answers from the mirror');
  assert.equal(s.view._offset, 64);
  const renders = host.trainer.calls.filter((c) => c[0] === 'renderView').length;
  assert.equal(await s.view.renderNow(), true);
  assert.equal(host.trainer.calls.filter((c) => c[0] === 'renderView').length, renders + 1);
  assert.equal(s.view._dirty, false, 'a submitted remote render clears dirty');
  const shot = await s.view.capture();
  assert.equal(shot.w, 8);
  assert.ok(shot.rgba instanceof Uint8ClampedArray && shot.rgba.length === 256);
  // fire-and-forget tick render (a rejection here must never go unhandled)
  s.view._dirty = true;
  s.view._tick(1, true);
  await tick();
  assert.ok(host.trainer.calls.filter((c) => c[0] === 'renderView').length >= renders + 2);
  // a canvas bound to a dead worker is a loud error, not a silent black view
  s._workerGen++;
  assert.throws(() => s.view.attach(cv), /fresh canvas/);
}

// ---- remote renderNow behaves like main's local viewer (2026-11 parity) ----
{
  const hostP = {};
  const sP = createSession({ trainWorker: true, maxIters: 100, createTrainWorker: mockWorkerFactory(hostP) });
  await sP._bootRemoteTrainer({ maxIters: 100 }, boot());
  sP.view.enabled = true;
  sP.view.camera = { w: 8, h: 8 };
  // accepted renders push back the auto-refresh, like the local branch —
  // without this the tick re-fires every frame during training
  sP.view._send = () => Promise.resolve(true);
  sP.view._lastAuto = 0;
  sP.view._dirty = true;
  assert.equal(await sP.view.renderNow(), true);
  assert.ok(sP.view._lastAuto > 0, 'an accepted remote render stamps the auto-refresh clock');
  assert.equal(sP.view._dirty, false);
  // latest-wins: a render already draining its round trip coalesces — the
  // next tick retries with the newest camera instead of queueing stale
  // renders behind the chain (continuous orbiting used to backlog them all)
  let release = null;
  const held = new Promise((r) => { release = r; });
  let sends = 0;
  sP.view._send = () => { sends++; return held; };
  const p1 = sP.view.renderNow();
  p1.catch(() => {});
  assert.equal(sP.view._remoteInFlight, true);
  sP.view._dirty = false;
  assert.equal(await sP.view.renderNow(), true, 'a coalesced render resolves accepted');
  assert.equal(sends, 1, '...without queueing a second send');
  assert.equal(sP.view._dirty, true, '...keeping dirty for the retry');
  assert.equal(sP.view._falseSince, null, 'coalescing is not a refusal: the streak stays untouched');
  release(true);
  assert.equal(await p1, true);
  assert.equal(sP.view._remoteInFlight, false);
  // force (snapshots) bypasses the gate
  let release2 = null;
  const held2 = new Promise((r) => { release2 = r; });
  sP.view._send = () => { sends++; return held2; };
  const q1 = sP.view.renderNow();
  q1.catch(() => {});
  const q2 = sP.view.renderNow(true);
  q2.catch(() => {});
  assert.equal(sends, 3, 'force submits unconditionally');
  release2(true);
  assert.deepEqual(await Promise.all([q1, q2]), [true, true]);
  // a rejecting send drops the gate (and still rejects to the caller)
  sP.view._send = () => Promise.reject(new Error('port torn down'));
  await assert.rejects(sP.view.renderNow(), /torn down/);
  assert.equal(sP.view._remoteInFlight, false, 'rejections must not wedge the gate');
  // teardown drops the gate with the chain (aborted sends never settle)
  sP.view._remoteInFlight = true;
  sP._terminateWorker();
  assert.equal(sP.view._remoteInFlight, false);
  sP.dispose();
}

// ---- worker port errors surface (2026-11: silent no-start + black view) ----
{
  const hostE = {};
  const sE = createSession({ trainWorker: true, maxIters: 100, createTrainWorker: mockWorkerFactory(hostE) });
  await sE._bootRemoteTrainer({ maxIters: 100 }, boot());
  const errs = [];
  sE.on('event', (e) => { if (e.kind === 'worker-error') errs.push(e); });
  sE.training = true;
  const inFlight = sE.trainer.runSteps(2);
  hostE.ports.mainPort.onerror({ message: 'boom' });
  await assert.rejects(inFlight, /boom/, 'in-flight calls fail fast on worker error');
  await tick();
  assert.equal(errs.length, 1, 'one worker-error event per fault');
  assert.equal(errs[0].message, 'boom');
  assert.equal(sE.training, false, 'a faulted worker stops the loop (start() can revive)');
  sE.dispose();
}

// pre-boot worker faults stay quiet: seed()'s throw (or the local fallback) owns them
{
  const sB = createSession({ trainWorker: true, createTrainWorker: mockWorkerFactory({}) });
  const evts = [];
  sB.on('event', (e) => evts.push(e));
  sB._workerFailed(new Error('pre-boot'));
  assert.equal(evts.length, 0);
  sB.dispose();
}

// ---- _frameFailed: a throwing frame stops the loop LOUDLY ----
{
  const hostF = {};
  const sF = createSession({ trainWorker: true, maxIters: 100, createTrainWorker: mockWorkerFactory(hostF) });
  await sF._bootRemoteTrainer({ maxIters: 100 }, boot());
  const evts = [], flogs = [];
  sF.on('event', (e) => evts.push(e));
  sF.on('log', (m) => flogs.push(m));
  sF.training = true;
  sF._frameFailed(new Error('gpu hiccup'));
  assert.equal(sF.training, false, 'training must not stay true (start() no-ops while it is)');
  const terr = evts.find((e) => e.kind === 'train-error');
  assert.ok(terr, 'a train-error event explains the stop');
  assert.match(terr.message, /gpu hiccup/);
  assert.ok(flogs.some((m) => m.includes('frame loop: gpu hiccup')), 'the log keeps its line');
  sF.start();
  assert.equal(sF.training, true, 'start() revives after a train-error');
  sF.pause();
  await tick(60);

  // the instant-fence detector: three meaningless fences engage drain pacing
  const dF = createSession({});
  dF._noteFence(0.2, 32, null);
  assert.equal(dF._instantFences, undefined, 'one instant fence is only suspicion');
  assert.equal(dF._instantSuspect, true);
  dF._noteFence(0.1, 32, null);
  dF._noteFence(0.3, 32, null);
  assert.equal(dF._instantFences, true, 'three in a row engage drain pacing');
  assert.equal(dF._instantChecked, true);
  // ...but a timed batch or a healthy fence concludes it the other way
  const dT = createSession({});
  dT._noteFence(0.1, 32, null);
  dT._noteFence(0.1, 32, 12);
  assert.equal(dT._instantFences, undefined, 'a batch timer makes fences irrelevant');
  assert.equal(dT._instantChecked, true);
  const dH = createSession({});
  dH._noteFence(0.1, 32, null);
  dH._noteFence(41, 32, null);
  assert.equal(dH._instantFences, undefined, 'a fence that actually gated is healthy');
  assert.equal(dH._instantChecked, true);
  const dN = createSession({});
  dN._noteFence(null, 32, null);
  assert.equal(dN._instantSuspect, undefined, 'nothing gated = no signal either way');
  const dS = createSession({});
  dS._noteFence(0.1, 2, null);
  assert.equal(dS._instantSuspect, undefined, 'trivial batches settle fast legitimately');

  // the stall watchdog: a frame that never completes surfaces train-stall,
  // re-arms while stuck, and disarms when the frame lands
  const hostW = {};
  const sW = createSession({ trainWorker: true, maxIters: 100,
    createTrainWorker: mockWorkerFactory(hostW), stallMs: 30 });
  await sW._bootRemoteTrainer({ maxIters: 100 }, boot());
  sW._ensureScheduler();   // the closure carries the watchdog; rAF degrades to setTimeout here
  const stalls = [];
  sW.on('event', (e) => { if (e.kind === 'train-stall') stalls.push(e); });
  sW.training = true;
  let releaseFrame = null;
  sW._frameLoop = () => new Promise((r) => { releaseFrame = r; });
  sW._scheduleFrame();
  await tick(100);
  assert.ok(stalls.length >= 2, `stuck frame surfaces repeating train-stall (got ${stalls.length})`);
  assert.match(String(stalls[0].frameMs), /30/);
  releaseFrame();
  const landed = stalls.length;
  await tick(80);
  assert.equal(stalls.length, landed, 'a landed frame disarms the watchdog');
  assert.equal(sW._stallT._destroyed !== false, true, 'the timer is cleared');
  // ...and a failing frame still routes to _frameFailed
  let failed = null;
  sW._frameFailed = (e) => { failed = e; };
  sW._frameLoop = () => Promise.reject(new Error('boom'));
  sW._scheduleFrame();
  await tick(20);
  assert.match(String((failed && failed.message) || ''), /boom/);
  sW.training = false;
  sW.dispose();

  // the no-present streak: sustained silent renderNow=false surfaces view-error
  const hostV = {};
  const sV = createSession({ trainWorker: true, maxIters: 100,
    createTrainWorker: mockWorkerFactory(hostV), viewStallMs: 20 });
  await sV._bootRemoteTrainer({ maxIters: 100 }, boot());
  sV.view.enabled = true;
  sV.view.camera = { w: 4, h: 4 };
  sV.view._send = () => Promise.resolve(false);   // refused, never rejected
  const verrs = [];
  sV.on('event', (e) => { if (e.kind === 'view-error') verrs.push(e); });
  await sV.view.renderNow();
  assert.equal(verrs.length, 0, 'a lone refusal stays silent (transient)');
  // the probe capture decides the diagnosis: lit pixels = the worker
  // renders fine and only the present fails...
  const lit = new Uint8ClampedArray(64 * 4);
  for (let i = 0; i < 64; i += 2) lit[i * 4] = 200;   // half the sample lit
  sV.view.capture = async () => ({ w: 8, h: 8, rgba: lit });
  await tick(40);
  await sV.view.renderNow();
  assert.equal(verrs.length, 1, 'the streak past the window surfaces');
  assert.equal(verrs[0].cmd, 'renderNow');
  assert.match(verrs[0].message, /no frames/);
  assert.equal(verrs[0].diagnosis, 'presentation');
  assert.match(verrs[0].message, /present fails, not the raster/);
  await sV.view.renderNow();
  assert.equal(verrs.length, 1, 'one streak = one emission');
  // ...a black capture blames the raster/camera instead
  sV.view._viewFailed = false; sV.view._falseSince = null;
  sV.view.capture = async () => ({ w: 8, h: 8, rgba: new Uint8ClampedArray(64 * 4) });
  await sV.view.renderNow();
  await tick(40);
  await sV.view.renderNow();
  assert.equal(verrs.length, 2);
  assert.equal(verrs[1].diagnosis, 'raster');
  assert.match(verrs[1].message, /rasterizes nothing/);
  // ...and a throwing probe is itself a diagnosis
  sV.view._viewFailed = false; sV.view._falseSince = null;
  sV.view.capture = async () => { throw new Error('readback blew up'); };
  await sV.view.renderNow();
  await tick(40);
  await sV.view.renderNow();
  assert.equal(verrs.length, 3);
  assert.equal(verrs[2].diagnosis, 'capture-failed');
  assert.match(verrs[2].message, /readback blew up/);
  sV.view._send = () => Promise.resolve(true);    // recovery: a present lands
  assert.equal(await sV.view.renderNow(), true);
  sV.view._send = () => Promise.resolve(false);
  await sV.view.renderNow(true);
  assert.equal(verrs.length, 3, 'a successful present restarts the clock');
  sV.dispose();
  const n0 = evts.length;
  sF._frameFailed(new Error('after stop'));
  assert.equal(evts.length, n0, 'no event when the loop already stopped');
  sF._lost = true; sF.training = true;
  sF._frameFailed(new Error('device gone'));
  assert.equal(evts.length, n0, 'device loss owns its event (no train-error echo)');
  sF.dispose();
}

// ---- integration: a runBatch throw kills the scheduled frame loudly ----
{
  const hostD = {};
  const factoryD = mockWorkerFactory(hostD, { wrapTrainer: (t) => {
    t.stepOnce = () => { throw new Error('cl-owned-zero-copy'); };
  } });
  const sD = createSession({ trainWorker: true, maxIters: 100, createTrainWorker: factoryD });
  await sD._bootRemoteTrainer({ maxIters: 100 }, boot());
  const evts = [];
  sD.on('event', (e) => evts.push(e));
  sD.training = true;
  sD._ensureScheduler();
  sD._sched.raf = (fn) => { sD._rafFn = fn; };   // capture the scheduled frame
  sD._scheduleFrame();
  assert.ok(sD._rafFn, 'the frame schedules');
  sD._rafFn();
  await tick(60);
  assert.equal(sD.training, false, 'the failed frame stops the loop instead of wedging it');
  const terr = evts.find((e) => e.kind === 'train-error');
  assert.ok(terr, 'train-error surfaces the worker message');
  assert.match(terr.message, /cl-owned-zero-copy/);
  sD.dispose();
}

// ---- _terminateWorker aborts in-flight RPCs and drops the hung view chain ----
{
  const hostT = {};
  const sT = createSession({ trainWorker: true, maxIters: 100, createTrainWorker: mockWorkerFactory(hostT) });
  await sT._bootRemoteTrainer({ maxIters: 100 }, boot());
  hostT.ports.workerPort.onmessage = null;   // the worker answers nothing, as when dead
  const hung = sT.trainer.runSteps(2);
  const hungView = sT.view._send('renderNow', {});
  hungView.catch(() => {});   // the assertion below re-checks; avoid interim unhandled noise
  await tick(0);              // _send posts a microtask later — let both calls get in flight
  assert.ok(sT.view._chain, 'the view command queued behind the chain');
  sT._terminateWorker();
  await assert.rejects(hung, /terminated/);
  await assert.rejects(hungView, /terminated/);
  assert.equal(sT.view._chain, null, 'terminate drops the hung tail (post-recovery commands flow)');
  assert.equal(sT.view._viewFailed, true, 'teardown aborts are not view failures');
  assert.equal(sT._remote, false);
  sT.dispose();
}

// ---- attach failure: no poison, one view-error per streak, retry carries the canvas ----
{
  const hostV = {};
  const sV = createSession({ trainWorker: true, maxIters: 100, createTrainWorker: mockWorkerFactory(hostV) });
  await sV._bootRemoteTrainer({ maxIters: 100 }, boot());
  const evts = [];
  sV.on('event', (e) => evts.push(e));
  // an offscreen whose context fails worker-side (the phone's OffscreenCanvas gap)
  const cv = { width: 640, height: 480, transferControlToOffscreen: () => ({ width: 0, height: 0 }) };
  await assert.rejects(sV.view.attach(cv), /getContext/);
  await tick();
  const verrs = () => evts.filter((e) => e.kind === 'view-error');
  assert.equal(verrs().length, 1, 'the failed attach surfaces once');
  assert.equal(verrs()[0].cmd, 'attach');
  assert.equal(sV.view._offs.get(cv).sent, false, 'a failed attach retries WITH the canvas');
  await assert.rejects(sV.view.attach(cv), /getContext/);
  const attaches = hostV.ports.mainPort.sent.filter((m) => m.msg.m === 'view' && m.msg.p.cmd === 'attach');
  assert.equal(attaches.length, 2);
  assert.equal(attaches[1].xfer.length, 1, 'the retry carries the canvas again');
  await tick();
  assert.equal(verrs().length, 1, 'one event per failure streak');
  await sV.view.setEnabled(false);
  await assert.rejects(sV.view.attach(cv), /getContext/);
  await tick();
  assert.equal(verrs().length, 2, 'a success re-arms the event');
  sV.dispose();
}

// ---- resync re-pushes enabled + the live camera behind a re-attach ----
{
  const hostR = {};
  const sR = createSession({ trainWorker: true, maxIters: 100, createTrainWorker: mockWorkerFactory(hostR) });
  await sR._bootRemoteTrainer({ maxIters: 100 }, boot());
  const cv = { width: 320, height: 240,
    transferControlToOffscreen: () => ({ width: 0, height: 0, getContext: () => ({ configure() {} }) }) };
  await sR.view.attach(cv);
  await sR.view.lookThrough(1);
  assert.equal(sR.view._lookIdx, 1, 'lookThrough records its frame');
  sR.view._viewFailed = true;
  const n0 = hostR.ports.mainPort.sent.length;
  assert.equal(await sR.view.resync(), true);
  await tick(0);   // the chain posts a microtask later
  const cmds = hostR.ports.mainPort.sent.slice(n0).map((m) => m.msg.p.cmd);
  assert.ok(cmds.includes('setEnabled'), 'resync re-pushes enabled');
  assert.ok(cmds.includes('lookThrough'), 'resync re-pushes the frame camera');
  assert.equal(sR.view._viewFailed, false, 'resync starts a fresh failure streak');
  sR.view.setCamera({ R: [], t: [], f: 1, cx: 0, cy: 0, w: 8, h: 8 });
  assert.equal(sR.view._lookIdx, null, 'setCamera clears the frame camera');
  const n1 = hostR.ports.mainPort.sent.length;
  await sR.view.resync();
  await tick(0);
  const cmds2 = hostR.ports.mainPort.sent.slice(n1).map((m) => m.msg.p.cmd);
  assert.ok(cmds2.includes('setCamera') && !cmds2.includes('lookThrough'),
    'resync re-pushes the free camera instead');
  const local = createSession({ trainWorker: false });
  assert.equal(await local.view.resync(), false, 'local sessions have no worker state to push');
  sR.dispose();
}
s.dispose();

// ---- source pins: the local path keeps its exact shape ----
assert.match(sessionSource, /const s = await trainer\.runSteps\(batch\);\n        gpuMs = s\.gpuMs \?\? null;\n        fenceMs = s\.fenceMs \?\? null;/,
  'the frame loop takes the batch GPU time AND gated-fence ms from the remote result');
assert.match(sessionSource, /if \(typeof trainer\.beginGpuBatch === 'function'\) trainer\.beginGpuBatch\(\);/,
  'the frame loop brackets local steps with the batch GPU timer');
assert.match(sessionSource, /for \(let k = 0; k < batch; k\+\+\) trainer\.stepOnce\(\);/,
  'the frame loop steps locally');
assert.match(sessionSource, /this\._govNoteGpu\(gpuMs, batch, trainer\.n\);/,
  'the frame loop feeds the batch GPU time to the governor');
assert.match(sessionSource, /const cutMs = this\._submitCutMs\(batch\);/,
  'the frame loop cuts submits over the queued-work budget');
assert.match(sessionSource, /if \(!this\._remote\) \{\n        this\._fences\.push\(/,
  'the fence ring is local-only (remote gates worker-side)');
assert.match(sessionSource, /if \(this\._remote\) await trainer\.sync\(\);\n      else await trainer\.device\.queue\.onSubmittedWorkDone\(\);/,
  'the idle branch drains remotely via sync()');
assert.match(sessionSource, /if \(!this\.gpu && !this\._shouldOffload\(\)\) \{/,
  'solve skips the session device when the worker will own one');
assert.match(sessionSource, /const dilate = trainerOpts\.dilate \?\? 0\.1;/,
  'unbake derives dilate from opts (no trainer needed pre-boot)');
assert.match(sessionSource, /if \(boot\.hullKillFn\) this\.trainer\.hullKill = boot\.hullKillFn;/,
  'local boot keeps the prebuilt hull test (no behaviour change)');
assert.match(sessionSource, /training on the main thread \(trainWorker disabled\)/,
  'an explicitly local session says so in the log (which path ran is diagnosable)');
assert.match(sessionSource, /this\._chain = p\.catch\(\(\) => \{\}\);/,
  'remote view commands serialize through a chain');
assert.match(sessionSource, /attach a fresh canvas/,
  'stale canvas transfers fail loudly after recover()');
assert.match(sessionSource, /async capture\(\) \{\n    if \(!this\.camera\) throw new Error\('view\.capture: no camera'\);/,
  'the view offers canvas-free capture in both modes');
assert.match(sessionSource, /await this\._commitSplit\(excluded\);/,
  'the training split commits once (single remote round trip)');
assert.match(sessionSource, /this\._frameLoop\(\)\.then\(\n        \(\) => \{ if \(this\._frameSeq === id\) clearTimeout\(this\._stallT\); \},\n        \(e\) => \{ if \(this\._frameSeq === id\) clearTimeout\(this\._stallT\); this\._frameFailed\(e\); \}\);\n/,
  'frame failures stop the loop loudly instead of wedging training=true (and land the stall watchdog)');
assert.match(sessionSource, /kind: 'train-error'/,
  'a throwing frame emits train-error (the app offers resume)');
assert.match(sessionSource, /kind: 'worker-error'/,
  'worker faults emit worker-error (the app reboots the worker)');
assert.match(sessionSource, /kind: 'view-error'/,
  'rejected view commands emit view-error (the app re-mounts the canvas)');
assert.match(sessionSource, /rec\.sent = false/,
  'a failed attach retries with the canvas (no poisoned element)');
assert.match(sessionSource, /resync\(\) \{/,
  'the view can re-push enabled + camera behind a re-attach');
assert.match(sessionSource, /if \(this\.view\) \{\n      this\.view\._chain = null;/,
  'worker teardown drops the hung view chain');
assert.match(remoteSource, /this\._workerErrored\(e && e\.message/,
  'port errors notify the session, not just failAll');
assert.match(remoteSource, /abort\(err\) \{\n    this\._rpc\.failAll/,
  'teardown aborts in-flight RPCs so nothing hangs silently');

// ---- hang hardening (instant fences, stall surfacing, silent-black detection) ----
assert.match(sessionSource, /_noteFence\(fenceMs, batch, gpuMs\)/,
  'every frame feeds gated-fence ms to the instant-fence detector');
assert.match(sessionSource, /batch = Math\.min\(batch, 8\);/,
  'timerless + pacing suspect holds small batches until the fence check concludes');
assert.match(sessionSource, /await this\._locked\(\(\) => trainer\.readLoss\(\)\);/,
  'drain pacing leans on the readback when fences are meaningless');
assert.match(sessionSource, /kind: 'train-stall'/,
  'a wedged frame surfaces train-stall (the app names the stall, the loop keeps owning)');
assert.match(sessionSource, /_falseSince/,
  'sustained silent render refusal gets a streak clock');
assert.match(sessionSource, /worker presented no frames/,
  'the no-present streak surfaces as view-error (the app falls back to snapshots)');
assert.match(sessionSource, /_diagnoseNoPresent\(\)/,
  'the streak probes a canvas-free capture before blaming a link');
assert.match(sessionSource, /diagnosis: d\.kind/,
  'the no-present view-error carries the probe diagnosis');
assert.match(sessionSource, /const stallMs = this\.opts\.stallMs \?\? 60000;/,
  'the frame stall watchdog is tunable (tests arm it fast)');

// ---- litSample: the shared "renders content" sampler ----
{
  const black = new Uint8ClampedArray(64 * 64 * 4);   // all zeros
  assert.deepEqual(litSample(black), { lit: 0, n: 2048, frac: 0 });
  const white = new Uint8ClampedArray(64 * 64 * 4).fill(255);
  const w = litSample(white);
  assert.equal(w.lit, w.n);
  assert.equal(w.frac, 1);
  // near-black floor: 12 and under is dark, 13+ is lit (any channel)
  const edge = new Uint8ClampedArray([12, 12, 12, 255, 0, 0, 13, 255, 5, 5, 5, 255]);
  assert.deepEqual(litSample(edge), { lit: 1, n: 3, frac: 1 / 3 });
  // short reads sample every pixel (no stride gaps on thumbnails)
  const tiny = new Uint8ClampedArray([200, 0, 0, 255]);
  assert.deepEqual(litSample(tiny), { lit: 1, n: 1, frac: 1 });
  // empty / missing readbacks are dark, never NaN
  assert.deepEqual(litSample(null), { lit: 0, n: 0, frac: 0 });
  assert.deepEqual(litSample(new Uint8ClampedArray(3)), { lit: 0, n: 0, frac: 0 });
  // large frames stride down to ~2k samples (Uint8ClampedArray and plain alike)
  const big = new Uint8Array(1920 * 1080 * 4);
  big[100] = 255;
  const b = litSample(big);
  assert.ok(b.n > 2000 && b.n <= 2100, `strided to ~2k samples (got ${b.n})`);
  assert.match(sessionSource, /const s = litSample\(shot && shot\.rgba\);/,
    'the no-present probe samples through the shared sampler');
}

console.log('session worker wiring (boot, split, drift, loop, view) verified');
