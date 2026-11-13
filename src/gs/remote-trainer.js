// remote-trainer.js — main-thread proxy for a GSTrainer living in trainworker.js.
//
// Mirrors the scalars the Session shell reads (iter, n, cap, camMeta, ...) so
// session code keeps working with either a local GSTrainer or this proxy;
// every GPU touch becomes one id-correlated RPC. Big uploads (seed, targets)
// are structured-cloned by default; pass transferOwnership (the lowMem path,
// where the session drops its CPU copies after setup anyway) to move the
// buffers instead of copying them.

import { RpcEndpoint, collectBuffers } from './trainer-protocol.js';

const mirrorKeys = ['n', 'cap', 'iter', 'tileGrad', 'shK', 'dcMode', 'dilate',
  'mipComp', 'horizon', 'holdout', 'lastCam', 'camMeta', 'gpuInfo'];

export class RemoteTrainer {
  /** port: the Worker (or mock port). trainerOpts: main-side copy kept for
   *  reads; behaviour-affecting mutations go through setConfig. */
  constructor(port, trainerOpts = {}) {
    this.opts = { ...trainerOpts };
    this.excluded = new Set();
    // session-side cadence state only (the trainer never reads it)
    this.lastRefine = 0;
    // mirrors, filled by setupRemote and refreshed per batch
    this.n = 0; this.cap = 0; this.iter = 0;
    this.camMeta = [];
    this._listeners = new Map();
    this._rpc = new RpcEndpoint(port);
    this._rpc.onEvent((kind, payload) => {
      if (kind === 'device-lost') {
        // readbacks on the dead device never resolve — fail fast, as _locked
        this._rpc.failAll(new Error(`GPU device lost (${(payload && payload.reason) || 'unknown'})`));
      }
      const fns = this._listeners.get(kind);
      if (fns) for (const fn of fns) fn(payload);
    });
    port.addEventListener
      ? port.addEventListener('error', (e) => this._workerErrored(e && e.message ? new Error(e.message) : new Error('worker error')))
      : (port.onerror = (e) => this._workerErrored(e && e.message ? new Error(e.message) : new Error('worker error')));
  }

  /** An uncaught worker failure: fail the in-flight calls AND tell the
   *  session shell (failAll alone left training=true with a dead loop and a
   *  black view — silent on both counts, 2026-11). */
  _workerErrored(err) {
    this._rpc.failAll(err);
    const fns = this._listeners.get('worker-error');
    if (fns) for (const fn of fns) fn(err);
  }

  /** Reject every in-flight call (session teardown; the worker is gone and
   *  its answers are never coming — without this new calls hang silently). */
  abort(err) {
    this._rpc.failAll(err instanceof Error ? err : new Error(String(err || 'training worker terminated')));
  }

  onEvent(kind, fn) {
    if (!this._listeners.has(kind)) this._listeners.set(kind, []);
    this._listeners.get(kind).push(fn);
  }

  /** Remote twin of GSTrainer.create + setup: uploads everything, returns the
   *  state mirror. payload: { gaussians, cams, images, maxViewW, maxViewH,
   *  radius, hull?, hullOpts?, protect?, freezePos?, holdout?, excluded?,
   *  iter?, transferOwnership? }. */
  async setupRemote(payload) {
    const { transferOwnership, ...rest } = payload;
    const transfer = transferOwnership ? collectBuffers(rest) : [];
    const mirror = await this._rpc.call('init', rest, transfer);
    this._applyMirror(mirror);
    if (payload.excluded) this.excluded = new Set(payload.excluded);
    if (payload.holdout !== undefined) this.holdout = payload.holdout;
    return mirror;
  }

  /** One frame's worth of training, fence-gated worker-side. */
  async runSteps(steps) {
    const s = await this._rpc.call('runBatch', { steps });
    this.iter = s.iter; this.n = s.n; this.lastCam = s.lastCam;
    return s;
  }

  async refine() {
    const r = await this._rpc.call('refine', {});
    this.iter = r.iter; this.n = r.n;
    return r;
  }

  async softPrune(ratio = null, opts = null) {
    const r = await this._rpc.call('softPrune', { ratio, skipRatchet: !!(opts && opts.skipRatchet) });
    this.iter = r.iter; this.n = r.n;
    return r;
  }

  async pruneDead() {
    const r = await this._rpc.call('pruneDead', {});
    this.iter = r.iter; this.n = r.n;
    return r;
  }

  async readLoss() {
    const r = await this._rpc.call('readLoss', {});
    if (r.entryOverflowTiles) this.entryOverflowTiles = r.entryOverflowTiles;
    return r.mse;
  }

  evalCamPsnr(ci) { return this._rpc.call('eval', { kind: 'psnr', ci }); }
  evalCamMetrics(ci) { return this._rpc.call('eval', { kind: 'metrics', ci }); }
  evalCamPsnrRefined(ci, steps) { return this._rpc.call('eval', { kind: 'refined', ci, steps }); }
  evalTestPsnr(cams) { return this._rpc.call('testPsnr', { cams }); }
  readGaussians() { return this._rpc.call('readGaussians', {}); }
  camDrift() { return this._rpc.call('camDrift', {}); }
  capture(cam) { return this._rpc.call('capture', { cam }); }
  sync() { return this._rpc.call('sync', {}); }

  /** Push an allow-listed live set (see trainworker CONFIG_KEYS); known keys
   *  also update the main-side mirror. */
  async setConfig(patch) {
    const ok = await this._rpc.call('config', { patch });
    if (patch.holdout !== undefined) this.holdout = patch.holdout;
    if (patch.excluded !== undefined) this.excluded = new Set(patch.excluded);
    if (patch.maxIters !== undefined) {
      this.opts.maxIters = patch.maxIters;
      this.horizon = patch.maxIters;
    }
    if (patch.iter !== undefined) this.iter = patch.iter;
    return ok;
  }

  setFreezePos(from, to) { return this.setConfig({ freezePos: { from, to } }); }

  /** Viewport command. attach's canvas must be passed in transfer. */
  viewCmd(cmd, params = {}, transfer) {
    return this._rpc.call('view', { cmd, ...params }, transfer);
  }

  async dispose() {
    try { await this._rpc.call('dispose', {}); } catch { /* already gone */ }
  }

  _applyMirror(m) {
    for (const k of mirrorKeys) if (m[k] !== undefined) this[k] = m[k];
  }
}
