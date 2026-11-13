// trainer-protocol.js — the main-thread <-> training-worker contract.
//
// The GPU-owning training core (device + GSTrainer + batch driver + view
// renderer) can run in a dedicated worker so encoding, submits and readback
// continuations never compete with input handling on the main thread. This
// module is DOM-free on purpose: both ends (trainworker.js, remote-trainer.js)
// and the Node unit tests share it.
//
// Wire format (compact keys; `p` stays structured-cloneable):
//   main -> worker  { t:'req', id, m:<method>, p:<params> }
//   worker -> main  { t:'res', id, ok, v:<value> }  or  { t:'res', id, ok:false, e:<message> }
//   worker -> main  { t:'evt', k:<kind>, p:<payload> }   (log, device-lost; no reply)
// Big buffers ride the postMessage transfer list, never the clone path.

/** Methods the worker handler serves (RemoteTrainer is the typed caller). */
export const TRAIN_METHODS = [
  'init',         // build device+trainer, upload seed+targets -> state mirror
  'runBatch',     // run N training steps with fence gating -> { iter, n, lastCam }
  'refine',       // densification barrier -> refine stats
  'softPrune',    // importance-cull barrier (params.ratio overrides, params.skipRatchet frees milestones) -> { culled, n }
  'readLoss',     // accumulated error -> { mse, entryOverflowTiles }
  'eval',         // { kind: 'psnr'|'metrics'|'refined', ci, steps? } -> metric(s)
  'testPsnr',     // mean PSNR over cameras -> { psnr, frames }
  'readGaussians',// model download (transferred) -> { data, n, sh, shK, dc }
  'camDrift',     // pose-optimisation report + moved poses -> { report, poses }
  'config',       // allow-listed live sets (holdout, excluded, growLimit, ...) -> true
  'view',         // viewport commands against a transferred OffscreenCanvas
  'capture',      // offscreen render + pixel download (transferred) -> { w, h, rgba }
  'sync',         // resolve when the device drained its queue -> true
  'dispose',      // release device + trainer -> true
];

/** Events the worker emits (no reply expected). */
export const TRAIN_EVENTS = ['log', 'device-lost'];

/** Collect the transferable ArrayBuffers behind a value: every TypedArray /
 *  DataView buffer found in plain objects/arrays, deduplicated. SharedArray-
 *  Buffers are skipped (transferring one throws). OffscreenCanvas and other
 *  non-view transferables must be added by the caller. */
export function collectBuffers(value, into = []) {
  const seen = new Set(into);
  const visit = (v) => {
    if (!v || typeof v !== 'object') return;
    if (ArrayBuffer.isView(v)) {
      const b = v.buffer;
      if (b instanceof ArrayBuffer && !seen.has(b)) { seen.add(b); into.push(b); }
      return;
    }
    if (v instanceof ArrayBuffer) {
      if (!seen.has(v)) { seen.add(v); into.push(v); }
      return;
    }
    if (typeof SharedArrayBuffer !== 'undefined' && v instanceof SharedArrayBuffer) return;
    if (Array.isArray(v)) { for (const x of v) visit(x); return; }
    const proto = Object.getPrototypeOf(v);
    if (proto !== Object.prototype && proto !== null) return; // class instances: opaque
    for (const k of Object.keys(v)) visit(v[k]);
  };
  visit(value);
  return into;
}

/** Request/response + event demultiplexer over any MessagePort-like:
 *  { postMessage(msg, transfer?), onmessage }. Both ends use it; unknown or
 *  late responses (failed/timed-out calls) are ignored. */
export class RpcEndpoint {
  constructor(port) {
    this._port = port;
    this._nextId = 1;
    this._pending = new Map();
    this._handler = null;
    this._evtHandler = null;
    port.onmessage = (e) => { this._route(e && e.data); };
  }

  /** Serve incoming requests: async (method, params) -> { value, transfer? }. */
  setHandler(fn) { this._handler = fn; }

  /** Receive-вы notifications: (kind, payload) -> void. */
  onEvent(fn) { this._evtHandler = fn; }

  get pending() { return this._pending.size; }

  call(method, params, transfer) {
    const id = this._nextId++;
    return new Promise((resolve, reject) => {
      this._pending.set(id, { resolve, reject });
      this._post({ t: 'req', id, m: method, p: params }, transfer);
    });
  }

  notify(kind, payload, transfer) {
    this._post({ t: 'evt', k: kind, p: payload }, transfer);
  }

  /** Reject every in-flight call (worker error, teardown, device loss). */
  failAll(err) {
    if (!this._pending.size) return;
    const e = err instanceof Error ? err : new Error(String(err));
    for (const { reject } of this._pending.values()) reject(e);
    this._pending.clear();
  }

  _post(msg, transfer) {
    if (transfer && transfer.length) this._port.postMessage(msg, transfer);
    else this._port.postMessage(msg);
  }

  async _route(msg) {
    if (!msg || typeof msg !== 'object') return;
    if (msg.t === 'res') {
      const p = this._pending.get(msg.id);
      if (!p) return;
      this._pending.delete(msg.id);
      if (msg.ok) p.resolve(msg.v);
      else p.reject(new Error(msg.e || 'worker error'));
      return;
    }
    if (msg.t === 'evt') {
      if (this._evtHandler) this._evtHandler(msg.k, msg.p);
      return;
    }
    if (msg.t === 'req') {
      if (!this._handler) {
        this._post({ t: 'res', id: msg.id, ok: false, e: 'no request handler' });
        return;
      }
      try {
        const r = await this._handler(msg.m, msg.p);
        this._post({ t: 'res', id: msg.id, ok: true, v: r ? r.value : undefined },
          r ? r.transfer : undefined);
      } catch (e) {
        this._post({ t: 'res', id: msg.id, ok: false, e: (e && e.message) || String(e) });
      }
    }
  }
}
