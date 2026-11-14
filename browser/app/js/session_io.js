// session_io.js — save a trained run, load it back, share it as a link.
//
// Two artifacts, packed into one session zip next to the web-ready SOG:
//
//   recon.json — cameras at feature scale, frame metadata, a downsampled
//     sparse cloud, the source file list: everything the app needs to
//     PRESENT a model (capture-path tour, frustums, orbit pivot) without
//     re-solving, and to refetch the images for a training resume.
//   state.bin  — the trainer's parameters and SH coefficients as float16
//     words (v2 — half the bytes; a resume re-converges the quantization
//     noise, and v1 float32 blobs still read): no opacity baking, no SOG
//     quantization.
//
// ?model=<url> accepts a session .zip, a .ply, or a .sog (decoded via the
// vendored splat-transform); &recon=<url> adds presentation state to bare
// model files. Bare exports carry baked opacities — fine for viewing, and
// re-exports pass them through unchanged.

import { zipStore, readZip } from './zip.js';
import { loadST } from './sog.js';

const STRIDE = 16;
const SH_C0 = 0.28209479177387814;

// ---------------------------------------------------------------------------
// save
// ---------------------------------------------------------------------------

/** Presentation + resume metadata for the current run. */
/** The trainer RECIPE: every option the session's trainer was created with
 *  except the per-run sizing (maxSplats / capMult, recorded as the run's cap),
 *  plus the session's refine cadence. Saved in the state header and the recon
 *  JSON so a resume, a continue-from-share or a view-from-state rebuilds the
 *  SAME trainer — mipComp, anisoReg, minScale, refineV2, growRate, the MCMC
 *  set … Before this (2026-09-03) every resume fell back to bare defaults,
 *  and a model trained without Mip compensation would have been un-baked and
 *  re-rendered with it. Absent recipe (old records) = the legacy defaults. */
export function trainRecipe(ses) {
  const t = { ...((ses.opts && ses.opts.trainer) || {}) };
  delete t.maxSplats; delete t.capMult;
  if (ses.opts && ses.opts.refineEvery != null) t.refineEvery = ses.opts.refineEvery;
  return t;
}

/** The splat budget training was CONFIGURED with (the session trainer opts'
 *  maxSplats), falling back to the live allocation ceiling, then the model
 *  size. trainer.cap alone lies low whenever seed x capMult under-allocates
 *  (the default for explicit sub-1M budgets): the continue card's "Original
 *  budget" and every ceiling that inherits it must see the configured limit,
 *  not the allocation. */
export function trainBudget(ses) {
  const t = (ses && ses.opts && ses.opts.trainer) || {};
  const tr = (ses && ses.trainer) || {};
  return Math.max(t.maxSplats || 0, tr.cap || 0, tr.n || 0);
}

/** Capture-world gravity in the DATA frame, from a capture.json leveling
 *  report (row-major rotation R): leveling rotates the world by R, so the
 *  capture world's +Y — gravity, ARCore sessions are gravity-aligned — sits
 *  at R·Y in the data frame. Null unless the report applied a rotation. */
export function levelingUpFromReport(leveling) {
  const r = leveling && leveling.applied === true && leveling.rotation;
  if (!Array.isArray(r) || r.length !== 9 || !r.every(Number.isFinite)) return null;
  const g = [r[1], r[4], r[7]];   // R·[0,1,0] is the rotation's middle column
  const m = Math.hypot(g[0], g[1], g[2]);
  return m > 1e-6 ? [g[0] / m, g[1] / m, g[2] / m] : null;
}

const validUp = (v) => Array.isArray(v) && v.length === 3 && v.every(Number.isFinite) &&
  Math.hypot(v[0], v[1], v[2]) > 1e-6;

/** The orbit's up axis for a known-camera run: capture-world gravity in the
 *  data frame. The cameras' own up axes CANNOT serve — phone captures are
 *  portrait-held, so sensor-up lies sideways to gravity and averaging it
 *  levels the horizon 90° off. Solver runs keep detectUp (their frame is
 *  arbitrary), so this helper is only called for known-camera runs. */
export function gravityUpForKnownRun({ persisted = null, leveling = null } = {}) {
  if (validUp(persisted)) {
    const m = Math.hypot(persisted[0], persisted[1], persisted[2]);
    return [persisted[0] / m, persisted[1] / m, persisted[2] / m];
  }
  return levelingUpFromReport(leveling) || [0, 1, 0];
}

/** A continuation's default input scale: the stored size is the COMBINED
 *  size (input × buffer), so the buffer is divided back out before the value
 *  passes through adaptiveTrainCap's multiplier again — else the buffer
 *  applies twice and the continuation trains at (e.g.) 1/16 instead of the
 *  run's 1/4. Unrecorded buffers (old records) replay at scale 1. */
export function continuedInputDim(prevDim, prevBuf) {
  if (!prevDim || !prevBuf || prevBuf === 1) return prevDim;
  return Math.round(prevDim / prevBuf);
}

/** A continuation's replayed working buffer: the recon's own trainScale
 *  (exact, current records); else the run record's saved settings when its
 *  resolution × buffer triple reproduces the stored size (older records —
 *  the check rejects settings that drifted after the run started); else the
 *  saved buffer alone (Auto resolutions have no triple to check); else 1. */
export function continuedBuf(trainScale, settings, prevDim) {
  if (Number.isFinite(trainScale) && trainScale > 0) return trainScale;
  const buf = settings ? +settings.buf : NaN;
  if (!Number.isFinite(buf) || buf <= 0) return 1;
  const res = settings ? +settings.res : NaN;
  if (Number.isFinite(res) && res > 0 && Number.isFinite(prevDim) && prevDim > 0) {
    // a saved step at or above the set's size commits as Full (-1), so a
    // positive res always ran unclamped: res × buf is the stored size
    if (Math.abs(res * buf - prevDim) > Math.max(2, prevDim * 0.02)) return 1;
  }
  return buf;
}

export function buildReconJson(S) {
  const ses = S.session;
  const recon = ses.recon;
  const r6 = (a) => Array.from(a, (v) => +v.toPrecision(7));
  const pts = recon.points || [];
  const cloud = { xyz: [], rgb: [] };
  // The display cloud is a capture-time snapshot (especially for AR depth
  // seeds). Keep that exact cloud in saved reconstructions instead of reading
  // potentially-mutated optimizer/reconstruction state after training.
  if (S.scene?.xyz?.length) {
    const count = Math.floor(S.scene.xyz.length / 3);
    const step = Math.max(1, Math.ceil(count / 4000));
    for (let i = 0; i < count; i += step) {
      cloud.xyz.push(+S.scene.xyz[i * 3].toPrecision(5), +S.scene.xyz[i * 3 + 1].toPrecision(5), +S.scene.xyz[i * 3 + 2].toPrecision(5));
      cloud.rgb.push(S.scene.rgb[i * 3], S.scene.rgb[i * 3 + 1], S.scene.rgb[i * 3 + 2]);
    }
  } else {
    const step = Math.max(1, Math.ceil(pts.length / 4000));
    for (let i = 0; i < pts.length; i += step) {
      cloud.xyz.push(+pts[i].X[0].toPrecision(5), +pts[i].X[1].toPrecision(5), +pts[i].X[2].toPrecision(5));
      const c = pts[i].rgb;
      cloud.rgb.push(Math.round(c[0] * 255), Math.round(c[1] * 255), Math.round(c[2] * 255));
    }
  }
  return {
    version: 1,
    app: 'splat.js',
    name: S.preset && S.preset.name || 'splat',
    recipe: trainRecipe(ses),
    // known-camera (ARCore/WebXR import) runs keep the capture world's y-up
    // frame; solver runs keep the SfM frame. Viewers use this to orient the
    // model instead of guessing from the coordinates.
    knownCameras: !!recon.known,
    trainScale: ses.opts?.frames?.trainScale ?? 1,
    // gravity in the data frame (see gravityUpForKnownRun): the orbit's up
    // axis when this run is viewed back. Solver runs omit it (detectUp).
    ...(recon.known ? { levelingUp: r6(gravityUpForKnownRun({
      persisted: S._localRun?.recon?.levelingUp,
      leveling: S.preset?.capture?.metadata?.leveling,
    })) } : {}),
    iter: ses.trainer.iter,
    splats: ses.trainer.n,
    // the CONFIGURED budget (records old enough to hold the allocation read
    // low here, exactly as before — new saves carry the real limit)
    maxSplats: trainBudget(ses),
    shK: ses.trainer.shK,
    sceneRadius: ses.model.radius,
    center: ses.model.center ? r6(ses.model.center) : null,
    k1: recon.k1, k2: recon.k2, fFeat: recon.fFeat ?? null,
    cams: recon.cams.map((c) => ({
      imgIdx: c.imgIdx, name: ses.frames[c.imgIdx].name,
      R: r6(c.R), t: r6(c.t), f: +c.f.toPrecision(7),
      ...(c.fy != null && c.fy !== c.f ? { fy: +c.fy.toPrecision(7) } : {}),
      cx: c.cx, cy: c.cy,
    })),
    frames: ses.frames.map((f) => ({ name: f.name, fw: f.fw, fh: f.fh, tw: f.tw, th: f.th,
      // the file's own pixel size (decodeFrames srcW/srcH): Full-size
      // continuations need true native, not the feature-capped fw/fh
      ...(Number.isFinite(f.srcW) ? { nw: f.srcW, nh: f.srcH } : {}) })),
    source: {
      preset: S.preset && !String(S.preset.id).startsWith('__') ? S.preset.id : null,
      names: (S.loadedFiles || []).map((f) => f.name),
      // absolute URLs where the images live (preset runs resolve against the
      // deployment's data root — own captures have no URL and stay names-only)
      urls: (S.loadedFiles || []).map((f) => f.url ? new URL(f.url, location.href).href : null),
    },
    stats: {
      minutes: S.minutes || 0,
      psnrTrain: S.psnrTrain ?? null,
      psnrHold: S.psnrHold ?? null,
      psnrTest: S.psnrTest ? { psnr: S.psnrTest.psnr, frames: S.psnrTest.frames.length } : null,
      // the training curve, decimated to ~400 points
      chart: decimate(ses.lossHistory || [], 800).map(([i, p]) => [i, +p.toFixed(3)]),
      holds: (S.holdHist || []).map(([i, p]) => [i, +p.toFixed(3)]),
      splats: decimate(S.splatHist || [], 800).map(([i, n]) => [i, Math.round(n)]),
      events: (S.chartEvents || []).slice(-300).map((e) => ({ ...e })),
    },
    cloud,
  };
}

const decimate = (arr, max) => {
  if (arr.length <= max) return arr;
  const step = arr.length / max;
  const out = [];
  for (let i = 0; i < arr.length; i += step) out.push(arr[Math.floor(i)]);
  return out;
};

const SHK_OF_DEGREE = { 0: 0, 1: 3, 2: 8, 3: 15 };

/** Resample decoded gaussians' spherical harmonics to another degree (0-3):
 *  lowering keeps the low bands, raising zero-pads them. The SH buffer is
 *  channel-major per splat ([R_0..R_{K-1}, G_0.., B_0..], coefficients in
 *  increasing band order — the shaders read sh[sb + ch*K + k]), so a lower
 *  degree is a per-channel prefix. Mutates and returns g. Used when
 *  continuing a run at a different SH degree than it trained with. */
export function convertSHDegree(g, degree) {
  const newK = SHK_OF_DEGREE[degree];
  if (newK == null) throw new Error(`SH degree must be 0-3, got ${degree}`);
  const oldK = g.shK ?? (g.sh ? null : 0);
  if (oldK == null || ![0, 3, 8, 15].includes(oldK)) {
    throw new Error(`cannot resample SH from unknown band count ${g.shK}`);
  }
  if (oldK === newK) return g;
  const n = g.n, keep = Math.min(oldK, newK);
  const out = new Float32Array(n * newK * 3);
  for (let i = 0; i < n; i++) {
    for (let c = 0; c < 3; c++) {
      const src = (i * 3 + c) * oldK, dst = (i * 3 + c) * newK;
      for (let k = 0; k < keep; k++) out[dst + k] = g.sh[src + k];
    }
  }
  g.sh = newK ? out : null;
  g.shK = newK;
  return g;
}

// float16 word codec for the v2 state payload — plain bit-twiddling, no
// DataView float16 (too new for the WebViews this ships in). Round to
// nearest, ties to even; NaN -> quiet NaN, overflow -> infinity.
const _f32u = new Float32Array(1);
const _f32b = new Uint32Array(_f32u.buffer);
export function floatToHalf(v) {
  _f32u[0] = v;
  const x = _f32b[0];
  const s = (x >> 16) & 0x8000;
  const be = (x >> 23) & 0xff;
  // inf, or quiet NaN for NaN — tested BEFORE the rebias, so a nonzero
  // mantissa on an overflowing (but finite) value still saturates to inf
  if (be === 255) return s | 0x7c00 | (x & 0x7fffff ? 0x200 : 0);
  const e = be - 112;   // rebias the exponent 127 -> 15
  const m = x & 0x7fffff;
  if (e <= 0) {
    if (e < -10) return s;   // underflows to signed zero
    const sub = (m | 0x800000) >> (1 - e);
    return s | ((sub + 0xfff + ((sub >> 13) & 1)) >> 13);
  }
  if (e >= 31) return s | 0x7c00;   // overflow -> infinity
  // ADD, not OR: a mantissa rounding up to 0x400 carries into the exponent,
  // and OR would swallow that carry whenever e is odd (e.g. 0.49999… -> 0.25)
  return s | ((e << 10) + (((m + 0xfff + ((m >> 13) & 1)) >> 13)));
}
export function halfToFloat(h) {
  const s = (h & 0x8000) << 16;
  const e = (h >> 10) & 0x1f;
  const m = h & 0x3ff;
  if (e === 0) {
    if (m === 0) { _f32b[0] = s; return _f32u[0]; }
    let mm = m, ee = -14;   // subnormal: renormalize
    while (!(mm & 0x400)) { mm <<= 1; ee--; }
    _f32b[0] = s | ((ee + 127) << 23) | ((mm & 0x3ff) << 13);
    return _f32u[0];
  }
  if (e === 31) { _f32b[0] = s | 0x7f800000 | (m << 13); return _f32u[0]; }
  _f32b[0] = s | ((e + 112) << 23) | (m << 13);
  return _f32u[0];
}
const floatsToHalfs = (src, dst) => { for (let i = 0; i < src.length; i++) dst[i] = floatToHalf(src[i]); };
const halfsToFloats = (src, dst) => { for (let i = 0; i < src.length; i++) dst[i] = halfToFloat(src[i]); };

/** The trainer's state as one binary blob: float16 params + SH (v2 — half
 *  the bytes of the v1 float32 layout, for values a resume re-converges
 *  within a few hundred iterations; the optimizer already cold-starts, so
 *  nothing else is lost). */
export async function packState(ses) {
  const { data, n, sh, shK, dc } = await ses.exportRawState();
  const head = new TextEncoder().encode(JSON.stringify({
    magic: 'splatjs-state', version: 2, n, shK, iter: ses.trainer.iter,
    // convention tags (absent in old blobs = the v1 defaults): dc names the
    // color-slot encoding, engine picks the trainer a resume must rebuild
    dc: dc || 'sigmoid', engine: ses.trainer.v2 ? 'v2' : 'v1',
    recipe: trainRecipe(ses),
  }));
  // explicit-length views: short export buffers throw here, not silently
  // (readGaussians hands over fresh float32 arrays, so these views align)
  const src = new Float32Array(data.buffer, data.byteOffset, n * STRIDE);
  const params = new Uint16Array(n * STRIDE);
  floatsToHalfs(src, params);
  const shCount = sh ? n * shK * 3 : 0;
  const shHalf = new Uint16Array(shCount);
  if (sh) floatsToHalfs(new Float32Array(sh.buffer, sh.byteOffset, shCount), shHalf);
  const out = new Uint8Array(4 + head.length + params.byteLength + shHalf.byteLength);
  new DataView(out.buffer).setUint32(0, head.length, true);
  out.set(head, 4);
  out.set(new Uint8Array(params.buffer), 4 + head.length);
  out.set(new Uint8Array(shHalf.buffer), 4 + head.length + params.byteLength);
  return out;
}

/** model.sog + recon.json + state.bin -> one resumable zip. */
export async function buildSessionZip(S, sogBlob) {
  const recon = buildReconJson(S);
  const state = await packState(S.session);
  return zipStore([
    { name: 'model.sog', data: new Uint8Array(await sogBlob.arrayBuffer()) },
    { name: 'recon.json', data: new TextEncoder().encode(JSON.stringify(recon)) },
    { name: 'state.bin', data: state },
  ]);
}

// ---------------------------------------------------------------------------
// load
// ---------------------------------------------------------------------------

/** STORED-entry zip reader (the writer next door emits nothing else). */
export function unzipStore(bytes) {
  const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const out = new Map();
  let o = 0;
  while (o + 30 <= bytes.length && dv.getUint32(o, true) === 0x04034b50) {
    const method = dv.getUint16(o + 8, true);
    const size = dv.getUint32(o + 18, true);
    const nameLen = dv.getUint16(o + 26, true);
    const extraLen = dv.getUint16(o + 28, true);
    const nameEnd = o + 30 + nameLen;
    const start = nameEnd + extraLen;
    const end = start + size;
    if (nameEnd > bytes.length || start > bytes.length || end > bytes.length) {
      throw new Error('truncated zip entry');
    }
    const name = new TextDecoder().decode(bytes.subarray(o + 30, nameEnd));
    if (method !== 0) throw new Error(`zip entry ${name} is compressed — not a splat.js session zip`);
    out.set(name, bytes.subarray(start, end));
    o = end;
  }
  return out;
}

/** unzipStore that also inflates deflated entries (OS-compressor zips) —
 *  async, via the platform DecompressionStream. Same Map shape; our own
 *  stored exports still take the fast sync path first. */
export async function unzipStoreAsync(bytes) {
  try {
    return unzipStore(bytes);
  } catch {
    return readZip(bytes);   // stored parse failed — maybe deflated (or corrupt: readZip says so)
  }
}

export function parseState(bytes) {
  if (bytes.length < 4) throw new Error('truncated state header');
  const headLen = new DataView(bytes.buffer, bytes.byteOffset).getUint32(0, true);
  if (headLen > bytes.length - 4) throw new Error('truncated state header');
  const head = JSON.parse(new TextDecoder().decode(bytes.subarray(4, 4 + headLen)));
  if (head.magic !== 'splatjs-state' || (head.version !== 1 && head.version !== 2)) {
    throw new Error('not a supported splat.js state blob');
  }
  if (!Number.isSafeInteger(head.n) || head.n < 1 ||
      ![0, 3, 8, 15].includes(head.shK) ||
      !Number.isSafeInteger(head.iter) || head.iter < 0) {
    throw new Error('invalid state metadata');
  }
  const o = 4 + headLen;
  const word = head.version === 2 ? 2 : 4;   // v2 stores float16 words
  const expected = head.n * (STRIDE + head.shK * 3) * word;
  if (!Number.isSafeInteger(expected) || bytes.length - o !== expected) {
    throw new Error('state data length does not match its header');
  }
  const params = new Float32Array(head.n * STRIDE);
  let sh = null;
  if (head.version === 2) {
    // the JSON header is an arbitrary length, so the payload is rarely
    // 2-aligned: one copy onto an aligned buffer (the constructor copies
    // even Buffer views), then typed views over it
    const body = new Uint8Array(bytes.subarray(o));
    halfsToFloats(new Uint16Array(body.buffer, 0, params.length), params);
    if (head.shK) {
      sh = new Float32Array(head.n * head.shK * 3);
      halfsToFloats(new Uint16Array(body.buffer, params.length * 2, sh.length), sh);
    }
  } else {
    new Uint8Array(params.buffer).set(bytes.subarray(o, o + params.byteLength));
    if (head.shK) {
      sh = new Float32Array(head.n * head.shK * 3);
      new Uint8Array(sh.buffer).set(bytes.subarray(o + params.byteLength, o + params.byteLength + sh.byteLength));
    }
  }
  return {
    gaussians: { data: params, n: head.n, sh, shK: head.shK, dc: head.dc === 'sh' ? 'sh' : 'sigmoid' },
    iter: head.iter,
    engine: head.engine === 'v2' ? 'v2' : 'v1',
    recipe: head.recipe && typeof head.recipe === 'object' ? head.recipe : null,
  };
}

function readPlyHeader(bytes) {
  // PLY headers can exceed a few KiB when tools add comments or many
  // properties. Scan a bounded prefix instead of assuming the terminator is in
  // the first 4096 bytes, but still reject files with an unreasonably large or
  // missing header before allocating typed-array views for the body.
  const limit = Math.min(bytes.length, 1024 * 1024);
  const headText = new TextDecoder().decode(bytes.subarray(0, limit));
  // LF or CRLF terminator — some Windows tools write CRLF headers
  const m = headText.match(/end_header\r?\n/);
  if (!m) throw new Error('not a PLY file');
  return {
    header: headText.slice(0, m.index).replace(/\r\n/g, '\n'),
    bodyAt: m.index + m[0].length,
  };
}

/** Standard 3DGS PLY -> the trainer's raw layout (exact inverse of the
 *  exporter's activations; opacities stay as stored — baked is fine to view
 *  and passes through re-exports unchanged). */
export function parsePlyGaussians(bytes) {
  const { header, bodyAt } = readPlyHeader(bytes);
  if (!header.startsWith('ply\n')) throw new Error('not a PLY file');
  if (!/^format binary_little_endian 1\.0$/m.test(header)) {
    throw new Error('PLY must use binary_little_endian 1.0');
  }
  let n = 0, vertex = false;
  const props = [];
  for (const l of header.split('\n')) {
    const me = l.match(/^element (\S+) (\d+)/);
    if (me) {
      vertex = me[1] === 'vertex';
      if (vertex) n = +me[2];
      continue;
    }
    const mp = l.match(/^property (\S+) (\S+)/);
    if (vertex && mp) {
      if (mp[1] !== 'float') throw new Error(`unsupported PLY vertex property type ${mp[1]}`);
      props.push(mp[2]);
    }
  }
  if (!Number.isSafeInteger(n) || n < 1 || new Set(props).size !== props.length) {
    throw new Error('invalid PLY vertex header');
  }
  const idx = Object.fromEntries(props.map((p, i) => [p, i]));
  const need = [
    'x', 'y', 'z', 'f_dc_0', 'f_dc_1', 'f_dc_2', 'opacity',
    'scale_0', 'scale_1', 'scale_2', 'rot_0', 'rot_1', 'rot_2', 'rot_3',
  ];
  for (const k of need) if (idx[k] == null) throw new Error(`PLY has no ${k} — not a 3DGS splat file`);
  const rest = props.filter((p) => p.startsWith('f_rest_'));
  const K = rest.length / 3;
  if (![0, 3, 8, 15].includes(K) ||
      rest.some((_, k) => idx[`f_rest_${k}`] == null)) {
    throw new Error('unsupported or incomplete PLY spherical harmonics');
  }
  const stride = props.length;
  const bodyBytes = n * stride * 4;
  if (!Number.isSafeInteger(bodyBytes) || bodyBytes > bytes.length - bodyAt) {
    throw new Error('truncated PLY vertex data');
  }
  // slice: the body must land 4-aligned for the Float32Array view (a PLY
  // header has arbitrary length, and zip entries sit at arbitrary offsets)
  const body = bytes.slice(bodyAt, bodyAt + bodyBytes);
  const src = new Float32Array(body.buffer, 0, n * stride);
  const data = new Float32Array(n * STRIDE);
  const sh = K ? new Float32Array(n * K * 3) : null;
  const logit = (v) => {
    const c = Math.min(1 - 1e-5, Math.max(1e-5, v));
    return Math.log(c / (1 - c));
  };
  for (let i = 0; i < n; i++) {
    const s = i * stride, d = i * STRIDE;
    data[d] = src[s + idx.x]; data[d + 1] = src[s + idx.y]; data[d + 2] = src[s + idx.z];
    data[d + 3] = src[s + idx.scale_0]; data[d + 4] = src[s + idx.scale_1]; data[d + 5] = src[s + idx.scale_2];
    data[d + 6] = src[s + idx.rot_0]; data[d + 7] = src[s + idx.rot_1];
    data[d + 8] = src[s + idx.rot_2]; data[d + 9] = src[s + idx.rot_3];
    // keep the STANDARD SH-DC convention (tagged below): converting to
    // sigmoid logits here destroyed out-of-range colors before the trainer
    // could see them (v2 exports lost ~1.1 dB on re-import). The trainer's
    // setup() bridges to whichever convention its engine needs.
    data[d + 10] = src[s + idx.f_dc_0];
    data[d + 11] = src[s + idx.f_dc_1];
    data[d + 12] = src[s + idx.f_dc_2];
    data[d + 13] = src[s + idx.opacity];
    for (let k = 0; k < 3 * K; k++) sh[i * 3 * K + k] = src[s + idx[`f_rest_${k}`]];
  }
  return { data, n, sh, shK: K, dc: 'sh' };
}

/** .sog bytes -> gaussians, via the vendored splat-transform (sog -> ply in
 *  memory -> parse). */
export async function sogToGaussians(bytes) {
  const st = await loadST();
  const rfs = new st.MemoryReadFileSystem();
  rfs.set('model.sog', bytes);
  const [source] = await st.readFile({ filename: 'model.sog', inputFormat: 'sog', fileSystem: rfs });
  const out = new st.MemoryFileSystem();
  const pool = st.createChunkDataPool();
  try {
    await st.writeSource({ filename: 'model.ply', outputFormat: 'ply', source, pool, options: {} }, out);
  } finally {
    source.close?.();
  }
  const ply = out.results.get('model.ply');
  if (!ply) throw new Error('SOG decode produced no PLY');
  return parsePlyGaussians(ply);
}

/** Fetch + identify + decode a ?model= target.
 *  Returns { gaussians, reconJson|null, state:{iter}|null }. */
export async function fetchModel(modelUrl, reconUrl) {
  const resp = await fetch(modelUrl);
  if (!resp.ok) throw new Error(`model fetch failed (${resp.status})`);
  return decodeModel(new Uint8Array(await resp.arrayBuffer()), reconUrl);
}

/** Same, from bytes already in hand (a dropped file). */
export async function decodeModel(bytes, reconUrl) {
  let gaussians = null, reconJson = null, state = null;
  if (bytes[0] === 0x50 && bytes[1] === 0x4b) {
    // PK: either our session zip — or a bare SOG bundle, which is a zip too
    let entries = null;
    try { entries = await unzipStoreAsync(bytes); } catch { /* not a readable zip */ }
    if (entries && entries.has('recon.json')) {
      reconJson = JSON.parse(new TextDecoder().decode(entries.get('recon.json')));
    }
    if (entries && entries.has('state.bin')) {
      const st = parseState(entries.get('state.bin'));
      gaussians = st.gaussians;
      state = { iter: st.iter };
    } else if (entries && entries.has('model.sog')) {
      gaussians = await sogToGaussians(entries.get('model.sog'));
    } else if (entries && entries.has('model.ply')) {
      gaussians = parsePlyGaussians(entries.get('model.ply'));
    } else {
      gaussians = await sogToGaussians(bytes);
    }
  } else if (headTextIs(bytes, 'ply')) {
    gaussians = parsePlyGaussians(bytes);
  } else {
    gaussians = await sogToGaussians(bytes);
  }
  if (!gaussians) throw new Error('nothing loadable in the model file');
  if (!reconJson && reconUrl) {
    const r = await fetch(reconUrl);
    if (r.ok) {
      // the upload API takes no bare .json — a shared recon travels as a
      // one-entry STORED zip, so accept both forms here
      const rb = new Uint8Array(await r.arrayBuffer());
      if (rb[0] === 0x50 && rb[1] === 0x4b) {
        const entries = await unzipStoreAsync(rb);
        const entry = entries.get('recon.json') || [...entries.values()][0];
        if (entry) reconJson = JSON.parse(new TextDecoder().decode(entry));
      } else {
        reconJson = JSON.parse(new TextDecoder().decode(rb));
      }
    }
  }
  return { gaussians, reconJson, state };
}

const headTextIs = (bytes, tag) =>
  new TextDecoder().decode(bytes.subarray(0, tag.length)) === tag;

// ---------------------------------------------------------------------------
// resume trimming
// ---------------------------------------------------------------------------

/** Trim a saved detailProgress to a resume point: the record's per-iteration
 *  histories (chart, holds, events, metrics) can run AHEAD of its state
 *  checkpoint — a Details persist lands checkpoint-free — so a continued
 *  run would plot new points over the already-plotted stretch. Keeps rows
 *  at or before `iter` (metrics key on [1], events on .iter); rows without
 *  a readable iteration are kept, never destroy what we can't place.
 *  Returns a FRESH object — the record keeps its full history for viewing. */
export function trimProgressToIter(progress, iter) {
  const src = progress && typeof progress === 'object' ? progress : {};
  if (!Number.isFinite(iter)) return { ...src };
  const rowBefore = (r) => !Array.isArray(r) || !Number.isFinite(r[0]) || r[0] <= iter;
  const evBefore = (e) => !e || typeof e !== 'object' || !Number.isFinite(e.iter) || e.iter <= iter;
  const metBefore = (m) => !Array.isArray(m) || !Number.isFinite(m[1]) || m[1] <= iter;
  return {
    ...src,
    ...(Array.isArray(src.chart) ? { chart: src.chart.filter(rowBefore) } : {}),
    ...(Array.isArray(src.holds) ? { holds: src.holds.filter(rowBefore) } : {}),
    ...(Array.isArray(src.splats) ? { splats: src.splats.filter(rowBefore) } : {}),
    ...(Array.isArray(src.events) ? { events: src.events.filter(evBefore) } : {}),
    ...(Array.isArray(src.metrics) ? { metrics: src.metrics.filter(metBefore) } : {}),
  };
}
