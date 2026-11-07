// quality.js — in-browser quality scenarios, driven by tests/quality/run.mjs.
//
// Deliberately consumes ONLY the public library surface (src/index.js) — the
// gates double as an integration test of the packaged API. Results are POSTed
// to /scratch/quality_<run>.json where the node runner asserts thresholds.
//
//   ?scene=synthetic-solve   SfM accuracy on the bundled synthetic set
//   ?scene=synthetic-train   short training run, PSNR floors
//   ?scene=truck-ate         truck-42 SfM, poses posted for ATE vs COLMAP GT
//   ?scene=camping-ate       camping-113 SfM, poses posted for ATE vs GT
//   ?scene=rig-ate           360 rig SfM on generated pano faces, ATE vs GT

import { createSession } from '../../src/index.js';
// rig-ate generates its views in-page (48 canvases beat 48 PNGs in the repo)
import { generatePanoRigRaw } from '../../src/synthetic.js';
import { processSource } from '../../src/io/frames.js';
import { FACE_ROTS } from '../../src/io/pano.js';
import { jacobiEigen } from '../../src/sfm/geometry.js';

const q = new URLSearchParams(location.search);
const scene = q.get('scene') || 'synthetic-solve';
const runId = q.get('run') || 'local';
const out = document.getElementById('out');
const lines = [];
const say = (m) => { lines.push(m); out.textContent = lines.slice(-40).join('\n'); };

async function fetchSet(dir, pattern, start, end, stride = 1, max = 1000) {
  const files = [];
  for (let i = start; i <= end && files.length < max; i += stride) {
    const name = pattern.replace(/\{i:(\d+)\}/, (_, w) => String(i).padStart(+w, '0'));
    const resp = await fetch(`../../data/${dir}/${name}`);
    if (!resp.ok) throw new Error(`dataset missing: data/${dir}/${name}`);
    files.push({ source: await resp.blob(), name });
  }
  return files;
}

async function post(result) {
  result.scene = scene;
  result.gpu = !!navigator.gpu;
  await fetch(`/scratch/quality_${runId}.json`, { method: 'POST', body: JSON.stringify(result) });
  say('\nresult posted: ' + JSON.stringify(result, null, 1));
}

/** ATE after the best global similarity fit (Horn), as % of trajectory span. */
function hornAtePct(rec, gt) {
  const N = rec.length;
  const cen = (a) => [0, 1, 2].map((k) => a.reduce((x, p) => x + p[k], 0) / a.length);
  const cr = cen(rec), cg = cen(gt);
  const rc = rec.map((p) => [p[0] - cr[0], p[1] - cr[1], p[2] - cr[2]]);
  const gc = gt.map((p) => [p[0] - cg[0], p[1] - cg[1], p[2] - cg[2]]);
  const rms = (a) => Math.sqrt(a.reduce((x, p) => x + p[0] ** 2 + p[1] ** 2 + p[2] ** 2, 0) / a.length);
  const sc = rms(gc) / rms(rc);
  let Sxx = 0, Sxy = 0, Sxz = 0, Syx = 0, Syy = 0, Syz = 0, Szx = 0, Szy = 0, Szz = 0;
  for (let i = 0; i < N; i++) {
    const a = rc[i], b = gc[i];
    Sxx += a[0] * b[0]; Sxy += a[0] * b[1]; Sxz += a[0] * b[2];
    Syx += a[1] * b[0]; Syy += a[1] * b[1]; Syz += a[1] * b[2];
    Szx += a[2] * b[0]; Szy += a[2] * b[1]; Szz += a[2] * b[2];
  }
  const Nm = [
    Sxx + Syy + Szz, Syz - Szy, Szx - Sxz, Sxy - Syx,
    Syz - Szy, Sxx - Syy - Szz, Sxy + Syx, Szx + Sxz,
    Szx - Sxz, Sxy + Syx, -Sxx + Syy - Szz, Syz + Szy,
    Sxy - Syx, Szx + Sxz, Syz + Szy, -Sxx - Syy + Szz,
  ];
  const { vecs } = jacobiEigen(Nm, 4);
  const [w, x, y, z] = [vecs[0], vecs[4], vecs[8], vecs[12]];
  const Ra = [
    1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
    2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
    2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
  ];
  let se = 0;
  for (let i = 0; i < N; i++) {
    const a = rc[i];
    const p = [
      sc * (Ra[0] * a[0] + Ra[1] * a[1] + Ra[2] * a[2]),
      sc * (Ra[3] * a[0] + Ra[4] * a[1] + Ra[5] * a[2]),
      sc * (Ra[6] * a[0] + Ra[7] * a[1] + Ra[8] * a[2]),
    ];
    se += (p[0] - gc[i][0]) ** 2 + (p[1] - gc[i][1]) ** 2 + (p[2] - gc[i][2]) ** 2;
  }
  const span = Math.max(...gc.map((p) => Math.hypot(...p))) * 2;
  return Math.sqrt(se / N) / span * 100;
}

const SETS = {
  synthetic: { dir: 'synthetic', pattern: 'synthetic_{i:2}.png', start: 0, end: 11 },
  truck: { dir: 'truck', pattern: '{i:6}.jpg', start: 1, end: 42 },
  camping: { dir: 'camping', pattern: 'frame_{i:5}.jpg', start: 1, end: 113 },
  // the README's Truck row is the FULL set (250/250, 0.00 % ATE); truck-ate above is the 42-image gate
  truckfull: { dir: 'truck', pattern: '{i:6}.jpg', start: 1, end: 251 },
  // a person orbit from video (Tom, 65 picked frames of a 1080p iPhone clip) against a COLMAP
  // reference: the video chain solved with the app's video gates (absolute pair gates 100 / 15,
  // what app.js VIDEO_GATES passes) and the default final trials — the merge gate for the
  // 2026-09-17 solver work (RANSAC underflow, clean passes, trials)
  tom: { dir: 'tom', pattern: 'frame_{i:5}.jpg', start: 1, end: 65, sfm: { pairMinInliers: 100, pairMinInliersAdj: 15 } },
};

async function solveSet(setName, session) {
  const s = SETS[setName];
  say(`fetching ${setName} ...`);
  const files = await fetchSet(s.dir, s.pattern, s.start, s.end);
  await session.load(files);
  say(`solving ${files.length} frames ...`);
  const t0 = performance.now();
  const recon = await session.solve();
  const solveSec = (performance.now() - t0) / 1000;
  say(`solved: ${recon.cams.length}/${files.length} cams, rms ${recon.rmsBA && recon.rmsBA.toFixed(2)}px, ${solveSec.toFixed(0)}s`);
  return { recon, solveSec, total: files.length };
}

try {
  const session = createSession({ maxIters: 3000, evalHoldEvery: 1500 });
  session.on('log', say);

  if (scene === 'synthetic-solve' || scene === 'synthetic-train') {
    const { recon, solveSec, total } = await solveSet('synthetic', session);
    const gt = await (await fetch('../../data/synthetic/cameras.json')).json();
    const fGT = gt.cameras[0].fx;
    const result = {
      cams: recon.cams.length, total,
      fEst: recon.cams[0].f, fGT, fErr: Math.abs(recon.cams[0].f - fGT) / fGT,
      rmsBA: recon.rmsBA, solveSec,
    };
    if (scene === 'synthetic-train') {
      await session.seed();
      say('training 3000 iterations ...');
      const t1 = performance.now();
      await new Promise((res) => { session.on('event', (e) => { if (e.kind === 'train-complete') res(); }); session.start(); });
      result.trainSec = (performance.now() - t1) / 1000;
      const m = await session.metrics();
      result.iter = m.iter;
      result.splats = m.splats;
      result.psnrHold = m.psnrHold;
      const hist = session.lossHistory;
      result.psnrTrain = hist.length ? hist[hist.length - 1][1] : null;
      // interactive view render at a LARGER-than-training canvas (the black-
      // render class of bug: view buffers sized only for the training res)
      const cv = document.createElement('canvas');
      cv.width = 1200; cv.height = 800;
      session.view.attach(cv);
      const meta = session.trainer.camMeta[0];
      const s = Math.min(1200 / meta.w, 800 / meta.h);
      session.view.setCamera({
        R: meta.R, t: meta.t, f: meta.f * s,
        cx: 600, cy: 400, w: 1200, h: 800,
      });
      session.view.renderNow();
      const c2 = document.createElement('canvas');
      c2.width = 64; c2.height = 64;
      const x2 = c2.getContext('2d');
      x2.drawImage(cv, 0, 0, 64, 64);
      const px = x2.getImageData(0, 0, 64, 64).data;
      let sum = 0;
      for (let i = 0; i < px.length; i += 4) sum += px[i] + px[i + 1] + px[i + 2];
      result.viewPixelSum = sum;
    }
    await post(result);
  } else if (scene === 'rig-ate') {
    // 8 panos x 6 cube faces of the synthetic room, solved as rigs with the
    // known face focal; ATE (Horn similarity fit) vs the exact GT centres
    say('generating 8 rigs x 6 faces ...');
    const raw = generatePanoRigRaw(8, 512, 100);
    session.useFrames(raw.map((v) => processSource(v.canvas, v.w, v.h, v.name, 512)));
    say('solving 48 faces ...');
    const t0 = performance.now();
    const recon = await session.solve({
      rigs: raw.map((v) => ({ id: v.rig, R: FACE_ROTS[v.face] })),
      focalPx: raw[0].f,
    });
    const solveSec = (performance.now() - t0) / 1000;
    const rec = [], gt = [];
    for (const c of recon.cams) {
      if (c.imgIdx % 6 !== 0) continue;
      const { R, t } = c;
      rec.push([
        -(R[0] * t[0] + R[3] * t[1] + R[6] * t[2]),
        -(R[1] * t[0] + R[4] * t[1] + R[7] * t[2]),
        -(R[2] * t[0] + R[5] * t[1] + R[8] * t[2]),
      ]);
      gt.push(raw[c.imgIdx].eye);
    }
    await post({ cams: recon.cams.length, total: raw.length, rmsBA: recon.rmsBA, solveSec, atePct: hornAtePct(rec, gt) });
  } else if (scene === 'truck-ate' || scene === 'camping-ate' || scene === 'truckfull-ate' || scene === 'tom-ate') {
    const setName = scene.split('-')[0];
    const ses = SETS[setName].sfm ? createSession({ maxIters: 3000, evalHoldEvery: 1500, sfm: SETS[setName].sfm }) : session;
    if (ses !== session) ses.on('log', say);
    const { recon, solveSec, total } = await solveSet(setName, ses);
    await post({
      cams: recon.cams.length, total, rmsBA: recon.rmsBA, solveSec,
      poses: recon.cams.map((c) => ({
        name: ses.frames[c.imgIdx].name, R: Array.from(c.R), t: Array.from(c.t), f: c.f,
      })),
    });
  } else {
    throw new Error('unknown scene: ' + scene);
  }
  say('DONE');
} catch (e) {
  say('ERROR: ' + (e.stack || e.message));
  await post({ error: String(e.message || e) });
}
