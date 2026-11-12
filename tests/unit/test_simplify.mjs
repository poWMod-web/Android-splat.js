import assert from 'node:assert/strict';
import { simplifyGaussians } from '../../src/gs/simplify.js';

// Deterministic fixtures: tight spatial clusters so merges have sane
// candidates, unit-ish quats, mid-range colors, controllable opacity.
function makeRng(seed) {
  let a = seed >>> 0 || 1;
  return () => {
    a |= 0; a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const logit = (p) => Math.log(p / (1 - p));

function makeModel(n, { shK = 0, dc = 'sigmoid', clusters = 4, spread = 0.05, seed = 7, op = null } = {}) {
  const rng = makeRng(seed);
  const data = new Float32Array(n * 16);
  const sh = shK ? new Float32Array(n * shK * 3) : null;
  const centers = Array.from({ length: clusters }, () => [(rng() - 0.5) * 4, (rng() - 0.5) * 4, (rng() - 0.5) * 4]);
  for (let i = 0; i < n; i++) {
    const b = i * 16, c = centers[i % clusters];
    data[b] = c[0] + (rng() - 0.5) * spread;
    data[b + 1] = c[1] + (rng() - 0.5) * spread;
    data[b + 2] = c[2] + (rng() - 0.5) * spread;
    data[b + 3] = Math.log(0.01 + rng() * 0.03);
    data[b + 4] = Math.log(0.01 + rng() * 0.03);
    data[b + 5] = Math.log(0.01 + rng() * 0.03);
    // random unit quat from the RNG stream (not uniform, but normalized)
    const q = [rng() - 0.5, rng() - 0.5, rng() - 0.5, rng() - 0.5];
    const qn = Math.hypot(...q) || 1;
    data[b + 6] = q[0] / qn; data[b + 7] = q[1] / qn;
    data[b + 8] = q[2] / qn; data[b + 9] = q[3] / qn;
    data[b + 10] = dc === 'sh' ? (rng() - 0.5) * 2 : logit(0.2 + rng() * 0.6);
    data[b + 11] = dc === 'sh' ? (rng() - 0.5) * 2 : logit(0.2 + rng() * 0.6);
    data[b + 12] = dc === 'sh' ? (rng() - 0.5) * 2 : logit(0.2 + rng() * 0.6);
    data[b + 13] = logit(op ? op(i) : 0.5 + rng() * 0.4);
    if (sh) for (let k = 0; k < shK * 3; k++) sh[i * shK * 3 + k] = (rng() - 0.5) * 0.4;
  }
  return { data, n, sh, shK, dc };
}

const TAU_POW_15 = Math.pow(2 * Math.PI, 1.5);
const sig = (v) => 1 / (1 + Math.exp(-v));
function massOf(data, i) {
  const b = i * 16;
  return TAU_POW_15 * sig(data[b + 13]) *
    Math.exp(data[b + 3]) * Math.exp(data[b + 4]) * Math.exp(data[b + 5]);
}

// ---- reaches the target fraction -------------------------------------------
{
  const g = makeModel(64, { seed: 11 });
  const out = await simplifyGaussians(g, { ratio: 0.5, mcSamples: 1, seed: 3 });
  assert.equal(out.n, 32, `64 splats at ratio 0.5 land on 32, got ${out.n}`);
  assert.equal(out.data.length, 32 * 16);
  assert.equal(out.sh, null);
  assert.equal(out.dc, 'sigmoid');
}

// ---- opacity prune: the low tail goes, survivors pass through bit-exact ----
{
  const g = makeModel(30, {
    shK: 3, seed: 5,
    op: (i) => (i < 10 ? 1e-4 : 0.9),   // first ten are ghosts
  });
  const out = await simplifyGaussians(g, { ratio: 0.9, mcSamples: 1, seed: 3 });
  assert.equal(out.n, 20, `the prune keeps the 20 solid splats, got ${out.n}`);
  assert.equal(out.sh.length, 20 * 9);
  for (let r = 0; r < 20; r++) {
    assert.deepEqual(Array.from(out.data.subarray(r * 16, r * 16 + 16)),
      Array.from(g.data.subarray((r + 10) * 16, (r + 10) * 16 + 16)),
      `survivor row ${r} is bit-exact`);
    assert.deepEqual(Array.from(out.sh.subarray(r * 9, r * 9 + 9)),
      Array.from(g.sh.subarray((r + 10) * 9, (r + 10) * 9 + 9)),
      `survivor SH block ${r} is bit-exact`);
  }
}

// ---- deterministic: same seed, byte-identical output ------------------------
{
  const opts = { ratio: 0.5, mcSamples: 2, seed: 9 };
  const a = await simplifyGaussians(makeModel(48, { shK: 3, seed: 21 }), opts);
  const b = await simplifyGaussians(makeModel(48, { shK: 3, seed: 21 }), opts);
  assert.equal(a.n, b.n);
  assert.deepEqual(Array.from(a.data), Array.from(b.data), 'rows repeat exactly');
  assert.deepEqual(Array.from(a.sh), Array.from(b.sh), 'SH repeats exactly');
}

// ---- unmerged rows pass through bit-exactly --------------------------------
{
  const g = makeModel(40, { seed: 13 });
  const out = await simplifyGaussians(g, { ratio: 0.95, mcSamples: 1, seed: 3 });
  assert.equal(out.n, 38, `40 at 0.95 lands on 38, got ${out.n}`);
  const inputs = new Set();
  for (let i = 0; i < 40; i++) inputs.add(Array.from(g.data.subarray(i * 16, i * 16 + 16)).join(','));
  let exact = 0;
  for (let r = 0; r < out.n; r++) {
    if (inputs.has(Array.from(out.data.subarray(r * 16, r * 16 + 16)).join(','))) exact++;
  }
  assert.equal(exact, 36, `36 untouched rows survive bit-exact, got ${exact}`);
}

// ---- merged splats stay sane ------------------------------------------------
{
  const g = makeModel(64, { shK: 8, dc: 'sh', seed: 17 });
  const out = await simplifyGaussians(g, { ratio: 0.5, mcSamples: 1, seed: 3 });
  assert.equal(out.n, 32);
  assert.equal(out.dc, 'sh');
  assert.equal(out.sh.length, 32 * 24);
  for (let r = 0; r < out.n; r++) {
    const row = out.data.subarray(r * 16, r * 16 + 16);
    for (const v of row) assert.ok(Number.isFinite(v), `row ${r} is finite`);
    const q = Math.hypot(row[6], row[7], row[8], row[9]);
    assert.ok(Math.abs(q - 1) < 1e-5, `row ${r} quat is unit (${q})`);
    assert.ok(sig(row[13]) <= 0.99 + 1e-6, `row ${r} opacity stays capped`);
    assert.ok(row[3] > -25 && row[3] < 25, `row ${r} log-scale stays bounded`);
  }
  for (const v of out.sh) assert.ok(Number.isFinite(v), 'SH stays finite');
}

// ---- identical twins: shape/color/rest preserved, mass mostly kept ---------
{
  const one = makeModel(1, { shK: 3, seed: 23 });
  const data = new Float32Array(32);
  data.set(one.data, 0); data.set(one.data, 16);
  const sh = new Float32Array(18);
  sh.set(one.sh, 0); sh.set(one.sh, 9);
  const out = await simplifyGaussians({ data, n: 2, sh, shK: 3, dc: 'sigmoid' },
    { ratio: 0.4, mcSamples: 2, seed: 3 });
  assert.equal(out.n, 1, 'twins collapse to one');
  const m = out.data;
  assert.ok(Math.abs(m[0] - data[0]) < 1e-6, 'twin mean is preserved');
  assert.ok(Math.abs(sig(m[10]) - sig(data[10])) < 1e-5, 'twin color is preserved');
  assert.deepEqual(Array.from(out.sh), Array.from(one.sh), 'twin SH rest is preserved');
  const kept = massOf(out.data, 0) / (massOf(data, 0) + massOf(data, 1));
  assert.ok(kept > 0.5 && kept <= 1.01, `twin mass mostly kept (${kept.toFixed(3)})`);
}

// ---- color steers the matching: no red+blue mergers ------------------------
{
  const rng = makeRng(29);
  const n = 24, data = new Float32Array(n * 16);
  for (let i = 0; i < n; i++) {
    const b = i * 16, red = i % 2 === 0;
    // interleaved positions: every splat's nearest neighbors mix colors
    data[b] = (rng() - 0.5) * 0.1; data[b + 1] = (rng() - 0.5) * 0.1; data[b + 2] = (rng() - 0.5) * 0.1;
    data[b + 3] = data[b + 4] = data[b + 5] = Math.log(0.02);
    data[b + 6] = 1; data[b + 7] = data[b + 8] = data[b + 9] = 0;
    data[b + 10] = logit(red ? 0.95 : 0.05);
    data[b + 11] = logit(0.05);
    data[b + 12] = logit(red ? 0.05 : 0.95);
    data[b + 13] = logit(0.8);
  }
  const out = await simplifyGaussians({ data, n, sh: null, shK: 0, dc: 'sigmoid' },
    { ratio: 0.5, mcSamples: 1, seed: 3 });
  assert.equal(out.n, 12);
  for (let r = 0; r < out.n; r++) {
    const b = r * 16;
    const purity = Math.abs(sig(out.data[b + 10]) - sig(out.data[b + 12]));
    assert.ok(purity > 0.8, `merged row ${r} keeps its color (purity ${purity.toFixed(2)})`);
  }
}

// ---- bad ratios throw -------------------------------------------------------
for (const ratio of [0, 1, -0.5, 2, NaN]) {
  await assert.rejects(
    simplifyGaussians(makeModel(8, { seed: 1 }), { ratio, seed: 3 }),
    /ratio must be in/,
    `ratio ${ratio} is rejected`);
}

// ---- tiny models survive ----------------------------------------------------
{
  const one = makeModel(1, { seed: 31 });
  const out = await simplifyGaussians(one, { ratio: 0.5, seed: 3 });
  assert.equal(out.n, 1);
  assert.deepEqual(Array.from(out.data), Array.from(one.data));
  const empty = await simplifyGaussians(
    { data: new Float32Array(0), n: 0, sh: null, shK: 0, dc: 'sigmoid' }, { ratio: 0.5 });
  assert.equal(empty.n, 0);
}

// ---- status and progress callbacks fire -------------------------------------
{
  const statuses = [], stages = [];
  const out = await simplifyGaussians(makeModel(32, { seed: 37 }), {
    ratio: 0.5, mcSamples: 1, seed: 3,
    onStatus: (m) => statuses.push(m),
    onProgress: (p) => stages.push(p.stage),
  });
  assert.equal(out.n, 16);
  assert.ok(statuses.length >= 3, 'status narrates the run');
  assert.deepEqual(stages, ['start', 'prune', 'pass-start', 'pass', 'done']);
  assert.ok(statuses.some((m) => /final 16 splats/.test(m)), 'status reports the count');
}

console.log('NanoGS-method splat simplification: passed');
