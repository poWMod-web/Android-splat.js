import assert from 'node:assert/strict';
import { planSoftPrune, chunkPlan, COMPACT_SRC, COMPACT_TEMP_BYTES, CULLED } from '../../src/gs/softprune.js';

// ---- basic cull: the bottom fraction by mass goes --------------------------
{
  const n = 20000;
  const masses = new Float32Array(n);
  for (let i = 0; i < n; i++) masses[i] = i;
  const p = planSoftPrune(masses, n, 0.2);
  assert.equal(p.kept, 16000);
  assert.equal(p.culled, 4000);
  assert.equal(p.threshold, 4000);
  for (let i = 0; i < 4000; i++) assert.equal(p.rank[i], CULLED, `row ${i} culled`);
  for (let i = 4000; i < n; i++) assert.equal(p.rank[i], i - 4000, `row ${i} ranks ${i - 4000}`);
}

// ---- ties fill in ascending row order (deterministic) ----------------------
{
  const n = 5000;
  const p = planSoftPrune(new Float32Array(n).fill(5), n, 0.5, { minCull: 1, floor: 1 });
  assert.equal(p.kept, 2500);
  for (let i = 0; i < 2500; i++) assert.equal(p.rank[i], i);
  for (let i = 2500; i < n; i++) assert.equal(p.rank[i], CULLED);
}

// ---- fail-safe refusals ----------------------------------------------------
{
  const n = 20000;
  assert.equal(planSoftPrune(new Float32Array(n), n, 0.2), null,
    'all-zero mass (stats never accumulated) refuses — the cull would be arbitrary');
  const nan = new Float32Array(n).fill(1); nan[7] = NaN;
  assert.equal(planSoftPrune(nan, n, 0.2), null, 'non-finite mass refuses');
  const ones = new Float32Array(n).fill(1);
  assert.equal(planSoftPrune(ones, 2000, 0.1), null, 'a 200-row cull is not worth the compaction');
  assert.equal(planSoftPrune(ones, 4000, 0.5), null, 'models under the floor never prune');
  assert.throws(() => planSoftPrune(ones, n, 0), /ratio must be in/);
  assert.throws(() => planSoftPrune(ones, n, 1), /ratio must be in/);
  assert.throws(() => planSoftPrune(new Float32Array(10), n, 0.2), /holds 10 rows/);
}

// ---- chunk plans cover every survivor exactly once -------------------------
{
  const n = 10000;
  const masses = new Float32Array(n);
  for (let i = 0; i < n; i++) masses[i] = (i * 7919) % 1000;   // scattered culls
  const p = planSoftPrune(masses, n, 0.3, { minCull: 1, floor: 1 });
  assert.equal(p.kept, 7000, `scattered masses keep 7000, got ${p && p.kept}`);
  for (const stride of [16, 45]) {
    const chunks = chunkPlan(p.rank, n, stride, 65536);
    assert.ok(chunks.length > 1, `stride ${stride} splits into chunks`);
    let dst = 0, src = 0, covered = 0;
    for (const c of chunks) {
      assert.equal(c.dstBase, dst, 'destinations land contiguously');
      assert.ok(c.srcBase >= src, 'sources walk forward without overlap');
      assert.ok(c.count * stride * 4 <= 65536, 'every chunk fits the temp');
      // the chunk's survivors really are its ranked rows
      let expect = 0;
      for (let i = c.srcBase; i < c.srcBase + c.count; i++) if (p.rank[i] !== CULLED) expect++;
      assert.equal(c.survivors, expect);
      dst += c.survivors; src = c.srcBase + c.count; covered += c.survivors;
    }
    assert.equal(covered, p.kept, `stride ${stride} covers every survivor`);
    assert.equal(dst, p.kept);
  }
}

// ---- the kernel scatters by rank, skipping the culled ----------------------
assert.match(COMPACT_SRC, /0xFFFFFFFFu/, 'culled rows never scatter');
assert.match(COMPACT_SRC, /tmp\[di \+ k\] = src\[si \+ k\]/, 'a ranked scatter into temp');
assert.match(COMPACT_SRC, /workgroup_size\(256\)/, 'matches the gather dispatch shape');
assert.equal(COMPACT_TEMP_BYTES, 16 * 1024 * 1024);

console.log('soft-prune planning (rank + chunk): passed');
