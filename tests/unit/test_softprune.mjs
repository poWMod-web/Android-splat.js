import assert from 'node:assert/strict';
import { planSoftPrune, planDeadCull, chunkPlan, COMPACT_SRC, COMPACT_TEMP_BYTES, CULLED } from '../../src/gs/softprune.js';

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
  for (let i = 4001; i < n; i++) assert.equal(p.rank[i], i - 4001, `row ${i} ranks ${i - 4001}`);
  assert.equal(p.rank[4000], 15999, 'the threshold tie ranks after the strict survivors');
  assert.equal(p.floorBound, null, 'no floor binds on a clean cull');
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

// ---- the growth ratchet: below the cap a prune never shrinks the curve ----
{
  const n = 20000;
  const masses = new Float32Array(n);
  for (let i = 0; i < n; i++) masses[i] = i;
  // 20k now, 19k after the last prune: a 20% cull (→16k) is held at 19k
  const p = planSoftPrune(masses, n, 0.2, { last: 19000, cap: 1000000, minCull: 1 });
  assert.equal(p.kept, 19000, 'the ratchet holds the last post-prune count');
  assert.equal(p.culled, 1000);
  assert.equal(p.floorBound, 'ratchet');
  // at the cap the ratchet releases: recycling is the point up there
  const q = planSoftPrune(masses, n, 0.2, { last: 19000, cap: 20000 });
  assert.equal(q.kept, 16000, 'at the cap the full cull goes through');
  assert.equal(q.floorBound, null);
  // regrowth past the floor prunes normally (the cull eats the regrowth)
  const overshoot = new Float32Array(22000);
  for (let i = 0; i < 22000; i++) overshoot[i] = i;
  const r = planSoftPrune(overshoot, 22000, 0.2, { last: 16000, cap: 1000000 });
  assert.equal(r.kept, 17600, 'a 20% cull of 22k lands on 17600, above the 16k floor');
  assert.equal(r.floorBound, null);
}

// ---- structural tie-break: big-and-opaque outranks tiny-and-faint ---------
{
  const n = 20000;
  const masses = new Float32Array(n).fill(7);   // every row ties
  const logitOp = new Float32Array(n).fill(-4); // faint …
  const meanLogS = new Float32Array(n).fill(-5); // … and tiny …
  logitOp[15000] = 4; meanLogS[15000] = 2;       // … except one load-bearing giant
  const p = planSoftPrune(masses, n, 0.5, { minCull: 1, floor: 1, logitOp, meanLogS });
  assert.equal(p.kept, 10000);
  assert.equal(p.rank[15000], 0, 'the giant survives first among equals');
  // without the structure arrays the old ascending fill still rules
  const q = planSoftPrune(masses, n, 0.5, { minCull: 1, floor: 1 });
  for (let i = 0; i < 10000; i++) assert.equal(q.rank[i], i);
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

// ---- hysteresis: candidates vote, only consecutive offenders die ------------
{
  const n = 20000;
  const masses = new Float32Array(n);
  for (let i = 0; i < n; i++) masses[i] = i;
  const votes = new Uint16Array(n);
  const w1 = planSoftPrune(masses, n, 0.2, { votes });
  assert.equal(w1.culled, 0, 'one bad window votes but culls nothing');
  assert.equal(w1.kept, n);
  assert.equal(w1.rank, null, 'no cull means no compaction rank');
  assert.equal(w1.candidates, 4000, 'the bottom fraction still counts as candidates');
  assert.ok(votes.slice(0, 4000).every((v) => v === 1), 'candidates vote once');
  assert.ok(votes.slice(4000).every((v) => v === 0), 'the kept reset');
  const w2 = planSoftPrune(masses, n, 0.2, { votes });
  assert.equal(w2.culled, 4000, 'a second bad window confirms the cull');
  assert.equal(w2.kept, 16000);
  for (let i = 0; i < 4000; i++) assert.equal(w2.rank[i], CULLED, `row ${i} culled`);
  for (let i = 4000; i < n; i++) assert.equal(w2.rank[i], i - 4000, `row ${i} ranks ${i - 4000}`);
  // the window turns: yesterday's candidates recover, today's vote fresh
  const flipped = new Float32Array(n);
  for (let i = 0; i < n; i++) flipped[i] = n - 1 - i;
  const w3 = planSoftPrune(flipped, n, 0.2, { votes });
  assert.equal(w3.culled, 0, 'recovered rows are spared — votes are consecutive, not lifetime');
  assert.equal(votes[0], 0, 'a good window clears the count');
  assert.equal(votes[n - 1], 1, 'the new bottom starts at one');
}
// ---- deeper confirmation asks more windows ----------------------------------
{
  const n = 5000;
  const votes = new Uint16Array(n);
  const masses = new Float32Array(n);
  for (let i = 0; i < n; i++) masses[i] = i;
  assert.equal(planSoftPrune(masses, n, 0.5, { votes, votesNeeded: 3, minCull: 1, floor: 1 }).culled, 0);
  assert.equal(planSoftPrune(masses, n, 0.5, { votes, votesNeeded: 3, minCull: 1, floor: 1 }).culled, 0);
  assert.equal(planSoftPrune(masses, n, 0.5, { votes, votesNeeded: 3, minCull: 1, floor: 1 }).culled, 2500,
    'three strikes with votesNeeded 3');
}
// ---- the ratchet can be bypassed (pure milestone culls) ---------------------
{
  const n = 20000;
  const masses = new Float32Array(n);
  for (let i = 0; i < n; i++) masses[i] = i;
  const p = planSoftPrune(masses, n, 0.2, { last: 19000, cap: 1000000, minCull: 1, skipRatchet: true });
  assert.equal(p.kept, 16000, 'a bypassed ratchet holds nothing back');
  assert.equal(p.culled, 4000);
  assert.equal(p.floorBound, null);
  assert.equal(p.candidates, 4000, 'candidates equal the cull without votes');
  const short = new Uint16Array(10);
  assert.throws(() => planSoftPrune(masses, n, 0.2, { votes: short }), /holds 10 rows/,
    'a short vote array refuses instead of misvoting');
}

// ---- dead cull: zeros die on the second consecutive window ----------------
{
  const n = 3000;
  const votes = new Uint16Array(n);
  const masses = new Float32Array(n);
  for (let i = 0; i < n; i++) masses[i] = i < 2000 ? 0 : i;
  const first = planDeadCull(masses, n, { votes, minCull: 1, floor: 1 });
  assert.equal(first.culled, 0, 'one window only votes');
  assert.equal(first.candidates, 2000, 'every zero-mass row is a candidate');
  assert.equal(first.rank, null, 'no plan ranks a no-cull window');
  assert.equal(votes[0], 1, 'a zero row votes up');
  assert.equal(votes[2999], 0, 'a contributing row stays clean');
  const second = planDeadCull(masses, n, { votes, minCull: 1, floor: 1 });
  assert.equal(second.culled, 2000, 'two consecutive zero windows kill');
  assert.equal(second.kept, 1000);
  assert.equal(second.rank[0], CULLED, 'a confirmed zero never scatters');
  assert.equal(second.rank[2000], 0, 'survivors compact from zero');
  assert.equal(second.rank[2999], 999);
}
// ---- dead cull: signal resets the count ------------------------------------
{
  const n = 2000;
  const votes = new Uint16Array(n);
  const masses = new Float32Array(n);
  for (let i = 0; i < n; i++) masses[i] = i < 1500 ? 0 : 3;
  planDeadCull(masses, n, { votes, minCull: 1, floor: 1 });
  assert.equal(votes[0], 1);
  for (let i = 0; i < n; i++) masses[i] = i + 1;
  const p = planDeadCull(masses, n, { votes, minCull: 1, floor: 1 });
  assert.equal(p.culled, 0, 'a good window saves every row');
  assert.equal(votes[0], 0, 'signal clears the count');
  assert.equal(p.candidates, 0, 'no zeros, no candidates');
}
// ---- dead cull: broken stats refuse -----------------------------------------
{
  const n = 100;
  assert.equal(planDeadCull(new Float32Array(n), n, { votes: new Uint16Array(n) }), null,
    'all-zero mass refuses (stats never accumulated)');
  const bad = new Float32Array(n).fill(1);
  bad[50] = NaN;
  assert.equal(planDeadCull(bad, n, { votes: new Uint16Array(n) }), null,
    'non-finite mass refuses');
  assert.throws(() => planDeadCull(new Float32Array(n), n, {}), /votes required/,
    'hysteresis is mandatory');
  assert.throws(() => planDeadCull(new Float32Array(n), n, { votes: new Uint16Array(10) }), /holds 10 rows/,
    'a short vote array refuses instead of misvoting');
}
// ---- dead cull: small kills wait, pathological kills refuse ------------------
{
  const n = 5000;
  const votes = new Uint16Array(n);
  const masses = new Float32Array(n);
  for (let i = 0; i < n; i++) masses[i] = i < 10 ? 0 : i + 1;
  planDeadCull(masses, n, { votes, minCull: 1024, floor: 1 });
  const p = planDeadCull(masses, n, { votes, minCull: 1024, floor: 1 });
  assert.equal(p.culled, 0, 'ten dead rows do not pay for a compaction');
  assert.equal(votes[0], 2, 'but the votes persist for the next window');
  const votes2 = new Uint16Array(n);
  const masses2 = new Float32Array(n);
  for (let i = 0; i < n; i++) masses2[i] = i < 4900 ? 0 : 1;
  planDeadCull(masses2, n, { votes: votes2, minCull: 1, floor: 1000 });
  const q = planDeadCull(masses2, n, { votes: votes2, minCull: 1, floor: 1000 });
  assert.equal(q.culled, 0, 'a cull leaving 100 rows under a 1000 floor refuses');
  const votes3 = new Uint16Array(n);
  const masses3 = new Float32Array(n);
  for (let i = 0; i < n; i++) masses3[i] = i < 3000 ? 0 : i + 1;
  planDeadCull(masses3, n, { votes: votes3, minCull: 1, floor: 1000 });
  const r = planDeadCull(masses3, n, { votes: votes3, minCull: 1, floor: 1000 });
  assert.equal(r.culled, 3000, '2000 survivors over a 1000 floor cull');
  assert.equal(r.kept, 2000);
}
// ---- dead cull: the trainer defaults -----------------------------------------
{
  const n = 20000;
  const votes = new Uint16Array(n);
  const masses = new Float32Array(n);
  for (let i = 0; i < n; i++) masses[i] = i < 2000 ? 0 : i + 1;
  planDeadCull(masses, n, { votes });
  const p = planDeadCull(masses, n, { votes });
  assert.equal(p.culled, 2000, '2000 confirmed dead clear the default minCull');
}

console.log('soft-prune + dead-cull planning (rank + chunk): passed');
