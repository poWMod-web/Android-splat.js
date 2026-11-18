import assert from 'node:assert/strict';
import { selectRetrievedPairs } from '../../src/sfm/gpumatch.js';

const base = [[0, 1], [1, 2], [2, 3], [3, 4]];
const scores = [
  { i: 0, j: 4, score: 18 },
  { i: 0, j: 3, score: 12 },
  { i: 1, j: 4, score: 9 },
  { i: 1, j: 3, score: 2 }, // below retrieval floor
];
const pairs = selectRetrievedPairs(5, base, scores, 1, 4);
const keys = new Set(pairs.map((p) => p.join(':')));
for (const p of base) assert.ok(keys.has(p.join(':')), 'ordered graph edges must survive retrieval');
assert.ok(keys.has('0:4'), 'the strongest unordered overlap should be added');
assert.ok(keys.has('1:4'), 'per-image partner selection should connect otherwise distant views');
assert.ok(!keys.has('1:3'), 'weak descriptor votes should not expand the expensive full graph');
assert.deepEqual(pairs, [...pairs].sort((a, b) => a[0] - b[0] || a[1] - b[1]), 'pair order must be deterministic');
console.log('GPU descriptor retrieval pair selection: passed');
