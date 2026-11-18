import assert from 'node:assert/strict';
import { alignReconstructionToKnown } from '../../src/io/cameras.js';

const Q = [0, -1, 0, 1, 0, 0, 0, 0, 1];
const scale = 2.25, offset = [3, -2, 1];
const mv = (M, v) => [0, 1, 2].map((r) => M[r * 3] * v[0] + M[r * 3 + 1] * v[1] + M[r * 3 + 2] * v[2]);
const map = (v) => mv(Q, v).map((x, k) => scale * x + offset[k]);
const cam = (C, imgIdx) => ({ imgIdx, R: [1, 0, 0, 0, 1, 0, 0, 0, 1], t: C.map((x) => -x), f: 900 });
const centres = [[0, 0, 0], [1, 0, 0], [0, 2, 0], [0, 0, 3], [-1, 1, 2]];
const recon = {
  cams: centres.map(cam),
  points: [{ X: [1, 2, 3], rgb: [0.2, 0.4, 0.6] }],
  fFeat: 900,
};
const known = { cams: centres.map((C, i) => cam(map(C), i)) };
const aligned = alignReconstructionToKnown(recon, known);

assert.equal(aligned.alignedToKnown, true);
assert.equal(aligned.knownAlignment.cameras, centres.length);
assert.ok(Math.abs(aligned.knownAlignment.scale - scale) < 1e-5);
assert.ok(aligned.knownAlignment.rms < 1e-5);
for (let i = 0; i < centres.length; i++) {
  const R = aligned.cams[i].R, t = aligned.cams[i].t;
  const C = [0, 1, 2].map((k) => -(R[k] * t[0] + R[3 + k] * t[1] + R[6 + k] * t[2]));
  assert.ok(Math.hypot(...C.map((x, k) => x - map(centres[i])[k])) < 1e-5);
}
assert.ok(Math.hypot(...aligned.points[0].X.map((x, k) => x - map([1, 2, 3])[k])) < 1e-5);
assert.deepEqual(aligned.points[0].rgb, recon.points[0].rgb);
console.log('SfM reconstruction similarity-alignment to known AR poses: passed');
