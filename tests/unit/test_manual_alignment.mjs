import assert from 'node:assert/strict';
import { manualPoseFromCorrespondences } from '../../app/js/align.js';
import { reprojError } from '../../src/sfm/geometry.js';

const intr = { f: 900, w: 1200, h: 900, cx: 600, cy: 450 };
const pose = (R, C) => ({ ...intr, R, t: [
  -(R[0] * C[0] + R[1] * C[1] + R[2] * C[2]),
  -(R[3] * C[0] + R[4] * C[1] + R[5] * C[2]),
  -(R[6] * C[0] + R[7] * C[1] + R[8] * C[2]),
] });
const I = [1,0,0, 0,1,0, 0,0,1];
const a = .11, targetR = [Math.cos(a),0,Math.sin(a), 0,1,0, -Math.sin(a),0,Math.cos(a)];
const cams = [pose(targetR, [.18, -.04, .08]), pose(I, [-.7, 0, 0]), pose(I, [.75, .08, .05])];
const points = [
  [-.9,-.5,4.2], [.8,-.55,4.7], [-.65,.62,5.1], [.72,.58,5.7], [.05,.02,6.4],
];
const mark = (cam, X) => {
  const z = cam.R[6]*X[0] + cam.R[7]*X[1] + cam.R[8]*X[2] + cam.t[2];
  const xn = (cam.R[0]*X[0] + cam.R[1]*X[1] + cam.R[2]*X[2] + cam.t[0]) / z;
  const yn = (cam.R[3]*X[0] + cam.R[4]*X[1] + cam.R[5]*X[2] + cam.t[1]) / z;
  return { x: (xn * cam.f + cam.cx) / cam.w, y: (yn * cam.f + cam.cy) / cam.h };
};
const marks = new Map(cams.map((cam, i) => [i, points.map((X) => mark(cam, X))]));
const solved = manualPoseFromCorrespondences({ cams, target: 0, marks });
assert.equal(solved.inliers, 5, 'the requested five numbered landmarks should be sufficient');
assert.ok(solved.rmsPx < .1, `manual pose should reproject exact landmarks (got ${solved.rmsPx} px)`);
for (let i = 0; i < points.length; i++) {
  const p = marks.get(0)[i];
  const xn = [(p.x * intr.w - intr.cx) / intr.f, (p.y * intr.h - intr.cy) / intr.f];
  assert.ok(reprojError(solved.R, solved.t, points[i], xn) * intr.f < .2);
}

const source = await import('node:fs').then(({ readFileSync }) =>
  readFileSync(new URL('../../app/js/app.js', import.meta.url), 'utf8'));
assert.match(source, /Automatically[\s\S]*Manually/,
  'Align should offer automatic and manual modes');
assert.match(source, /targetPoints\.length < 5[\s\S]*Place at least 5 points/,
  'the alignment image must receive five points before reference selection');
assert.match(source, /Hold and drag for 5× slower crosshair movement/,
  'the editor should explain its precision movement and loupe interaction');
assert.match(source, /manualPoseFromCorrespondences\([\s\S]*cloud: S\.scene\.xyz/,
  'Finish should solve against registered cameras and the reconstruction cloud');

console.log('manual numbered-feature camera alignment: passed');
