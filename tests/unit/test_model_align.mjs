import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

const app = readFileSync(new URL('../../app/js/app.js', import.meta.url), 'utf8');
const pcview = readFileSync(new URL('../../app/js/pcview.js', import.meta.url), 'utf8');
const sessionIo = readFileSync(new URL('../../app/js/session_io.js', import.meta.url), 'utf8');

// Viewer-side alignment: the render funnel counter-rotates the orbit pose so
// the displayed model turns about the scene centre; training data, cameras
// and exports are untouched.
assert.match(app, /const pose = applyModelAlign\(vp\.viewPose\(\)\);/,
  'the render funnel should counter-rotate the orbit pose for the locked alignment');
assert.match(app, /function alignMatrix\(sx, sy, sz\)/,
  'the 90-degree-step alignment rotation should exist');
assert.match(app, /R\[i \* 3\] \* At\[j\] \+ R\[i \* 3 \+ 1\] \* At\[3 \+ j\] \+ R\[i \* 3 \+ 2\] \* At\[6 \+ j\]/,
  'the aligned camera should carry R·A-transpose');
assert.match(app, /function storedAlign\(r\)/, 'the run record’s lock should load validated');
assert.match(app, /function alignCard\(\)/, 'the Align card should exist');
assert.match(app, /id="al-rx"[\s\S]*id="al-ry"[\s\S]*id="al-rz"[\s\S]*id="al-reset"[\s\S]*id="al-lock"[\s\S]*id="al-done"/,
  'the card should offer per-axis 90-degree steps, reset, lock and done');
assert.match(app, /patchRun\(S\.runId, \{ align/,
  'the alignment lock should persist to the run record');
assert.match(app, /id = 'c-align'/, 'the done controls should offer Align');
assert.match(app, /S\.align = storedAlign\(S\._localRun\);/,
  'restored runs should reopen with their locked alignment');
assert.match(app, /S\.align = storedAlign\(lr\);/,
  'continuations should view the run’s locked alignment');
assert.match(app, /S\.align = null;   \/\/ a new model starts unaligned/,
  'a new training should start unaligned');
assert.match(app, /S\.align = null;   \/\/ a fresh viewing starts unaligned/,
  'a fresh viewing should start unaligned');

// Known-camera (AR) worlds are y-up: the SOG-lite viewer maps them straight
// across instead of the y-down half-turn. Old recons lack the marker and
// keep the legacy flip.
assert.match(sessionIo, /knownCameras: !!recon\.known,/,
  'the recon JSON should record the known-camera world frame');
assert.match(app, /createSogView\(src\.url, \{ radius: reconJson\.sceneRadius \|\| 10, yUp: !!reconJson\.knownCameras \}\)/,
  'the lite viewer should learn the model’s world frame');
assert.match(pcview, /createSogView\(sogUrl, \{ radius = 10, yUp = false \} = \{\}\)/,
  'the lite viewer should default to the legacy y-down flip');
assert.match(pcview, /setEulerAngles\(yUp \? 0 : 180, 0, 0\)/,
  'y-up models should skip the entity half-turn');
assert.match(pcview, /const s = yUp \? 1 : -1;/,
  'the lite camera mapping should branch on the world frame');

// The exact alignment math the app builds: 90-degree steps compose to proper
// rotations, and the camera-side equivalent renders the turned model.
function alignMatrix(sx, sy, sz) {
  const step = (s) => ((s % 4) + 4) % 4;
  const rot = (axis, s) => {
    const [c, si] = [[1, 0], [0, 1], [-1, 0], [0, -1]][step(s)];
    if (axis === 0) return [1, 0, 0, 0, c, -si, 0, si, c];
    if (axis === 1) return [c, 0, si, 0, 1, 0, -si, 0, c];
    return [c, -si, 0, si, c, 0, 0, 0, 1];
  };
  const mul = (P, Q) => [0, 1, 2].map((i) => [0, 1, 2].map((j) =>
    P[i * 3] * Q[j] + P[i * 3 + 1] * Q[3 + j] + P[i * 3 + 2] * Q[6 + j])).flat();
  return mul(mul(rot(2, sz), rot(1, sy)), rot(0, sx));
}
const applyMat = (m, v) => [0, 1, 2].map((i) => m[i * 3] * v[0] + m[i * 3 + 1] * v[1] + m[i * 3 + 2] * v[2]);
const det = (m) => m[0] * (m[4] * m[8] - m[5] * m[7]) - m[1] * (m[3] * m[8] - m[5] * m[6]) + m[2] * (m[3] * m[7] - m[4] * m[6]);
const eq = (a, b, tol = 1e-9) => a.every((v, i) => Math.abs(v - b[i]) < tol);
// single steps turn the expected quarter-turns (right-hand rule)
assert.ok(eq(applyMat(alignMatrix(1, 0, 0), [0, 1, 0]), [0, 0, 1]), 'X+90 should carry +Y onto +Z');
assert.ok(eq(applyMat(alignMatrix(0, 1, 0), [0, 0, 1]), [1, 0, 0]), 'Y+90 should carry +Z onto +X');
assert.ok(eq(applyMat(alignMatrix(0, 0, 1), [1, 0, 0]), [0, 1, 0]), 'Z+90 should carry +X onto +Y');
assert.ok(eq(alignMatrix(0, 0, 0), [1, 0, 0, 0, 1, 0, 0, 0, 1]), 'zero steps should be identity');
// every combination stays a proper rotation and wraps every four steps
for (let x = 0; x < 4; x++) for (let y = 0; y < 4; y++) for (let z = 0; z < 4; z++) {
  const m = alignMatrix(x, y, z);
  assert.ok(Math.abs(det(m) - 1) < 1e-9, `steps (${x},${y},${z}) should keep determinant 1`);
  for (let i = 0; i < 3; i++) {
    const row = [m[i * 3], m[i * 3 + 1], m[i * 3 + 2]];
    assert.ok(Math.abs(Math.hypot(...row) - 1) < 1e-9, 'alignment rows should stay unit length');
  }
  assert.ok(eq(m, alignMatrix(x + 4, y - 4, z + 8)), 'steps should wrap every four');
}
// rendering equivalence: a point under the plain camera reads exactly like
// the turned point under the counter-rotated camera (the pivot never moves)
function alignCamera(R, t, c, sx, sy, sz) {
  const A = alignMatrix(sx, sy, sz);
  const At = [A[0], A[3], A[6], A[1], A[4], A[7], A[2], A[5], A[8]];
  const u = [0, 1, 2].map((i) => c[i] - (At[i * 3] * c[0] + At[i * 3 + 1] * c[1] + At[i * 3 + 2] * c[2]));
  return {
    R: [0, 1, 2].map((i) => [0, 1, 2].map((j) =>
      R[i * 3] * At[j] + R[i * 3 + 1] * At[3 + j] + R[i * 3 + 2] * At[6 + j])).flat(),
    t: [0, 1, 2].map((i) => t[i] + R[i * 3] * u[0] + R[i * 3 + 1] * u[1] + R[i * 3 + 2] * u[2]),
  };
}
let seed = 7;
const rand = () => ((seed = (seed * 1664525 + 1013904223) >>> 0) / 4294967296);
// random cameras from axis-angle rotations plus translations
const randRot = () => {
  const a = [rand() * 2 - 1, rand() * 2 - 1, rand() * 2 - 1], n = Math.hypot(...a) || 1;
  const [x, y, z] = a.map((v) => v / n), t = rand() * Math.PI * 2, c = Math.cos(t), s = Math.sin(t);
  return [c + x * x * (1 - c), x * y * (1 - c) - z * s, x * z * (1 - c) + y * s,
    y * x * (1 - c) + z * s, c + y * y * (1 - c), y * z * (1 - c) - x * s,
    z * x * (1 - c) - y * s, z * y * (1 - c) + x * s, c + z * z * (1 - c)];
};
for (let trial = 0; trial < 60; trial++) {
  const R = randRot();
  const t = [rand() * 4 - 2, rand() * 4 - 2, rand() * 4 - 2];
  const c = [rand() * 4 - 2, rand() * 4 - 2, rand() * 4 - 2];
  const steps = [rand() * 8 - 2, rand() * 8 - 2, rand() * 8 - 2].map(Math.round);
  if (!(steps[0] % 4) && !(steps[1] % 4) && !(steps[2] % 4)) continue; // the app skips identity
  const A = alignMatrix(...steps);
  const cam = alignCamera(R, t, c, ...steps);
  for (let k = 0; k < 4; k++) {
    const p = [rand() * 4 - 2, rand() * 4 - 2, rand() * 4 - 2];
    const turned = applyMat(A, [p[0] - c[0], p[1] - c[1], p[2] - c[2]]).map((v, i) => v + c[i]);
    const plain = applyMat(R, p).map((v, i) => v + t[i]);
    const shown = applyMat(cam.R, turned).map((v, i) => v + cam.t[i]);
    assert.ok(eq(plain, shown, 1e-8), 'the counter-rotated camera should render the turned model identically');
  }
  const pivotPlain = applyMat(R, c).map((v, i) => v + t[i]);
  const pivotShown = applyMat(cam.R, c).map((v, i) => v + cam.t[i]);
  assert.ok(eq(pivotPlain, pivotShown, 1e-8), 'the alignment pivot should render unmoved');
}
// the lite mapping: y-up passes positions and camera axes straight through,
// y-down keeps the legacy (x,-y,-z) flip exactly
function liteCam(R, t, s) {
  const C = [0, 1, 2].map((k) => -(R[k] * t[0] + R[k + 3] * t[1] + R[k + 6] * t[2]));
  return {
    pos: [C[0], s * C[1], s * C[2]],
    right: [R[0], s * R[1], s * R[2]],
    up: [-R[3], -s * R[4], -s * R[5]],
    back: [-R[6], -s * R[7], -s * R[8]],
  };
}
{
  const R = randRot(), t = [0.5, -1.2, 0.3];
  const C = [0, 1, 2].map((k) => -(R[k] * t[0] + R[k + 3] * t[1] + R[k + 6] * t[2]));
  const yup = liteCam(R, t, 1), legacy = liteCam(R, t, -1);
  assert.ok(eq(yup.pos, C), 'y-up positions should pass through');
  assert.ok(eq(yup.right, [R[0], R[1], R[2]]), 'y-up right should pass through');
  assert.ok(eq(yup.up, [-R[3], -R[4], -R[5]]), 'y-up up should be the negated second row');
  assert.ok(eq(yup.back, [-R[6], -R[7], -R[8]]), 'y-up back should be the negated third row');
  assert.ok(eq(legacy.pos, [C[0], -C[1], -C[2]]), 'legacy positions should keep the flip');
  assert.ok(eq(legacy.right, [R[0], -R[1], -R[2]]), 'legacy right should keep the flip');
  assert.ok(eq(legacy.up, [-R[3], R[4], R[5]]), 'legacy up should keep the flip');
  assert.ok(eq(legacy.back, [-R[6], R[7], R[8]]), 'legacy back should keep the flip');
}

console.log('viewer-side model alignment and lite y-up worlds: passed');
