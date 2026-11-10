import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { levelingNote } from '../../app/js/xr-dataset.js';

const java = readFileSync(new URL('../../android/app/src/main/java/space/arrival/splatxr/NativeCaptureActivity.java', import.meta.url), 'utf8');
const ui = readFileSync(new URL('../../app/js/xr-ui.js', import.meta.url), 'utf8');
const ds = readFileSync(new URL('../../app/js/xr-dataset.js', import.meta.url), 'utf8');

// The native export levels the world to the ARCore ground plane so a rolled
// session world cannot come out of training sideways.
assert.match(java, /import com\.google\.ar\.core\.Plane;/, 'the exporter should read ARCore planes');
assert.match(java, /setPlaneFindingMode\(Config\.PlaneFindingMode\.HORIZONTAL_AND_VERTICAL\)/,
  'plane trackables should be requested explicitly');
assert.match(java, /if \(tracking\) \{\s*updateGroundPlane\(\);/,
  'the best ground plane should be polled on every tracking frame');
assert.match(java, /getAllTrackables\(Plane\.class\)/, 'planes should be polled from the session');
assert.match(java, /HORIZONTAL_UPWARD_FACING/, 'only horizontal upward planes should level the world');
assert.match(java, /getSubsumedBy\(\) != null/, 'subsumed planes should be skipped');
assert.match(java, /center\.getYAxis\(\)/, "the plane normal should be the plane pose's +Y axis");
assert.match(java, /area >= 0\.04f/, 'plane patches smaller than ~20x20 cm should be ignored');
assert.match(java, /area < groundArea \* 1\.2f/, 'the best plane should win with hysteresis');
assert.match(java, /manual capture"\) \+ groundPlaneStatus\(\)/,
  'the capture status should show the ground-plane state');
assert.match(java, /hasGroundPlane = false; groundArea = 0f;/, 'reset should clear the plane snapshot');
assert.match(java, /JSONObject leveling = applyGroundLeveling\(\);/,
  'the export should level before serializing the dataset');
assert.match(java, /if \(toward < 0\) \{ up\[0\] = -up\[0\]; up\[1\] = -up\[1\]; up\[2\] = -up\[2\]; \}/,
  'the plane normal should face the cameras (up, never upside down)');
assert.match(java, /spread >= 0\.7f/, 'the camera-up fallback should require agreeing cameras');
assert.match(java, /if \(up\[1\] < 0\) \{ up\[0\] = -up\[0\]; up\[1\] = -up\[1\]; up\[2\] = -up\[2\]; \}/,
  'the fallback should keep the session +Y hemisphere (never flip a level world)');
assert.match(java, /angleDeg >= 0\.5f/, 'an already-level world should be left alone');
assert.match(java, /float ax = -up\[2\], az = up\[0\];/, 'the leveling axis should be up x +Y');
assert.match(java, /c \+ k1c \* kx \* kx, -s \* kz, k1c \* kx \* kz,/,
  'the leveling matrix should be the Rodrigues shortest arc');
assert.match(java, /s \* kz, c, -s \* kx,/,
  'the leveling matrix should be the Rodrigues shortest arc');
assert.match(java, /k1c \* kx \* kz, s \* kx, c \+ k1c \* kz \* kz/,
  'the leveling matrix should be the Rodrigues shortest arc');
assert.match(java, /out\[row\]\[3\] = r\[row \* 3\] \* p\[0\] \+ r\[row \* 3 \+ 1\] \* p\[1\] \+ r\[row \* 3 \+ 2\] \* p\[2\] \+ centroid\[row\];/,
  'camera centres should rotate about the centroid');
assert.match(java, /entry\.setValue\(new SeedPoint\(/, 'seed points should be leveled with the cameras');
assert.match(java, /\.put\("leveling", leveling\)/, 'capture.json should record the leveling provenance');
assert.match(java, /levelingApplied = true;/, 'leveling should apply exactly once');
assert.match(ui, /levelingNote\(dataset\.metadata\?\.leveling\)/,
  'the review dialog should show the leveling provenance');
assert.match(ds, /export function levelingNote\(leveling\)/, 'the leveling note should be a pure export');

// The exact rotation the Java builds (R = cI + sK + (1-c)kk^T,
// k = normalize(up x +Y)): it must carry any sign-fixed up onto +Y while
// staying a proper rotation.
function javaLevelingRotation(up) {
  const cos = Math.max(-1, Math.min(1, up[1]));
  const ax = -up[2], az = up[0];
  const s = Math.sqrt(ax * ax + az * az);
  const kx = ax / s, kz = az / s;
  const c = cos, k1c = 1 - c;
  return [
    c + k1c * kx * kx, -s * kz, k1c * kx * kz,
    s * kz, c, -s * kx,
    k1c * kx * kz, s * kx, c + k1c * kz * kz,
  ];
}
const apply = (r, v) => [0, 1, 2].map((i) => r[i * 3] * v[0] + r[i * 3 + 1] * v[1] + r[i * 3 + 2] * v[2]);
const det = (r) => r[0] * (r[4] * r[8] - r[5] * r[7]) - r[1] * (r[3] * r[8] - r[5] * r[6]) + r[2] * (r[3] * r[7] - r[4] * r[6]);
const norm = (v) => Math.hypot(...v);
let seed = 42;
const rand = () => ((seed = (seed * 1664525 + 1013904223) >>> 0) / 4294967296);
const cases = [[1, 0, 0], [-1, 0, 0], [0, 0, 1], [0, 0, -1]]; // the sideways session: up 90 deg off
for (let i = 0; i < 200; i++) {
  const v = [rand() * 2 - 1, rand(), rand() * 2 - 1]; // sign-fixed: up.y > 0 like the Java guarantees
  const n = norm(v);
  if (n < 1e-6 || v[1] / n > Math.cos(0.5 * Math.PI / 180)) continue; // skip the sub-gate level case
  cases.push([v[0] / n, v[1] / n, v[2] / n]);
}
for (const up of cases) {
  const r = javaLevelingRotation(up);
  const mapped = apply(r, up);
  assert.ok(Math.abs(mapped[0]) < 1e-9 && Math.abs(mapped[1] - 1) < 1e-9 && Math.abs(mapped[2]) < 1e-9,
    `the leveling rotation should carry up ${up.map((v) => v.toFixed(3))} onto +Y`);
  for (let i = 0; i < 3; i++) {
    const row = [r[i * 3], r[i * 3 + 1], r[i * 3 + 2]];
    assert.ok(Math.abs(norm(row) - 1) < 1e-9, 'leveling rows should stay unit length');
    for (let j = i + 1; j < 3; j++) {
      const other = [r[j * 3], r[j * 3 + 1], r[j * 3 + 2]];
      assert.ok(Math.abs(row[0] * other[0] + row[1] * other[1] + row[2] * other[2]) < 1e-9,
        'leveling rows should stay orthogonal');
    }
  }
  assert.ok(Math.abs(det(r) - 1) < 1e-9, 'the leveling rotation should keep determinant 1 (no mirror)');
}
// End to end: ground-plane points level to one height with the cameras above.
{
  const up = [0.2, 0.9, -0.35], n = norm(up), u = up.map((v) => v / n);
  const r = javaLevelingRotation(u);
  const centroid = [1.5, 2.0, -0.5], ground = [1.0, 1.2, -0.3]; // cameras orbit above the floor
  const level = (p) => apply(r, [p[0] - centroid[0], p[1] - centroid[1], p[2] - centroid[2]])
    .map((v, i) => v + centroid[i]);
  const arb = Math.abs(u[1]) < 0.9 ? [0, 1, 0] : [1, 0, 0];
  const cross = (p, q) => [p[1] * q[2] - p[2] * q[1], p[2] * q[0] - p[0] * q[2], p[0] * q[1] - p[1] * q[0]];
  const e1 = cross(u, arb), l1 = norm(e1), t1 = e1.map((v) => v / l1);
  const t2 = cross(u, t1); // already unit: the plane's tangent basis
  const pts = [[0, 0], [0.5, 0], [0, 0.5], [-0.4, 0.3]].map(([a, b]) =>
    [ground[0] + a * t1[0] + b * t2[0], ground[1] + a * t1[1] + b * t2[1], ground[2] + a * t1[2] + b * t2[2]]);
  const heights = pts.map((p) => level(p)[1]);
  assert.ok(Math.max(...heights) - Math.min(...heights) < 1e-9, 'the ground plane should level to one height');
  assert.ok(level(centroid)[1] > heights[0], 'the cameras should stay above the leveled ground');
}

// The review-dialog note covers every provenance outcome.
assert.equal(levelingNote(null), '');
assert.equal(levelingNote(undefined), '');
assert.equal(levelingNote({ method: 'arcore-horizontal-plane', applied: true, tiltDegrees: 89.7, planeArea: 1.234 }),
  'World levelled to the ARCore ground plane (1.23 m² ground plane); 89.7° tilt removed.');
assert.equal(levelingNote({ method: 'arcore-horizontal-plane', applied: false, tiltDegrees: 0.2, planeArea: 0.5 }),
  'ARCore ground plane found (0.50 m² ground plane); the session world was already level.');
assert.equal(levelingNote({ method: 'camera-up-average', applied: true, tiltDegrees: 12.34 }),
  "No ground plane in view; levelled to the cameras' average up (12.3°). Hold the phone upright while capturing.");
assert.equal(levelingNote({ method: 'none', applied: false }),
  'No ground plane in view and the camera orientations disagree; the session world is kept as captured.');

console.log('native ARCore ground-plane leveling: passed');
