// synthetic.js — renders a synthetic multi-view dataset (a textured room corner)
// with a 2D canvas, so the full SfM -> 3DGS pipeline can be tested end-to-end
// without real photos.
//
// Rendering is deliberately sort-free and view-INDEPENDENT (a global painter's
// sort z-fights on coplanar decals and mis-sorts large quads, which made
// rectangles pop between views — an impossible training target):
//   1. the three corner planes are concave from every camera in the arc, so
//      they never occlude each other — draw in any fixed order, decals after
//      their own base plane
//   2. the floating box is convex and always in front of the corner — draw
//      last, front-facing faces only (visible faces of a convex solid never
//      overlap in projection)

import { processSource, adaptiveTrainCap } from './io/frames.js';
import { makeRng } from './sfm/geometry.js';
import { FACE_ROTS } from './io/pano.js';
export { FACE_ROTS };

const W = 640, H = 480, F = 620;

const lerp3 = (o, u, v, a, b) => [
  o[0] + u[0] * a + v[0] * b,
  o[1] + u[1] * a + v[1] * b,
  o[2] + u[2] * a + v[2] * b,
];

// world: y points DOWN (matches the camera convention), scene centered at origin
// `closed` adds the other three walls + ceiling — a full room, for camera
// rigs INSIDE the scene (360 experiments). Looking out from inside a convex
// room, every ray hits exactly one wall, so the painter's order still holds.
function buildScene(rng, closed = false) {
  const planeDefs = [
    // floor: y = +1.0
    { o: [-1.6, 1.0, -1.6], u: [3.2, 0, 0], v: [0, 0, 3.2], base: '#8a7f6d' },
    // back wall: z = +1.6
    { o: [-1.6, 1.0, 1.6], u: [3.2, 0, 0], v: [0, -2.4, 0], base: '#7d8894' },
    // side wall: x = +1.6
    { o: [1.6, 1.0, 1.6], u: [0, 0, -3.2], v: [0, -2.4, 0], base: '#94847d' },
  ];
  if (closed) {
    planeDefs.push(
      // ceiling: y = -1.4
      { o: [-1.6, -1.4, -1.6], u: [3.2, 0, 0], v: [0, 0, 3.2], base: '#6d7a8a' },
      // front wall: z = -1.6
      { o: [-1.6, 1.0, -1.6], u: [3.2, 0, 0], v: [0, -2.4, 0], base: '#6d8a7f' },
      // left wall: x = -1.6
      { o: [-1.6, 1.0, -1.6], u: [0, 0, 3.2], v: [0, -2.4, 0], base: '#8a6d7f' },
    );
  }
  const planes = [];
  for (const pl of planeDefs) {
    const base = {
      pts: [lerp3(pl.o, pl.u, pl.v, 0, 0), lerp3(pl.o, pl.u, pl.v, 1, 0),
            lerp3(pl.o, pl.u, pl.v, 1, 1), lerp3(pl.o, pl.u, pl.v, 0, 1)],
      color: pl.base,
    };
    const decals = [];
    for (let k = 0; k < 180; k++) {
      const a = rng() * 0.92, b = rng() * 0.92;
      const sa = 0.015 + rng() * 0.09, sb = 0.015 + rng() * 0.09;
      const hue = (rng() * 360) | 0;
      const light = 25 + ((rng() * 55) | 0);
      decals.push({
        pts: [lerp3(pl.o, pl.u, pl.v, a, b), lerp3(pl.o, pl.u, pl.v, a + sa, b),
              lerp3(pl.o, pl.u, pl.v, a + sa, b + sb), lerp3(pl.o, pl.u, pl.v, a, b + sb)],
        color: `hsl(${hue} ${40 + ((rng() * 50) | 0)}% ${light}%)`,
      });
    }
    planes.push({ base, decals });
  }

  // a floating box for extra parallax
  const bx = [-0.3, 0.45, -0.2], bs = 0.55;
  const boxCenter = [bx[0] + 0.5 * bs, bx[1] - 0.5 * bs, bx[2] + 0.5 * bs];
  const c = (dx, dy, dz) => [bx[0] + dx * bs, bx[1] + dy * bs, bx[2] + dz * bs];
  const faceQuads = [
    [c(0, 0, 0), c(1, 0, 0), c(1, -1, 0), c(0, -1, 0)],   // z- side
    [c(0, 0, 1), c(1, 0, 1), c(1, -1, 1), c(0, -1, 1)],   // z+ side
    [c(0, 0, 0), c(0, 0, 1), c(0, -1, 1), c(0, -1, 0)],   // x- side
    [c(1, 0, 0), c(1, 0, 1), c(1, -1, 1), c(1, -1, 0)],   // x+ side
    [c(0, -1, 0), c(1, -1, 0), c(1, -1, 1), c(0, -1, 1)], // top (y-)
  ];
  const boxColors = ['#c25b4e', '#4e8ac2', '#c2a44e', '#5bc24e', '#8a4ec2'];
  const boxFaces = faceQuads.map((pts, fi) => {
    // outward normal via cross of edges, oriented away from the box center
    const e1 = [pts[1][0] - pts[0][0], pts[1][1] - pts[0][1], pts[1][2] - pts[0][2]];
    const e2 = [pts[3][0] - pts[0][0], pts[3][1] - pts[0][1], pts[3][2] - pts[0][2]];
    let nrm = [
      e1[1] * e2[2] - e1[2] * e2[1],
      e1[2] * e2[0] - e1[0] * e2[2],
      e1[0] * e2[1] - e1[1] * e2[0],
    ];
    const mid = [
      (pts[0][0] + pts[2][0]) / 2, (pts[0][1] + pts[2][1]) / 2, (pts[0][2] + pts[2][2]) / 2,
    ];
    const out = [mid[0] - boxCenter[0], mid[1] - boxCenter[1], mid[2] - boxCenter[2]];
    if (nrm[0] * out[0] + nrm[1] * out[1] + nrm[2] * out[2] < 0) nrm = nrm.map((v) => -v);
    // dots on the face (view-independent, drawn right after their face)
    const dots = [];
    for (let k = 0; k < 25; k++) {
      const a = 0.05 + Math.min(0.85, rng() * 0.8), b = 0.05 + Math.min(0.85, rng() * 0.8);
      const s = 0.04 + rng() * 0.10;
      const p = (aa, bb) => [
        pts[0][0] + (pts[1][0] - pts[0][0]) * aa + (pts[3][0] - pts[0][0]) * bb,
        pts[0][1] + (pts[1][1] - pts[0][1]) * aa + (pts[3][1] - pts[0][1]) * bb,
        pts[0][2] + (pts[1][2] - pts[0][2]) * aa + (pts[3][2] - pts[0][2]) * bb,
      ];
      dots.push({
        pts: [p(a, b), p(a + s, b), p(a + s, b + s), p(a, b + s)],
        color: `hsl(${(fi * 67 + ((rng() * 120) | 0)) % 360} 70% ${20 + ((rng() * 60) | 0)}%)`,
      });
    }
    return { pts, color: boxColors[fi], nrm, mid, dots };
  });

  return { planes, boxFaces };
}

function renderView(scene, pose, jitterRng, w = W, h = H, f = F) {
  const cv = document.createElement('canvas');
  cv.width = w; cv.height = h;
  const ctx = cv.getContext('2d');
  ctx.fillStyle = '#1c1c20';
  ctx.fillRect(0, 0, w, h);

  const { R, t } = pose;
  const camPos = [
    -(R[0] * t[0] + R[3] * t[1] + R[6] * t[2]),
    -(R[1] * t[0] + R[4] * t[1] + R[7] * t[2]),
    -(R[2] * t[0] + R[5] * t[1] + R[8] * t[2]),
  ];

  // camera-space transform + Sutherland-Hodgman clip against z = NEAR, then
  // project. Dropping quads with any vertex behind the camera was fine for
  // the outside-looking-in arc, but a rig INSIDE the room sees every adjacent
  // surface partially behind the near plane.
  const NEAR = 0.05;
  const toCam = (p) => [
    R[0] * p[0] + R[1] * p[1] + R[2] * p[2] + t[0],
    R[3] * p[0] + R[4] * p[1] + R[5] * p[2] + t[1],
    R[6] * p[0] + R[7] * p[1] + R[8] * p[2] + t[2],
  ];
  const drawQuad = (q) => {
    let poly = q.pts.map(toCam);
    const out = [];
    for (let i = 0; i < poly.length; i++) {
      const a = poly[i], b = poly[(i + 1) % poly.length];
      const ain = a[2] >= NEAR, bin = b[2] >= NEAR;
      if (ain) out.push(a);
      if (ain !== bin) {
        const s = (NEAR - a[2]) / (b[2] - a[2]);
        out.push([a[0] + (b[0] - a[0]) * s, a[1] + (b[1] - a[1]) * s, NEAR]);
      }
    }
    poly = out;
    if (poly.length < 3) return;
    ctx.fillStyle = q.color;
    ctx.beginPath();
    poly.forEach((p, i) => {
      const px = f * p[0] / p[2] + w / 2, py = f * p[1] / p[2] + h / 2;
      if (i) ctx.lineTo(px, py); else ctx.moveTo(px, py);
    });
    ctx.closePath();
    ctx.fill();
  };

  // 1. corner planes (mutually non-occluding), each base then its decals
  for (const pl of scene.planes) {
    drawQuad(pl.base);
    for (const d of pl.decals) drawQuad(d);
  }
  // 2. box: front-facing faces only, each followed by its dots
  for (const bf of scene.boxFaces) {
    const view = [camPos[0] - bf.mid[0], camPos[1] - bf.mid[1], camPos[2] - bf.mid[2]];
    if (bf.nrm[0] * view[0] + bf.nrm[1] * view[1] + bf.nrm[2] * view[2] <= 0) continue;
    drawQuad(bf);
    for (const d of bf.dots) drawQuad(d);
  }

  // mild sensor-like noise so BRIEF has gradients everywhere
  const id = ctx.getImageData(0, 0, w, h);
  for (let i = 0; i < id.data.length; i += 4) {
    const nz = (jitterRng() - 0.5) * 6;
    id.data[i] += nz; id.data[i + 1] += nz; id.data[i + 2] += nz;
  }
  ctx.putImageData(id, 0, 0);
  return cv;
}

/** Generate `count` raw synthetic views (deterministic, seed 42): the exact
 *  640x480 canvases + ground-truth world-to-camera poses (OpenCV convention:
 *  x right, y down, z forward; pc = R*p + t). */
export function generateSyntheticRaw(count = 12) {
  const rng = makeRng(42);
  const scene = buildScene(rng);
  const out = [];
  const center = [0, 0.2, 0];
  for (let i = 0; i < count; i++) {
    const a = -0.55 + 1.1 * (i / (count - 1));       // yaw arc
    const elev = -0.55 - 0.15 * Math.sin(i * 1.7);   // slight height variation
    const dist = 3.6 + 0.3 * Math.cos(i * 2.3);
    const eye = [
      center[0] + Math.sin(a) * dist * 0.9 - 0.5,
      center[1] + elev,
      center[2] - Math.cos(a) * dist,
    ];
    const pose = lookAtPose(eye, center);
    const cv = renderView(scene, pose, rng);
    out.push({
      name: `synthetic_${String(i).padStart(2, '0')}`,
      canvas: cv, pose, eye, f: F, cx: W / 2, cy: H / 2, w: W, h: H,
    });
  }
  return out;
}

/** Generate `count` synthetic views. Returns the same format as loadImageFiles. */
export function generateSyntheticDataset(count = 12, trainCap, opts = {}) {
  const cap = trainCap || adaptiveTrainCap(count, W, H, opts);
  return generateSyntheticRaw(count).map((v) => processSource(v.canvas, W, H, v.name, cap, opts));
}

// ── 360 rig test data ───────────────────────────────────────────────────────
// FACE_ROTS (imported from io/pano.js): camera-frame rotations for the six
// cube faces — +z, +x, -z, -x, up (-y), down (+y).

/** Generate a 360 rig dataset: `rigs` positions on a loop INSIDE the closed
 *  room, six square pinhole faces each (the cubemap a real equirect pano
 *  would be sliced into). fovDeg > 90 gives adjacent faces a small overlap,
 *  the way real slicers do. Ground truth: every face carries its exact pose,
 *  its rig index and the shared rig centre. Deterministic (seed 42). */
export function generatePanoRigRaw(rigs = 8, size = 512, fovDeg = 100) {
  const rng = makeRng(42);
  const scene = buildScene(rng, true);
  const f = (size / 2) / Math.tan((fovDeg / 2) * Math.PI / 180);
  const out = [];
  for (let r = 0; r < rigs; r++) {
    const a = (r / rigs) * 2 * Math.PI;
    const eye = [Math.sin(a) * 0.9, 0.15 + 0.1 * Math.sin(a * 2), Math.cos(a) * 0.9];
    // a per-rig yaw so faces are not axis-aligned with the walls
    const yaw = a + 0.35;
    const cy0 = Math.cos(yaw), sy0 = Math.sin(yaw);
    const Rrig = [cy0, 0, sy0, 0, 1, 0, -sy0, 0, cy0];
    for (let k = 0; k < 6; k++) {
      const Fk = FACE_ROTS[k];
      // world-to-camera: face rotation applied on top of the rig yaw
      const R = [
        Fk[0] * Rrig[0] + Fk[1] * Rrig[3] + Fk[2] * Rrig[6],
        Fk[0] * Rrig[1] + Fk[1] * Rrig[4] + Fk[2] * Rrig[7],
        Fk[0] * Rrig[2] + Fk[1] * Rrig[5] + Fk[2] * Rrig[8],
        Fk[3] * Rrig[0] + Fk[4] * Rrig[3] + Fk[5] * Rrig[6],
        Fk[3] * Rrig[1] + Fk[4] * Rrig[4] + Fk[5] * Rrig[7],
        Fk[3] * Rrig[2] + Fk[4] * Rrig[5] + Fk[5] * Rrig[8],
        Fk[6] * Rrig[0] + Fk[7] * Rrig[3] + Fk[8] * Rrig[6],
        Fk[6] * Rrig[1] + Fk[7] * Rrig[4] + Fk[8] * Rrig[7],
        Fk[6] * Rrig[2] + Fk[7] * Rrig[5] + Fk[8] * Rrig[8],
      ];
      const t = [
        -(R[0] * eye[0] + R[1] * eye[1] + R[2] * eye[2]),
        -(R[3] * eye[0] + R[4] * eye[1] + R[5] * eye[2]),
        -(R[6] * eye[0] + R[7] * eye[1] + R[8] * eye[2]),
      ];
      const pose = { R, t };
      const cv = renderView(scene, pose, rng, size, size, f);
      out.push({
        name: `rig${String(r).padStart(2, '0')}_f${k}`,
        canvas: cv, pose, eye, rig: r, face: k,
        f, cx: size / 2, cy: size / 2, w: size, h: size,
      });
    }
  }
  return out;
}

function lookAtPose(eye, center) {
  // world up (y is down) => up direction is -y
  const worldUp = [0, -1, 0];
  const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
  const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
  const norm = (a) => { const l = Math.hypot(...a); return [a[0] / l, a[1] / l, a[2] / l]; };
  const zc = norm(sub(center, eye));
  const xc = norm(cross(zc, worldUp));
  const yc = cross(zc, xc);
  const R = [xc[0], xc[1], xc[2], yc[0], yc[1], yc[2], zc[0], zc[1], zc[2]];
  const t = [
    -(R[0] * eye[0] + R[1] * eye[1] + R[2] * eye[2]),
    -(R[3] * eye[0] + R[4] * eye[1] + R[5] * eye[2]),
    -(R[6] * eye[0] + R[7] * eye[1] + R[8] * eye[2]),
  ];
  return { R, t };
}
