// viewport.js — the camera, and everything drawn AROUND the model.
//
// The model itself is rendered by the splat.js trainer (WebGPU) into an
// offscreen canvas the app blits underneath; this module owns the one camera
// (free orbit, or locked to a photograph's pose), draws the sparse-point view
// used during the solve, and overlays camera frustums + the capture path with
// the exact same projection the trainer used — so the two layers register.
//
// Adapted from the mockup's painter viewport; the fake splat rasterizer is
// gone, and the orbit is built around a measured dominant-up axis (SfM's
// world frame is arbitrary — detectUp levels the horizon) without touching
// the data.

/** world position of a camera from its world-to-camera pose */
export const camCentre = ({ R, t }) => [
  -(R[0] * t[0] + R[3] * t[1] + R[6] * t[2]),
  -(R[1] * t[0] + R[4] * t[1] + R[7] * t[2]),
  -(R[2] * t[0] + R[5] * t[1] + R[8] * t[2]),
];

// ── rotations ───────────────────────────────────────────────────────────────
// Shared with the intro-cutscene exporter: the flight the viewer plays and
// the one the space plays have to be built from the SAME maths.
export function quatFromR(R) {
  const tr = R[0] + R[4] + R[8];
  if (tr > 0) {
    const s = Math.sqrt(tr + 1) * 2;
    return [(R[7] - R[5]) / s, (R[2] - R[6]) / s, (R[3] - R[1]) / s, 0.25 * s];
  }
  if (R[0] > R[4] && R[0] > R[8]) {
    const s = Math.sqrt(1 + R[0] - R[4] - R[8]) * 2;
    return [0.25 * s, (R[1] + R[3]) / s, (R[2] + R[6]) / s, (R[7] - R[5]) / s];
  }
  if (R[4] > R[8]) {
    const s = Math.sqrt(1 + R[4] - R[0] - R[8]) * 2;
    return [(R[1] + R[3]) / s, 0.25 * s, (R[5] + R[7]) / s, (R[2] - R[6]) / s];
  }
  const s = Math.sqrt(1 + R[8] - R[0] - R[4]) * 2;
  return [(R[2] + R[6]) / s, (R[5] + R[7]) / s, 0.25 * s, (R[3] - R[1]) / s];
}

export function quatToR(q) {
  const [x, y, z, w] = q;
  return [
    1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
    2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
    2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y),
  ];
}

export function qslerp(a, b, t) {
  let d = a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3];
  let bb = b;
  if (d < 0) { d = -d; bb = [-b[0], -b[1], -b[2], -b[3]]; }
  if (d > 0.9995) {
    const o = a.map((v, i) => v + (bb[i] - v) * t);
    const n = Math.hypot(o[0], o[1], o[2], o[3]) || 1;
    return o.map((v) => v / n);
  }
  const th = Math.acos(Math.min(1, d)), s = Math.sin(th);
  const wa = Math.sin((1 - t) * th) / s, wb = Math.sin(t * th) / s;
  return [0, 1, 2, 3].map((i) => a[i] * wa + bb[i] * wb);
}

export class Viewport {
  constructor(canvas) {
    this.cv = canvas;
    // alpha: the model is a separate WebGPU canvas composited UNDERNEATH by
    // the browser — this canvas carries only overlays (and opaque solve views)
    this.ctx = canvas.getContext('2d');
    this.scene = null;          // { xyz, rgb, center, radius }
    this.lock = null;           // a camera whose pose the view sits on
    this.yaw = 0.6; this.pitch = 0.22; this.dist = 8; this.target = [0, 0, 0];
    this.fpv = false;           // per-set: rotate around the EYE, wheel walks
    this.setUp([0, -1, 0]);     // COLMAP-ish default until detectUp measures
    this.dirty = true;
    this.w = 1; this.h = 1;
    this.enabled = true;
    this.freeF = null;          // focal carried over when leaving a frame pose
    this.onLeave = null;        // called when a drag pulls off a frame pose
    this._bind();
  }

  _bind() {
    const cv = this.cv;
    const pts = new Map();   // pointerId -> {x, y} (all active pointers)
    let mode = null;         // 'orbit' | 'pan' | 'two'
    let pinch = null;        // two-finger baseline { d, cx, cy }

    // camera-relative pan: sideways along the view's right, vertical along
    // its up — content follows the fingers from any orientation
    const panBy = (dx, dy) => {
      const s = this.dist * 0.0016;
      const { right, down } = this._basis();
      for (let i = 0; i < 3; i++) {
        this.target[i] -= (right[i] * dx + down[i] * dy) * s;
      }
    };

    const twoBaseline = () => {
      const v = [...pts.values()];
      pinch = {
        d: Math.max(20, Math.hypot(v[1].x - v[0].x, v[1].y - v[0].y)),
        cx: (v[0].x + v[1].x) / 2, cy: (v[0].y + v[1].y) / 2,
      };
      mode = 'two';
    };

    cv.addEventListener('pointerdown', (e) => {
      if (!this.enabled) return;
      if (!pts.size) {
        if (this.lock) this.onLeave?.();
        this.onDragStart?.();
        this._cancelGlide();
        this._down = { x: e.clientX, y: e.clientY, t: performance.now() };
        // every drag orbits around the content actually in front of the
        // camera — a stale far pivot flings the view on the first move
        // (first-person sets rotate about the eye; no pivot to anchor)
        if (!this.fpv) this.anchorPivot();
        mode = (e.shiftKey || e.button === 2 || e.button === 1) ? 'pan' : 'orbit';
      }
      pts.set(e.pointerId, { x: e.clientX, y: e.clientY });
      if (pts.size >= 2) twoBaseline();
      try { cv.setPointerCapture(e.pointerId); } catch { /* synthetic events */ }
    });


    cv.addEventListener('pointermove', (e) => {
      const p = pts.get(e.pointerId);
      if (!p) return;
      const dx = e.clientX - p.x, dy = e.clientY - p.y;
      p.x = e.clientX; p.y = e.clientY;

      if (mode === 'two') {
        if (pts.size < 2) return;
        // pinch = dolly, centroid = pan — the standard two-finger camera
        const v = [...pts.values()];
        const d = Math.max(20, Math.hypot(v[1].x - v[0].x, v[1].y - v[0].y));
        const cx = (v[0].x + v[1].x) / 2, cy = (v[0].y + v[1].y) / 2;
        if (this.fpv) {
          // pinch out = walk forward
          const { fwd } = this._basis();
          const step = (((this.scene && this.scene.radius) || 10) * 0.06) * (d / pinch.d - 1);
          for (let i = 0; i < 3; i++) this.target[i] += fwd[i] * step;
        } else {
          this.dist = Math.max(0.05, Math.min(400, this.dist * pinch.d / d));
        }
        panBy(cx - pinch.cx, cy - pinch.cy);
        pinch = { d, cx, cy };
        this.dirty = true;
        return;
      }

      if (mode === 'pan') {
        panBy(dx, dy);
      } else if (this.fpv) {
        // first person, tuned like dragging a 360 sphere: the grabbed pixel
        // stays under the finger, so the angle per pixel is 1/focal
        const { fwd } = this._basis();
        const eye = [
          this.target[0] - fwd[0] * this.dist,
          this.target[1] - fwd[1] * this.dist,
          this.target[2] - fwd[2] * this.dist];
        const f = this.freeF || Math.min(this.w, this.h) * 0.86;
        this.yaw -= dx / f;
        this.pitch = Math.max(-1.45, Math.min(1.45, this.pitch + dy / f));
        const { fwd: f2 } = this._basis();
        this.target = [
          eye[0] + f2[0] * this.dist,
          eye[1] + f2[1] * this.dist,
          eye[2] + f2[2] * this.dist];
      } else {
        // the up-frame has fixed handedness, so one sign fits every world;
        // + matches the feel the app always had in y-down reconstructions
        this.yaw += dx * 0.006;
        this.pitch = Math.max(-1.45, Math.min(1.45, this.pitch - dy * 0.005));
      }
      this.dirty = true;
    });

    const end = (e) => {
      if (!pts.delete(e.pointerId)) return;
      if (pts.size >= 2) {
        twoBaseline();
      } else if (pts.size === 1) {
        mode = 'orbit'; pinch = null;   // the remaining finger orbits on
      } else {
        // a TAP (no drag) acts on the content under it: first person walks
        // a careful step toward it, orbit aims at it and pivots on it
        if (!this.lock && mode === 'orbit' && this._down &&
            Math.hypot(e.clientX - this._down.x, e.clientY - this._down.y) < 6 &&
            performance.now() - this._down.t < 500) {
          const r = cv.getBoundingClientRect();
          const px = (e.clientX - r.left) * (this.dpr || 1);
          const py = (e.clientY - r.top) * (this.dpr || 1);
          if (this.fpv) this.stepToward(px, py);
          else this.focusAt(px, py);
        }
        mode = null; pinch = null;
        this.onDragEnd?.();
      }
    };
    cv.addEventListener('pointerup', end);
    cv.addEventListener('pointercancel', end);
    cv.addEventListener('contextmenu', (e) => e.preventDefault());
    cv.addEventListener('wheel', (e) => {
      if (!this.enabled || this.lock) return;   // on a frame the wheel sizes the loupe
      e.preventDefault();
      if (this.fpv) {
        // walk along the view instead of zooming toward the pivot
        this._cancelGlide();
        const { fwd } = this._basis();
        const step = -e.deltaY * 0.0009 * ((this.scene && this.scene.radius) || 10) * 0.07;
        for (let i = 0; i < 3; i++) this.target[i] += fwd[i] * step;
      } else {
        this.dist = Math.max(0.05, Math.min(400, this.dist * Math.exp(e.deltaY * 0.0011)));
      }
      this.dirty = true;
    }, { passive: false });
  }

  resize() {
    const r = this.cv.getBoundingClientRect();
    // 1x CSS pixels: splats are soft content — device-pixel supersampling
    // buys little sharpness and costs phones dearly
    const dpr = 1;
    const w = Math.max(1, Math.round(r.width * dpr)), h = Math.max(1, Math.round(r.height * dpr));
    if (w !== this.cv.width || h !== this.cv.height) { this.cv.width = w; this.cv.height = h; }
    this.w = w; this.h = h; this.dpr = dpr;
    this.dirty = true;
  }

  /** scene = { xyz: Float32Array, rgb: Uint8Array|Float, center, radius } */
  setScene(scene) {
    this.scene = scene;
    this.n = scene.xyz ? scene.xyz.length / 3 : 0;
    this.frameScene();
  }

  /** The world frame out of SfM is arbitrary — "up" is not gravity. The
   *  dominant up over ALL cameras (each one's world up is minus its second
   *  row; photographers hold the camera roughly level on average) becomes
   *  the orbit's axis, so horizons sit balanced. */
  detectUp(cams) {
    const acc = [0, 0, 0];
    let n = 0;
    for (const c of cams) {
      if (!c.R) continue;
      acc[0] -= c.R[3]; acc[1] -= c.R[4]; acc[2] -= c.R[5];
      n++;
    }
    const m = Math.hypot(acc[0], acc[1], acc[2]);
    if (!n || m < 1e-6) return;
    this.setUp([acc[0] / m, acc[1] / m, acc[2] / m]);
  }

  /** set the orbit's up axis and its horizontal reference frame */
  setUp(u) {
    const m = Math.hypot(u[0], u[1], u[2]) || 1;
    this.up = [u[0] / m, u[1] / m, u[2] / m];
    const X = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
    let A = X(this.up, [0, 0, 1]);
    let am = Math.hypot(A[0], A[1], A[2]);
    if (am < 1e-3) { A = X(this.up, [1, 0, 0]); am = Math.hypot(A[0], A[1], A[2]); }
    this._A = [A[0] / am, A[1] / am, A[2] / am];
    this._B = X(this.up, this._A);
  }

  /** yaw/pitch of a unit forward vector in the current up-frame */
  anglesOf(fwd) {
    const u = this.up;
    const d = fwd[0] * u[0] + fwd[1] * u[1] + fwd[2] * u[2];
    const h = [fwd[0] - u[0] * d, fwd[1] - u[1] * d, fwd[2] - u[2] * d];
    const yaw = Math.atan2(
      h[0] * this._A[0] + h[1] * this._A[1] + h[2] * this._A[2],
      h[0] * this._B[0] + h[1] * this._B[1] + h[2] * this._B[2]);
    return { yaw, pitch: Math.asin(Math.max(-1, Math.min(1, d))) };
  }

  frameScene() {
    if (!this.scene) return;
    this.target = [...this.scene.center];
    this.dist = this.scene.radius * 2.5;
    this.yaw = 0.7; this.pitch = -0.32;   // a little above the scene, looking down
    this.dirty = true;
  }

  /** How far away is the stuff this camera looks at? Median depth of the
   *  sparse points inside its frustum — the scene RADIUS is dominated by far
   *  background, which used to push the orbit pivot way past the subject
   *  (small-feeling scenes, wild rotation). */
  _pivotDist(cam) {
    const fallback = (this.scene ? this.scene.radius : 4) * 0.9;
    const s = this.scene;
    if (!s || !s.xyz || !cam.R || !cam.t) return fallback;
    const { xyz } = s;
    const R = cam.R, t = cam.t;
    const inFrust = cam.f && cam.w;
    const zs = [];
    const n = xyz.length / 3;
    const step = Math.max(1, Math.floor(n / 4000));
    for (let i = 0; i < n; i += step) {
      const o = i * 3;
      const X = xyz[o], Y = xyz[o + 1], Z = xyz[o + 2];
      const z = R[6] * X + R[7] * Y + R[8] * Z + t[2];
      if (z <= 0.01) continue;
      if (inFrust) {
        const px = cam.f * (R[0] * X + R[1] * Y + R[2] * Z + t[0]) / z + cam.cx;
        const py = cam.f * (R[3] * X + R[4] * Y + R[5] * Z + t[1]) / z + cam.cy;
        if (px < 0 || py < 0 || px > cam.w || py > cam.h) continue;
      }
      zs.push(z);
    }
    if (zs.length < 20) return fallback;
    zs.sort((a, b) => a - b);
    return Math.min(fallback, zs[zs.length >> 1]);
  }

  /** orbit around a training camera's position without snapping to its pose */
  syncTo(cam) {
    const C = camCentre(cam);
    const fwd = [cam.R[6], cam.R[7], cam.R[8]];
    const d = this._pivotDist(cam);
    this.target = [C[0] + fwd[0] * d, C[1] + fwd[1] * d, C[2] + fwd[2] * d];
    this.dist = d;
    const ang = this.anglesOf(fwd);
    this.yaw = ang.yaw;
    this.pitch = Math.max(-1.45, Math.min(1.45, ang.pitch));
    this.dirty = true;
  }

  /** the free camera's orthonormal frame, built around the measured up axis
   *  (down = fwd x right keeps it a proper rotation, det +1 — negating a
   *  single row would be a reflection and the scene would mirror) */
  _basis() {
    const cp = Math.cos(this.pitch), sp = Math.sin(this.pitch);
    const cy = Math.cos(this.yaw), sy = Math.sin(this.yaw);
    const u = this.up, A = this._A, B = this._B;
    const fwd = [
      A[0] * sy * cp + B[0] * cy * cp + u[0] * sp,
      A[1] * sy * cp + B[1] * cy * cp + u[1] * sp,
      A[2] * sy * cp + B[2] * cy * cp + u[2] * sp,
    ];
    let right = [
      fwd[1] * u[2] - fwd[2] * u[1],
      fwd[2] * u[0] - fwd[0] * u[2],
      fwd[0] * u[1] - fwd[1] * u[0],
    ];
    const rm = Math.hypot(right[0], right[1], right[2]) || 1;
    right = [right[0] / rm, right[1] / rm, right[2] / rm];
    const down = [
      fwd[1] * right[2] - fwd[2] * right[1],
      fwd[2] * right[0] - fwd[0] * right[2],
      fwd[0] * right[1] - fwd[1] * right[0],
    ];
    return { fwd, right, down };
  }

  /** Median depth of sparse-cloud content inside a narrow cone along `dir`
   *  from `eye` — biased toward the NEAR content (the thing being looked at,
   *  not the wall behind it). Null when the cone comes up empty. */
  _contentDepth(eye, dir) {
    const s = this.scene;
    if (!s || !s.xyz || s.xyz.length < 9) return null;
    const xyz = s.xyz, rad = s.radius || 10;
    const ts = [];
    const collect = (tanCone) => {
      ts.length = 0;
      for (let o = 0; o < xyz.length; o += 3) {
        const px = xyz[o] - eye[0], py = xyz[o + 1] - eye[1], pz = xyz[o + 2] - eye[2];
        const t = px * dir[0] + py * dir[1] + pz * dir[2];
        if (t < rad * 0.01) continue;
        const d2 = px * px + py * py + pz * pz - t * t;
        const lim = t * tanCone + rad * 0.004;
        if (d2 < lim * lim) ts.push(t);
      }
    };
    collect(0.10);
    if (ts.length < 6) collect(0.30);
    if (!ts.length) return null;
    ts.sort((a, b) => a - b);
    return ts[Math.floor((ts.length - 1) * 0.35)];
  }

  /** Re-anchor the orbit pivot to the content straight ahead WITHOUT moving
   *  the camera: the target slides along the view ray, dist follows. Deep in
   *  a scene this makes rotation feel first-person; in front of an object it
   *  lands on the object — one rule, no mode switch. */
  anchorPivot() {
    if (this.lock || this.pose) return;
    const { fwd } = this._basis();
    const eye = [
      this.target[0] - fwd[0] * this.dist,
      this.target[1] - fwd[1] * this.dist,
      this.target[2] - fwd[2] * this.dist];
    const rad = (this.scene && this.scene.radius) || 10;
    let d = this._contentDepth(eye, fwd);
    if (d == null) d = this.dist;
    d = Math.max(rad * 0.02, Math.min(rad * 2.5, d));
    this.target = [eye[0] + fwd[0] * d, eye[1] + fwd[1] * d, eye[2] + fwd[2] * d];
    this.dist = d;
    this.dirty = true;
  }

  _cancelGlide() {
    if (this._glide) { cancelAnimationFrame(this._glide.raf); this._glide = null; }
  }

  /** First person: glide a careful step toward the content under a tap —
   *  about 45% of the way there, capped at scene scale, eased over 450ms,
   *  and refused when the nose is already against the wall. */
  stepToward(px, py) {
    if (!this.fpv || this.lock || this.pose) return;
    const { fwd, right, down } = this._basis();
    const eye = [
      this.target[0] - fwd[0] * this.dist,
      this.target[1] - fwd[1] * this.dist,
      this.target[2] - fwd[2] * this.dist];
    const f = this.freeF || Math.min(this.w, this.h) * 0.86;
    const nx = (px - this.w / 2) / f, ny = (py - this.h / 2) / f;
    let dir = [0, 1, 2].map((i) => fwd[i] + right[i] * nx + down[i] * ny);
    const m = Math.hypot(dir[0], dir[1], dir[2]) || 1;
    dir = dir.map((v) => v / m);
    const rad = (this.scene && this.scene.radius) || 10;
    let d = this._contentDepth(eye, dir);
    if (d == null) d = rad * 0.5;
    if (d < rad * 0.04) return;
    const step = Math.min(d * 0.45, rad * 0.3);
    const from = [...this.target];
    const t0 = performance.now();
    this._cancelGlide();
    const tick = () => {
      if (!this._glide) return;
      const u = Math.min(1, (performance.now() - t0) / 450);
      const k = 1 - (1 - u) ** 3;   // ease-out
      this.target = [
        from[0] + dir[0] * step * k,
        from[1] + dir[1] * step * k,
        from[2] + dir[2] * step * k];
      this.dirty = true;
      if (u < 1) this._glide.raf = requestAnimationFrame(tick);
      else this._glide = null;
    };
    this._glide = { raf: requestAnimationFrame(tick) };
  }

  /** Double-click focus: turn toward the content under the pixel and pivot
   *  on it. The camera POSITION stays put — only the aim and pivot move. */
  focusAt(px, py) {
    if (this.lock || this.pose) return;
    const { fwd, right, down } = this._basis();
    const eye = [
      this.target[0] - fwd[0] * this.dist,
      this.target[1] - fwd[1] * this.dist,
      this.target[2] - fwd[2] * this.dist];
    const f = this.freeF || Math.min(this.w, this.h) * 0.86;
    const nx = (px - this.w / 2) / f, ny = (py - this.h / 2) / f;
    let dir = [0, 1, 2].map((i) => fwd[i] + right[i] * nx + down[i] * ny);
    const m = Math.hypot(dir[0], dir[1], dir[2]) || 1;
    dir = dir.map((v) => v / m);
    const rad = (this.scene && this.scene.radius) || 10;
    let d = this._contentDepth(eye, dir);
    if (d == null) return;   // clicked into empty sky — nothing to focus
    d = Math.max(rad * 0.02, Math.min(rad * 2.5, d));
    this.target = [eye[0] + dir[0] * d, eye[1] + dir[1] * d, eye[2] + dir[2] * d];
    const u = this.up, A = this._A, B = this._B;
    const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    this.pitch = Math.asin(Math.max(-1, Math.min(1, dot(dir, u))));
    this.yaw = Math.atan2(dot(dir, A), dot(dir, B));
    this.dist = d;
    this.dirty = true;
  }

  /** current free-orbit pose in the trainer's camera shape */
  freePose() {
    const { fwd, right, down } = this._basis();
    const pos = [
      this.target[0] - fwd[0] * this.dist,
      this.target[1] - fwd[1] * this.dist,
      this.target[2] - fwd[2] * this.dist,
    ];
    const R = [right[0], right[1], right[2], down[0], down[1], down[2], fwd[0], fwd[1], fwd[2]];
    const t = [
      -(R[0] * pos[0] + R[1] * pos[1] + R[2] * pos[2]),
      -(R[3] * pos[0] + R[4] * pos[1] + R[5] * pos[2]),
      -(R[6] * pos[0] + R[7] * pos[1] + R[8] * pos[2]),
    ];
    return {
      R, t, cx: this.w / 2, cy: this.h / 2,
      f: this.freeF || Math.min(this.w, this.h) * 0.86,
    };
  }

  /** the pose the stage renders with (canvas-resolution intrinsics) */
  viewPose() {
    // full pose override (the capture-path tour): true rotation incl. roll,
    // which the yaw/pitch orbit cannot express
    if (this.pose) {
      return {
        R: this.pose.R, t: this.pose.t,
        cx: this.w / 2, cy: this.h / 2,
        f: this.freeF || Math.min(this.w, this.h) * 0.86,
      };
    }
    if (!this.lock) return this.freePose();
    const c = this.lock;
    const s = Math.min(this.w / c.w, this.h / c.h);
    return {
      R: c.R, t: c.t, f: c.f * s,
      cx: this.w / 2 + (c.cx - c.w / 2) * s,
      cy: this.h / 2 + (c.cy - c.h / 2) * s,
    };
  }

  /**
   * @param {object} o
   *   model        true: the trainer's WebGPU canvas sits under this one in
   *                the DOM — clear to transparent so it shows through.
   *                (Never drawImage a WebGPU canvas: iOS Safari can hand back
   *                either of the last two presented frames, which flickers.)
   *   points       true: draw the sparse cloud instead (during the solve)
   *   cams / showCams / reveal / active / sel / skip / faint / dimOthers /
   *   showPath     frustum overlays, as in the mockup
   */
  draw(o = {}) {
    const { ctx, w, h } = this;
    const P = this.viewPose();
    if (o.model) {
      ctx.clearRect(0, 0, w, h);
    } else if (o.points && this.scene && this.scene.xyz) {
      this._points(P);
    } else {
      ctx.fillStyle = '#070909';
      ctx.fillRect(0, 0, w, h);
    }
    // a raw flat [x,y,z,...] cloud (the solve's growing triangulation),
    // drawn under the frustums in a single soft tone
    if (o.cloud && o.cloud.length) {
      const { R, t, f, cx, cy } = P;
      const rgb = o.cloudRgb;   // real sampled colours when the solver sends them
      ctx.fillStyle = 'rgba(47, 212, 193, .8)';
      const s = 2 * (this.dpr || 1);
      for (let i = 0; i < o.cloud.length; i += 3) {
        const X = o.cloud[i], Y = o.cloud[i + 1], Z = o.cloud[i + 2];
        const zc = R[6] * X + R[7] * Y + R[8] * Z + t[2];
        if (zc <= 0.05) continue;
        const px = f * (R[0] * X + R[1] * Y + R[2] * Z + t[0]) / zc + cx;
        const py = f * (R[3] * X + R[4] * Y + R[5] * Z + t[1]) / zc + cy;
        if (px < 0 || py < 0 || px > w || py > h) continue;
        // frames store normalized 0..1 floats — CSS wants 0..255
        if (rgb) ctx.fillStyle = `rgb(${rgb[i] * 255 | 0},${rgb[i + 1] * 255 | 0},${rgb[i + 2] * 255 | 0})`;
        ctx.fillRect(px - s / 2, py - s / 2, s, s);
      }
    }
    if (o.showCams && o.cams) this._drawCams(o, P);
    this.dirty = false;
    return P;
  }

  /** the sparse solve: one dot per triangulated landmark, no model yet */
  _points(P) {
    const { ctx, w, h } = this;
    ctx.fillStyle = '#070909';
    ctx.fillRect(0, 0, w, h);
    const { xyz, rgb } = this.scene;
    const { R, t, f, cx, cy } = P;
    const s = Math.max(1, 1.35 * (this.dpr || 1));
    const step = Math.max(1, Math.floor(this.n / 22000));
    for (let i = 0; i < this.n; i += step) {
      const o3 = i * 3;
      const X = xyz[o3], Y = xyz[o3 + 1], Z = xyz[o3 + 2];
      const zc = R[6] * X + R[7] * Y + R[8] * Z + t[2];
      if (zc <= 0.05) continue;
      const px = f * (R[0] * X + R[1] * Y + R[2] * Z + t[0]) / zc + cx;
      const py = f * (R[3] * X + R[4] * Y + R[5] * Z + t[1]) / zc + cy;
      if (px < 0 || py < 0 || px > w || py > h) continue;
      ctx.fillStyle = `rgb(${rgb[o3]},${rgb[o3 + 1]},${rgb[o3 + 2]})`;
      ctx.fillRect(px - s / 2, py - s / 2, s, s);
    }
  }

  _project(P, X, Y, Z) {
    const { R, t, f, cx, cy } = P;
    const zc = R[6] * X + R[7] * Y + R[8] * Z + t[2];
    if (zc <= 0.02) return null;
    return [
      f * (R[0] * X + R[1] * Y + R[2] * Z + t[0]) / zc + cx,
      f * (R[3] * X + R[4] * Y + R[5] * Z + t[1]) / zc + cy,
    ];
  }

  _drawCams(o, P) {
    const ctx = this.ctx, dpr = this.dpr || 1;
    if (o.faint) ctx.globalAlpha = 0.4;
    const list = o.cams;
    // Pyramid depth follows the capture spacing (median gap between
    // consecutive shots), not the scene radius — radius-scaled frustums
    // overdraw the subject on tight captures. Clamped so sparse arcs don't
    // balloon and dense ones stay visible.
    const rad = this.scene ? this.scene.radius : 4;
    let s = rad * 0.075;
    if (list.length >= 3) {
      if (this._camSizeN !== list.length) {
        const gaps = [];
        let prev = null;
        for (const c of list) {
          if (!c.R) continue;
          const p = camCentre(c);
          if (prev) gaps.push(Math.hypot(p[0] - prev[0], p[1] - prev[1], p[2] - prev[2]));
          prev = p;
        }
        gaps.sort((a, b) => a - b);
        this._camSize = gaps.length ? gaps[gaps.length >> 1] * 0.8 : 0;
        this._camSizeN = list.length;
      }
      if (this._camSize) s = Math.min(s, Math.max(this._camSize, rad * 0.02));
    }
    const upto = o.reveal ?? list.length;
    const path = [];
    const stride = Math.max(1, Math.ceil(list.length / 30));

    for (let i = 0; i < list.length && i < upto; i++) {
      const c = list[i];
      if (!c.R || c.state === 'unplaced' || i === o.skip) continue;
      const keep = i % stride === 0 || i === o.active || i === o.sel || c.state === 'holdout';
      if (!keep) { const p = this._project(P, ...camCentre(c)); if (p) path.push(p); continue; }

      const R = c.R;
      const C = camCentre(c);
      const pc = this._project(P, C[0], C[1], C[2]);
      if (pc) path.push(pc);

      const isActive = i === o.active, isSel = i === o.sel;
      // bright: the solve's register beat — frustums are the subject there,
      // not an overlay on a model
      let col = o.bright ? 'rgba(200,212,210,.8)' : 'rgba(147,161,160,.42)';
      let lw = (o.bright ? 1.3 : 1) * dpr;
      if (c.state === 'holdout') col = 'rgba(242,160,63,.75)';
      if (isSel) { col = 'rgba(230,236,235,.95)'; lw = 1.4 * dpr; }
      if (isActive) { col = 'rgba(47,212,193,1)'; lw = 1.8 * dpr; }
      if (!isActive && !isSel && o.dimOthers) col = 'rgba(147,161,160,.2)';

      // image corners are at pixel (0|w, 0|h); camera-space direction is
      // (px - cx)/f — works for any principal point, not just centred ones
      const k = [[0, 0], [1, 0], [1, 1], [0, 1]].map(([ax, ay]) => {
        const dx = (ax * c.w - c.cx) / c.f * s, dy = (ay * c.h - c.cy) / c.f * s, dz = s;
        return this._project(P,
          C[0] + R[0] * dx + R[3] * dy + R[6] * dz,
          C[1] + R[1] * dx + R[4] * dy + R[7] * dz,
          C[2] + R[2] * dx + R[5] * dy + R[8] * dz);
      });
      if (!pc || k.some((p) => !p)) continue;

      if (isActive) {
        ctx.fillStyle = 'rgba(47,212,193,.13)';
        ctx.beginPath();
        ctx.moveTo(k[0][0], k[0][1]);
        for (let j = 1; j < 4; j++) ctx.lineTo(k[j][0], k[j][1]);
        ctx.closePath(); ctx.fill();
      }
      ctx.strokeStyle = col; ctx.lineWidth = lw;
      ctx.beginPath();
      for (let j = 0; j < 4; j++) {
        ctx.moveTo(pc[0], pc[1]); ctx.lineTo(k[j][0], k[j][1]);
        ctx.moveTo(k[j][0], k[j][1]); ctx.lineTo(k[(j + 1) % 4][0], k[(j + 1) % 4][1]);
      }
      ctx.stroke();
    }

    if (o.showPath && path.length > 1) {
      ctx.strokeStyle = 'rgba(47,212,193,.22)';
      ctx.lineWidth = 1 * dpr;
      ctx.setLineDash([3 * dpr, 4 * dpr]);
      ctx.beginPath();
      path.forEach((p, i) => (i ? ctx.lineTo(p[0], p[1]) : ctx.moveTo(p[0], p[1])));
      ctx.stroke();
      ctx.setLineDash([]);
    }
    ctx.globalAlpha = 1;
  }
}
