// chart.js — how the score develops. Two lines (trained-on photo, hidden photo),
// the moments where the optimiser changed its own setup, and an optional ghost
// run so "what if" is a comparison instead of a claim.

/** Approximate SSIM percentage from PSNR (in dB), per the app's calibration. */
export function psnrToApproxSsimPercent(psnr) {
  if (psnr === Infinity) return 100;
  if (!Number.isFinite(psnr) || psnr <= 0) return 0;
  const ratio = 18 / psnr;
  return Math.max(0, Math.min(100, 100 / (1 + ratio ** 4)));
}

/** Samples held from the START of each drawn line: nothing plots until
 *  the series runs past these — the first samples climb almost vertically
 *  off a cold start (large, disoriented splats) and would pin the floor
 *  far below the working range. The hold is up front, not a trim after
 *  the fact: short series draw nothing at all, so early points never
 *  appear and then vanish. Raw samples stay stored regardless. */
const SKIP_FIRST = 3;

/** Zero-lag exponential moving average with a Z-score spike clamp: each
 *  sample is clamped against the running level (beyond zThresh standard
 *  deviations of the tracked variance it counts as a spike, not signal),
 *  then smoothed with the lag taken back out (2*clean - lagged). Fast and
 *  allocation-free past construction: the lag window is a power-of-2 ring
 *  buffer with bitwise wrap. First sample latches the level.
 *  series() runs a fresh pass, so redraws and scale toggles can't desync
 *  the curve. Operates in raw dB; the caller converts. */
export class FastRobustZLEMA {
  /**
   * @param {number} period   - Equivalent smoothing window (e.g., 16 to 32)
   * @param {number} zThresh  - Gaussian tail cutoff in std devs (e.g., 2.2 to 2.5)
   * @param {number} minStd   - Minimum noise floor (~50% of expected Gaussian sigma)
   */
  constructor(period = 20, zThresh = 2.3, minStd = 0.0) {
    const p = Math.max(2, period | 0);
    this.lag = (p - 1) >> 1;
    // Standard EMA alpha for period N: 2 / (N + 1)
    this.alpha = 2.0 / (p + 1);
    this.oneMinusAlpha = 1.0 - this.alpha;
    // Next power of 2 for fast bitwise wrap-around (& mask instead of %)
    let cap = 1;
    while (cap <= this.lag) cap <<= 1;
    this.mask = cap - 1;
    this.buf = new Float64Array(cap);
    this.zThreshSq = zThresh * zThresh;
    this.minVariance = minStd * minStd;
    this.gamma = 0.05;
    this.oneMinusGamma = 0.95;
    this.writeIdx = 0;
    this.initialized = false;
    this.zlema = 0.0;
    this.variance = 0.0;
  }
  update(x) {
    if (!this.initialized) {
      this.buf.fill(x);
      this.writeIdx = 0;
      this.zlema = x;
      this.variance = this.minVariance > 1e-3 ? this.minVariance : 1e-3;
      this.initialized = true;
      return x;
    }
    // 1. Clamp extreme Gaussian tail spikes against the current ZLEMA center
    let cleanX = x;
    const err = x - this.zlema;
    let errSq = err * err;
    const maxErrSq = this.zThreshSq * this.variance;
    if (errSq > maxErrSq) {
      const limit = Math.sqrt(maxErrSq);
      cleanX = this.zlema + (err > 0 ? limit : -limit);
      errSq = maxErrSq;
    }
    const nextVar = this.oneMinusGamma * this.variance + this.gamma * errSq;
    this.variance = nextVar > this.minVariance ? nextVar : this.minVariance;
    // 2. Read lagged clean sample from power-of-2 ring buffer
    const laggedIdx = (this.writeIdx - this.lag) & this.mask;
    const laggedX = this.buf[laggedIdx];
    // Store current clean sample in ring buffer
    this.buf[this.writeIdx] = cleanX;
    this.writeIdx = (this.writeIdx + 1) & this.mask;
    // 3. De-lagged input: cleanX + (cleanX - laggedX) = 2 * cleanX - laggedX
    const deLagged = 2.0 * cleanX - laggedX;
    // 4. Single EMA update (zero division, zero float drift)
    this.zlema = this.alpha * deLagged + this.oneMinusAlpha * this.zlema;
    return this.zlema;
  }
  reset() {
    this.initialized = false;
    this.writeIdx = 0;
  }
  series(pts) {
    this.reset();
    return pts.map(([iter, v]) => [iter, this.update(v)]);
  }
}

export class Chart {
  constructor(canvas, { onHover, onScrub } = {}) {
    this.cv = canvas;
    this.ctx = canvas.getContext('2d');
    this.scale = 'db';
    this.train = []; this.hold = [];
    this.ghost = null; this.events = []; this.maxIter = 40000;
    this.onHover = onHover; this.onScrub = onScrub;
    this.hover = null; this.scrubbable = false; this.mark = null;

    const pos = (e) => {
      const r = this.cv.getBoundingClientRect();
      return { x: (e.clientX - r.left) / r.width, y: (e.clientY - r.top) / r.height };
    };
    canvas.addEventListener('pointermove', (e) => {
      const p = pos(e);
      this.hover = p;
      if (this._drag && this.onScrub) this.onScrub(this._frac(p.x));
      this.draw();
    });
    canvas.addEventListener('pointerleave', () => { this.hover = null; this.onHover?.(null); this.draw(); });
    canvas.addEventListener('pointerdown', (e) => {
      if (!this.scrubbable) return;
      this._drag = true; canvas.setPointerCapture(e.pointerId);
      this.onScrub?.(this._frac(pos(e).x));
    });
    canvas.addEventListener('pointerup', () => { this._drag = false; });
  }

  _frac(xf) {
    const { l, r } = this._box();
    const w = this.cv.width;
    return Math.max(0, Math.min(1, (xf * w - l) / (w - l - r)));
  }

  _displayValue(value) {
    return this.scale === 'similarity' ? psnrToApproxSsimPercent(value) : value;
  }

  _activeSeries() {
    // The approximate SSIM scale is derived from the same evaluated PSNR
    // samples; there is no second model evaluation or synthetic history.
    return { train: this.train, hold: this.hold };
  }

  /** Raw post-hold samples (still in dB): the dots' material and the
   *  hover's lookup — never the held-back cold start. */
  _heldRaw(series) {
    return series.length > SKIP_FIRST ? series.slice(SKIP_FIRST) : [];
  }

  /** Raw-dB ZLEMA pass over the post-hold samples: the curve's material.
   *  Fresh pass per call — redraws and scale toggles can't desync it. */
  _filteredRaw(series) {
    const held = this._heldRaw(series);
    if (!held.length) return [];
    return new FastRobustZLEMA().series(held);
  }

  _displaySeries(series) {
    return this._filteredRaw(series)
      .map(([iter, value]) => [iter, this._displayValue(value)]);
  }

  /** Dynamic x extent for the given last-sample iters: the data's reach
   *  plus a breath of headroom, never past the run's horizon. */
  _xMaxFor(lastIters) {
    const m = Math.max(0, ...lastIters);
    return Math.min(this.maxIter, Math.max(1, m * 1.06, m + 1));
  }

  // top just fits the event dots; there are no x labels, so the bottom is
  // nearly flush — the plot gets the height
  _box() { return { l: 8 * this.dpr, r: 48 * this.dpr, t: 8 * this.dpr, b: 4 * this.dpr }; }

  resize() {
    const rect = this.cv.getBoundingClientRect();
    const dpr = Math.min(2, devicePixelRatio || 1);
    this.cv.width = Math.max(1, rect.width * dpr);
    this.cv.height = Math.max(1, rect.height * dpr);
    this.dpr = dpr;
    // any layout shift (window resize, panel toggle, chart expand) leaves
    // the pointer over a different fraction — drop the stale hover so the
    // readout cannot re-fire detached from the cursor on the next redraw
    this.clearHover();
  }

  clearHover() {
    this.hover = null;
    this.onHover?.(null);
    this.draw();
  }

  push(iter, train, hold) {
    this.train.push([iter, train]);
    if (hold != null) this.hold.push([iter, hold]);
  }

  reset() { this.train = []; this.hold = []; }

  draw() {
    const ctx = this.ctx, W = this.cv.width, H = this.cv.height, dpr = this.dpr || 1;
    const { l, r, t, b } = this._box();
    const pw = W - l - r, ph = H - t - b;
    ctx.clearRect(0, 0, W, H);
    if (pw <= 0 || ph <= 0) return;

    const active = this._activeSeries();
    const trainHeld = this._heldRaw(active.train);
    const holdHeld = this._heldRaw(active.hold);
    const trainF = this._filteredRaw(active.train);
    const holdF = this._filteredRaw(active.hold);
    // the score readouts show the ZLEMA curve's end, not the latest raw
    // sample — one source of truth for the number and the line
    this.lastTrainF = trainF.length ? trainF[trainF.length - 1][1] : null;
    this.lastHoldF = holdF.length ? holdF[holdF.length - 1][1] : null;
    const train = trainF.map(([iter, value]) => [iter, this._displayValue(value)]);
    const hold = holdF.map(([iter, value]) => [iter, this._displayValue(value)]);
    const ghostTrain = this._displaySeries(this.ghost?.train || []);
    const ghostHold = this._displaySeries(this.ghost?.hold || []);
    // the raw dots behind the filtered lines: tiny and light — the honest
    // scatter the curve summarizes (ghost stays line-only)
    const trainDots = trainHeld.map(([iter, value]) => [iter, this._displayValue(value)]);
    const holdDots = holdHeld.map(([iter, value]) => [iter, this._displayValue(value)]);
    const ys = [...train, ...hold, ...ghostTrain, ...ghostHold, ...trainDots, ...holdDots].map((p) => p[1]);
    let lo = ys.length ? Math.min(...ys) : (this.scale === 'similarity' ? 0 : 10);
    let hi = ys.length ? Math.max(...ys) : (this.scale === 'similarity' ? 100 : 30);
    const minSpan = this.scale === 'similarity' ? 1 : 6;
    if (hi - lo < minSpan) { const m = (hi + lo) / 2; lo = m - minSpan / 2; hi = m + minSpan / 2; }
    // hug the data: the first sample sits a hair off the floor, so the climb
    // uses the full plot height instead of floating on rounded padding
    const pad = (hi - lo) * .03;
    lo -= pad; hi += pad;
    if (this.scale === 'similarity') { lo = Math.max(0, lo); hi = Math.min(100, hi); }

    // dynamic x: the axis grows with the plotted data instead of reserving
    // the whole horizon up front — early curves use the full width, the
    // way the y-axis already hugs the data
    const lastIt = (arr) => (arr.length ? arr[arr.length - 1][0] : 0);
    const xMax = this._xMaxFor([lastIt(active.train), lastIt(active.hold),
      lastIt(this.ghost?.train || []), lastIt(this.ghost?.hold || [])]);
    const X = (i) => l + (i / xMax) * pw;
    const Y = (v) => t + ph - ((v - lo) / (hi - lo)) * ph;

    // faint progress wash left of the latest sample — the chart quietly
    // doubles as a progress bar without competing with the curve
    if (active.train.length) {
      // the wash keeps progress meaning (fraction of the horizon) while
      // the curve above it enjoys the dynamic axis
      const frac = Math.min(1, active.train[active.train.length - 1][0] / this.maxIter);
      ctx.fillStyle = 'rgba(94, 231, 214, .055)';
      ctx.fillRect(l, t, Math.max(0, frac * pw), ph);
    }

    // grid
    ctx.font = `400 ${11 * dpr}px "Spline Sans Mono", monospace`;
    ctx.textBaseline = 'middle';
    const span = hi - lo;
    const step = this.scale === 'similarity'
      ? ([0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10].find((s) => span / s <= 5) || 10)
      : (span > 24 ? 10 : span > 12 ? 5 : 2);
    const precision = this.scale === 'similarity' ? (step < 0.1 ? 2 : step < 1 ? 1 : 0) : 0;
    for (let v = Math.ceil(lo / step) * step; v <= hi + step * 1e-6; v += step) {
      const y = Y(v);
      ctx.strokeStyle = 'rgba(59,69,71,.5)'; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(l, y); ctx.lineTo(l + pw, y); ctx.stroke();
      ctx.fillStyle = '#6b7877'; ctx.textAlign = 'left';
      ctx.fillText(this.scale === 'similarity' ? `${v.toFixed(precision)}%` : `${v} dB`, l + pw + 6 * dpr, y);
    }

    // events
    for (const e of this.events) {
      const x = X(e.at * this.maxIter);
      if (x > l + pw) continue;
      ctx.strokeStyle = e.kind === 'grow' ? 'rgba(242,160,63,.28)' : 'rgba(147,161,160,.24)';
      ctx.setLineDash([2 * dpr, 3 * dpr]); ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(x, t); ctx.lineTo(x, t + ph); ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = e.kind === 'grow' ? 'rgba(242,160,63,.75)' : 'rgba(147,161,160,.6)';
      ctx.beginPath(); ctx.arc(x, t - 4 * dpr, 2 * dpr, 0, 7); ctx.fill();
    }

    const line = (pts, color, dash, width) => {
      if (pts.length < 2) return;
      ctx.strokeStyle = color; ctx.lineWidth = (width || 1.6) * dpr;
      ctx.setLineDash(dash ? dash.map((d) => d * dpr) : []);
      ctx.lineJoin = 'round';
      ctx.beginPath();
      pts.forEach((p, i) => (i ? ctx.lineTo(X(p[0]), Y(p[1])) : ctx.moveTo(X(p[0]), Y(p[1]))));
      ctx.stroke();
      ctx.setLineDash([]);
    };

    if (this.ghost) {
      line(ghostTrain, 'rgba(147,161,160,.35)', [4, 3], 1.2);
      line(ghostHold, 'rgba(147,161,160,.22)', [2, 3], 1.2);
    }
    const dots = (pts, color) => {
      if (!pts.length) return;
      ctx.fillStyle = color;
      for (const p of pts) {
        ctx.beginPath(); ctx.arc(X(p[0]), Y(p[1]), 0.75 * dpr, 0, 7); ctx.fill();
      }
    };
    dots(holdDots, 'rgba(242,160,63,.35)');
    dots(trainDots, 'rgba(47,212,193,.30)');
    line(hold, '#f2a03f', [5, 3], 1.5);
    line(train, '#2fd4c1', null, 1.8);

    // playhead
    const last = active.train[active.train.length - 1];
    if (last) {
      const x = X(this.mark != null ? this.mark : last[0]);
      ctx.strokeStyle = 'rgba(230,236,235,.25)'; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(x, t); ctx.lineTo(x, t + ph); ctx.stroke();
      // the dot rides the drawn (trimmed, smoothed) line's end, not the raw
      // sample — else it floats off the curve it terminates
      const drawn = train[train.length - 1];
      if (this.mark == null && drawn) {
        ctx.fillStyle = '#2fd4c1';
        ctx.beginPath(); ctx.arc(X(drawn[0]), Y(drawn[1]), 3 * dpr, 0, 7); ctx.fill();
      }
    }

    // hover readout
    if (this.hover && trainHeld.length) {
      const it = this._frac(this.hover.x) * xMax;
      const near = (arr) => arr.length ? arr.reduce((a, p) => Math.abs(p[0] - it) < Math.abs(a[0] - it) ? p : a) : null;
      const a = near(trainHeld), h = near(holdHeld);
      const ev = this.events.find((e) => Math.abs(e.at * this.maxIter - it) < xMax * .012);
      if (a) {
        const x = X(a[0]);
        ctx.strokeStyle = 'rgba(230,236,235,.4)'; ctx.lineWidth = 1;
        ctx.beginPath(); ctx.moveTo(x, t); ctx.lineTo(x, t + ph); ctx.stroke();
        this.onHover?.({
          xPct: (x / W) * 100,
          // where the pointer sits, so the readout can ride the cursor
          // inside the plot instead of hanging off its top edge (the
          // expanded graph fills the stage — anything above it is clipped
          // behind the dock and unreadable)
          yPct: this.hover.y * 100,
          iter: Math.round(a[0]),
          train: this._displayValue(a[1]), hold: h ? this._displayValue(h[1]) : null,
          unit: this.scale === 'similarity' ? '%' : 'dB', event: ev ? ev.label : null,
        });
      }
    }
  }
}
