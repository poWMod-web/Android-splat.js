// chart.js — how the score develops. Two lines (trained-on photo, hidden photo),
// the moments where the optimiser changed its own setup, and an optional ghost
// run so "what if" is a comparison instead of a claim.

export function ssimToPercent(ssim) {
  if (!Number.isFinite(ssim)) return 0;
  return Math.max(0, Math.min(100, ssim * 100));
}

export class Chart {
  constructor(canvas, { onHover, onScrub } = {}) {
    this.cv = canvas;
    this.ctx = canvas.getContext('2d');
    this.scale = 'db';
    this.train = []; this.hold = [];
    this.ssimTrain = []; this.ssimHold = [];
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
    return this.scale === 'similarity' ? ssimToPercent(value) : value;
  }

  _activeSeries() {
    return this.scale === 'similarity'
      ? { train: this.ssimTrain, hold: this.ssimHold }
      : { train: this.train, hold: this.hold };
  }

  _displaySeries(series) {
    return series.map(([iter, value]) => [iter, this._displayValue(value)]);
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
    this.draw();
  }

  push(iter, train, hold) {
    this.train.push([iter, train]);
    if (hold != null) this.hold.push([iter, hold]);
  }

  reset() { this.train = []; this.hold = []; this.ssimTrain = []; this.ssimHold = []; }

  draw() {
    const ctx = this.ctx, W = this.cv.width, H = this.cv.height, dpr = this.dpr || 1;
    const { l, r, t, b } = this._box();
    const pw = W - l - r, ph = H - t - b;
    ctx.clearRect(0, 0, W, H);
    if (pw <= 0 || ph <= 0) return;

    const active = this._activeSeries();
    const train = this._displaySeries(active.train);
    const hold = this._displaySeries(active.hold);
    const ghostTrain = this._displaySeries(this.scale === 'similarity'
      ? (this.ghost?.ssimTrain || []) : (this.ghost?.train || []));
    const ghostHold = this._displaySeries(this.scale === 'similarity'
      ? (this.ghost?.ssimHold || []) : (this.ghost?.hold || []));
    const ys = [...train, ...hold, ...ghostTrain, ...ghostHold].map((p) => p[1]);
    let lo = ys.length ? Math.min(...ys) : (this.scale === 'similarity' ? 90 : 10);
    let hi = ys.length ? Math.max(...ys) : (this.scale === 'similarity' ? 100 : 30);
    const minSpan = this.scale === 'similarity' ? 0.2 : 6;
    if (hi - lo < minSpan) { const m = (hi + lo) / 2; lo = m - minSpan / 2; hi = m + minSpan / 2; }
    // hug the data: the first sample sits a hair off the floor, so the climb
    // uses the full plot height instead of floating on rounded padding
    const pad = (hi - lo) * .03;
    lo -= pad; hi += pad;
    if (this.scale === 'similarity') { lo = Math.max(0, lo); hi = Math.min(100, hi); }

    const X = (i) => l + (i / this.maxIter) * pw;
    const Y = (v) => t + ph - ((v - lo) / (hi - lo)) * ph;

    // faint progress wash left of the latest sample — the chart quietly
    // doubles as a progress bar without competing with the curve
    if (active.train.length) {
      const px = X(active.train[active.train.length - 1][0]);
      ctx.fillStyle = 'rgba(94, 231, 214, .055)';
      ctx.fillRect(l, t, Math.max(0, Math.min(px, l + pw) - l), ph);
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
    line(hold, '#f2a03f', [5, 3], 1.5);
    line(train, '#2fd4c1', null, 1.8);

    // playhead
    const last = active.train[active.train.length - 1];
    if (last) {
      const x = X(this.mark != null ? this.mark : last[0]);
      ctx.strokeStyle = 'rgba(230,236,235,.25)'; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(x, t); ctx.lineTo(x, t + ph); ctx.stroke();
      if (this.mark == null) {
        ctx.fillStyle = '#2fd4c1';
        ctx.beginPath(); ctx.arc(X(last[0]), Y(this._displayValue(last[1])), 3 * dpr, 0, 7); ctx.fill();
      }
    }

    // hover readout
    if (this.hover && active.train.length) {
      const it = this._frac(this.hover.x) * this.maxIter;
      const near = (arr) => arr.length ? arr.reduce((a, p) => Math.abs(p[0] - it) < Math.abs(a[0] - it) ? p : a) : null;
      const a = near(active.train), h = near(active.hold);
      const ev = this.events.find((e) => Math.abs(e.at * this.maxIter - it) < this.maxIter * .012);
      if (a) {
        const x = X(a[0]);
        ctx.strokeStyle = 'rgba(230,236,235,.4)'; ctx.lineWidth = 1;
        ctx.beginPath(); ctx.moveTo(x, t); ctx.lineTo(x, t + ph); ctx.stroke();
        this.onHover?.({
          xPct: (x / W) * 100,
          iter: Math.round(a[0]),
          train: this._displayValue(a[1]), hold: h ? this._displayValue(h[1]) : null,
          unit: this.scale === 'similarity' ? 'SSIM %' : 'dB', event: ev ? ev.label : null,
        });
      }
    }
  }
}
