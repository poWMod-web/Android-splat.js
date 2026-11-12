// zlema.js — the score smoother. One class, two readers: the score
// graph draws its output (see chart.js) and the refine trigger tests its
// slope for plateaus (see session.js) — same parameters, same cold-start
// hold, so the line on screen and the slope that pulls refines forward
// can never disagree. Pure math, no DOM: safe for Node tests.

/** Samples held from the START of each drawn line: nothing plots until
 *  the series runs past these — the first samples climb almost vertically
 *  off a cold start (large, disoriented splats) and would pin the floor
 *  far below the working range. The hold is up front, not a trim after
 *  the fact: short series draw nothing at all, so early points never
 *  appear and then vanish. Raw samples stay stored regardless. */
export const SKIP_FIRST = 3;

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

