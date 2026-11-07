// p3pBearings: central-camera P3P on unit bearing vectors (Grunert quartic +
// Horn absolute orientation). Validated against random synthetic poses with
// bearings in ALL directions — the omnidirectional case pinhole PnP can't do.
import { p3pBearings, rodrigues, makeRng } from '../../src/sfm/geometry.js';

let failures = 0;
const check = (name, cond, detail = '') => {
  console.log(`${cond ? 'PASS' : 'FAIL'} ${name} ${detail}`);
  if (!cond) failures++;
};

const rng = makeRng(4242);

const m3mulvT = (R, v) => [   // R * v, row-major
  R[0] * v[0] + R[1] * v[1] + R[2] * v[2],
  R[3] * v[0] + R[4] * v[1] + R[5] * v[2],
  R[6] * v[0] + R[7] * v[1] + R[8] * v[2],
];

// exact recovery over random configurations
{
  let worstR = 0, worstC = 0, solved = 0, trials = 500;
  for (let t = 0; t < trials; t++) {
    const Rgt = rodrigues([(rng() - 0.5) * 6, (rng() - 0.5) * 6, (rng() - 0.5) * 6]);
    const Cgt = [(rng() - 0.5) * 10, (rng() - 0.5) * 10, (rng() - 0.5) * 10];
    const P = [], brg = [];
    for (let i = 0; i < 3; i++) {
      // world points anywhere around the centre — bearings span the sphere
      const X = [Cgt[0] + (rng() - 0.5) * 20, Cgt[1] + (rng() - 0.5) * 20, Cgt[2] + (rng() - 0.5) * 20];
      const d = [X[0] - Cgt[0], X[1] - Cgt[1], X[2] - Cgt[2]];
      const len = Math.hypot(...d);
      if (len < 0.5) { i--; continue; }
      const v = m3mulvT(Rgt, d);
      brg.push([v[0] / len, v[1] / len, v[2] / len]);
      P.push(X);
    }
    const cands = p3pBearings(P, brg);
    let bestR = Infinity, bestC = Infinity;
    for (const c of cands) {
      let eR = 0;
      for (let k = 0; k < 9; k++) eR = Math.max(eR, Math.abs(c.R[k] - Rgt[k]));
      const eC = Math.hypot(c.C[0] - Cgt[0], c.C[1] - Cgt[1], c.C[2] - Cgt[2]);
      if (eR + eC < bestR + bestC) { bestR = eR; bestC = eC; }
    }
    if (bestR < 1e-5 && bestC < 1e-4) solved++;
    if (bestR < Infinity) { worstR = Math.max(worstR, Math.min(bestR, 1)); worstC = Math.max(worstC, Math.min(bestC, 1)); }
  }
  // Grunert can be numerically delicate near degenerate triangles; for RANSAC
  // use it just needs to recover the pose on the vast majority of samples
  check('p3p random recovery >= 97%', solved >= trials * 0.97, `${solved}/${trials}`);
}

// behind-the-camera bearings specifically (vz < 0 for all three)
{
  let solved = 0, trials = 200;
  for (let t = 0; t < trials; t++) {
    const Rgt = rodrigues([(rng() - 0.5) * 6, (rng() - 0.5) * 6, (rng() - 0.5) * 6]);
    const Cgt = [rng() * 4, rng() * 4, rng() * 4];
    const P = [], brg = [];
    for (let i = 0; i < 3; i++) {
      const X = [Cgt[0] + (rng() - 0.5) * 20, Cgt[1] + (rng() - 0.5) * 20, Cgt[2] + (rng() - 0.5) * 20];
      const d = [X[0] - Cgt[0], X[1] - Cgt[1], X[2] - Cgt[2]];
      const len = Math.hypot(...d);
      const v = m3mulvT(Rgt, d);
      if (len < 0.5 || v[2] > -0.1 * len) { i--; continue; }   // strictly behind
      brg.push([v[0] / len, v[1] / len, v[2] / len]);
      P.push(X);
    }
    const cands = p3pBearings(P, brg);
    for (const c of cands) {
      let eR = 0;
      for (let k = 0; k < 9; k++) eR = Math.max(eR, Math.abs(c.R[k] - Rgt[k]));
      const eC = Math.hypot(c.C[0] - Cgt[0], c.C[1] - Cgt[1], c.C[2] - Cgt[2]);
      if (eR < 1e-5 && eC < 1e-4) { solved++; break; }
    }
  }
  check('p3p behind-camera recovery >= 97%', solved >= trials * 0.97, `${solved}/${trials}`);
}

// degenerate: collinear world points must return no candidates, not garbage
{
  const P = [[0, 0, 0], [1, 1, 1], [2, 2, 2]];
  const brg = [[0, 0, 1], [0.1, 0, 0.99], [0.2, 0, 0.98]].map((v) => {
    const l = Math.hypot(...v); return [v[0] / l, v[1] / l, v[2] / l];
  });
  check('p3p collinear -> []', p3pBearings(P, brg).length === 0);
}

console.log(failures ? `${failures} failures` : 'p3p ok');
process.exit(failures ? 1 : 0);
