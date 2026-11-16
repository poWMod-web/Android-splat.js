// In-place hard prune on a mock GPU: the merge-down's rows take over the
// live trainer with no reseed — the lowMem path where the pixels are gone.
// The strict mock validates every upload's bounds, so an over-wide SH
// write or a patch past the buffers throws here, not on a phone.
import assert from 'node:assert/strict';
import { installWebGpuShims, strictMockGpu } from './mock_gpu.mjs';

installWebGpuShims();
const { GSTrainer } = await import('../../src/gs/trainer.js');
const { initGaussians } = await import('../../src/gs/init.js');

function synthPoints(np) {
  const pts = [];
  for (let i = 0; i < np; i++) {
    pts.push({
      X: [Math.sin(i * 1.7) * 2, Math.cos(i * 2.3), (i % 17) * 0.1 - 0.8],
      rgb: [0.5 + 0.1 * Math.sin(i), 0.5, 0.5],
    });
  }
  return pts;
}

function synthCams(n, w = 64, h = 48) {
  const cams = [];
  for (let i = 0; i < n; i++) {
    cams.push({
      imgIdx: i, R: [1, 0, 0, 0, 1, 0, 0, 0, 1], t: [i * 0.5, 0, 0],
      f: 50, cx: w / 2, cy: h / 2,
    });
  }
  return cams;
}

function synthFrames(n, w = 64, h = 48) {
  const frames = [];
  for (let i = 0; i < n; i++) {
    const rgb = new Float32Array(w * h * 3);
    for (let p = 0; p < w * h; p++) {
      rgb[p * 3] = 0.4 + 0.2 * ((p + i) % 5) / 5;
      rgb[p * 3 + 1] = 0.5; rgb[p * 3 + 2] = 0.6;
    }
    frames.push({ tw: w, th: h, fw: w, fh: h, rgb, alpha: null, sharpness: 100 + i, emptyFrac: 0 });
  }
  return frames;
}

const gpu = strictMockGpu({});
const trainer = await GSTrainer.create({ gpu, maxIters: 100, maxSplats: 20000, capMult: 4, shDeg: 1 });
const model = initGaussians(synthPoints(200), 2);
trainer.setup(model, synthCams(4), synthFrames(4), 256, 144, model.radius);
const n0 = trainer.n;
assert.ok(n0 > 16, 'test needs a model worth pruning');
assert.equal(trainer.shK, 3, 'test needs SH rows (shDeg 1)');

// guards fail loud, before touching anything
assert.throws(() => trainer.applyPrunedModel({ data: model.data, n: 0 }), /out of range/);
assert.throws(() => trainer.applyPrunedModel({ data: model.data, n: n0 + 1 }), /out of range/);
assert.throws(() => trainer.applyPrunedModel({ data: model.data.subarray(0, 16), n: 8 }), /short for the kept rows/);
assert.throws(() => trainer.applyPrunedModel({ data: model.data, n: 8, shK: 0 }), /does not match the trainer/);
assert.equal(trainer.n, n0, 'failed applies leave the count alone');

// the merge-down result: fewer rows, new values
const kept = Math.max(8, Math.floor(n0 / 2));
const data = Float32Array.from(model.data.subarray(0, kept * 16), (v, i) => v + (i % 7) * 0.01);
const sh = Float32Array.from({ length: kept * trainer.shK * 3 }, (_, i) => Math.sin(i * 0.3) * 0.1);
const out = trainer.applyPrunedModel({ data, n: kept, sh, shK: trainer.shK });
assert.deepEqual(out, { kept, n: kept });
assert.equal(trainer.n, kept);
assert.equal(trainer._pruneVotes.length, kept);
assert.ok(trainer._pruneVotes.every((v) => v === 0), 'merged rows start voteless');
assert.equal(trainer._softPruneFloor, kept);
assert.equal(trainer.adamData[23], kept * 16);
assert.equal(trainer._lastReloc, null);
assert.ok(trainer.camUniforms.length > 0, 'cam uniforms rebuilt');

console.log('test_prune_inplace: OK');
