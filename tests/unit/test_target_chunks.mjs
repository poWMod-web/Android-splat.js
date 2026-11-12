import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { planTargetChunks } from '../../src/gs/trainer.js';

const trainerSource = readFileSync(new URL('../../src/gs/trainer.js', import.meta.url), 'utf8');

// (a) a set that fits stays one chunk covering every camera
{
  const sizes = [100, 200, 300];
  const chunks = planTargetChunks(sizes, 4000);
  assert.deepEqual(chunks, [{ start: 0, count: 3, px: 600 }]);
}

// (a) chunks partition whole cameras in order, each under the binding ceiling
{
  const MB = 1e6 / 4; // pixels per MB at RGBA8
  const sizes = [50 * MB, 50 * MB, 50 * MB, 50 * MB, 10 * MB]; // 210 MB total
  const chunks = planTargetChunks(sizes, 128e6);
  assert.ok(chunks.length === 2, `210MB over 128MB should split into 2 chunks, got ${chunks.length}`);
  let cursor = 0, pxSum = 0;
  for (const ch of chunks) {
    assert.equal(ch.start, cursor, 'chunks must tile cameras contiguously from 0');
    assert.ok(ch.px * 4 <= 128e6, `chunk of ${ch.px * 4 / 1e6}MB exceeds the 128MB ceiling`);
    const expect = sizes.slice(ch.start, ch.start + ch.count).reduce((a, b) => a + b, 0);
    assert.equal(ch.px, expect, 'chunk px must equal its cameras\u2019 pixel sum');
    cursor += ch.count; pxSum += ch.px;
  }
  assert.equal(cursor, sizes.length, 'every camera must land in exactly one chunk');
  assert.equal(pxSum, sizes.reduce((a, b) => a + b, 0), 'no pixels lost or duplicated');
}

// (a) an exact fit at the ceiling stays one chunk (no off-by-one split)
{
  const chunks = planTargetChunks([250, 250], 2000);
  assert.deepEqual(chunks, [{ start: 0, count: 2, px: 500 }]);
}

// (a) one image over the ceiling throws and names training resolution (nothing to chunk)
{
  assert.throws(() => planTargetChunks([10, 4000], 8000), /exceeds the device binding limit/,
    'a single over-ceiling image must throw');
  assert.throws(() => planTargetChunks([4000], 8000), /reduce training resolution/,
    'the message must point at training resolution, not image count');
}

// (a) empty set yields one empty chunk, never a throw (degenerate setup stays constructible)
assert.deepEqual(planTargetChunks([], 128e6), [{ start: 0, count: 0, px: 0 }]);

// (a) setup wiring: camMeta carries chunk id + chunk-local offset, planner drives it
assert.match(trainerSource, /const chunkPlan = planTargetChunks\(pxPerCam, limit\)/,
  'setup must plan target chunks from the device binding limit');
assert.match(trainerSource, /offset: local, tgtChunk: k/,
  'camMeta must carry the chunk id and the chunk-local pixel offset');
assert.match(trainerSource, /this\.tgtChunks = targetChunks\.map/,
  'setup must upload one target buffer per chunk');
assert.match(trainerSource, /this\.bgRenderTrainCR = this\.uniTrainR\.map\(\(u\) => this\.tgtChunks\.map\(\(t\) => bgRender\(u, t\)\)\);/,
  'train bindings must be built per ring slot per chunk');
assert.match(trainerSource, /const tc = meta\.tgtChunk \?\? 0;/,
  'encodeRaster must select the train binding from the camera\u2019s chunk');
assert.match(trainerSource, /this\.bgRenderTrainC\[tc\]/,
  'the plain train path must bind the camera\u2019s chunk');
assert.doesNotMatch(trainerSource, /reduce image count or resolution/,
  'the old fail-the-whole-set error must be gone: chunking replaces it');

// (b) SSIM metric GPU objects are lazy: no eager creation, ensure-before-use
assert.match(trainerSource, /_ensureSsimMetric\(\) \{/,
  'the lazy SSIM-metric builder must exist');
assert.match(trainerSource, /if \(this\.bgSsimMetricC\) return;/,
  'the lazy builder must be idempotent');
assert.match(trainerSource, /this\._ensureSsimMetric\(\);\n      p\.setPipeline\(this\.pipeSsimMetric\)/,
  '_evalPass must ensure the metric objects before dispatching the metric pass');
assert.match(trainerSource, /this\.bgSsimMetricC\[meta\.tgtChunk \?\? 0\]/,
  'the metric pass must bind the evaluated camera\u2019s chunk');
assert.doesNotMatch(trainerSource, /this\.bufSsimMetric = buf\(/,
  'the metric storage buffer must not be created eagerly in setup');
assert.doesNotMatch(trainerSource, /this\.bufSsimMetricRead = buf\(/,
  'the metric readback buffer must not be created eagerly in setup');

// (c) densification reuses pooled buffers: no per-refine staging allocation
assert.match(trainerSource, /enc\.copyBufferToBuffer\(this\.bufParams, 0, this\.bufParamsRead, 0, liveBytes\)/,
  'the refine readback must reuse the cap-sized pooled buffer');
assert.match(trainerSource, /params\.set\(new Float32Array\(this\.bufParamsRead\.getMappedRange\(0, liveBytes\)\)\)/,
  'the refine readback must map only the live range');
assert.match(trainerSource, /if \(!this\.bufTileCntRead\) \{/,
  'the refine telemetry readback must reuse a pooled staging buffer');
assert.doesNotMatch(trainerSource, /const readBuf = async/,
  'the per-refine transient readback helper must be gone');

console.log('target chunking, lazy SSIM metric, and pooled refine readback all verified');
