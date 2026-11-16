import assert from 'node:assert/strict';
import { floatToHalf, halfToFloat, packState, parseState } from '../../app/js/session_io.js';

// ---- scalar codec: known answers -------------------------------------------
assert.equal(floatToHalf(1), 0x3c00);
assert.equal(floatToHalf(-2), 0xc000);
assert.equal(floatToHalf(0.5), 0x3800);
assert.equal(floatToHalf(0), 0x0000);
assert.equal(floatToHalf(-0), 0x8000);
assert.equal(halfToFloat(0x3c00), 1);
assert.equal(halfToFloat(0x7c00), Infinity);
assert.equal(halfToFloat(0xfc00), -Infinity);
assert.ok(Number.isNaN(halfToFloat(0x7e00)));
assert.ok(Object.is(halfToFloat(floatToHalf(-0)), -0));
assert.ok(Number.isNaN(halfToFloat(floatToHalf(NaN))));
// mantissa round-up carries into the exponent even when e is odd
// (regression: OR-ing swallowed the carry and decoded 0.49999… as 0.25)
assert.equal(floatToHalf(0.4999997913837433), 0x3800);
assert.equal(floatToHalf(3.9999), 0x4400);   // 3.9999 rounds up to 4.0
assert.equal(halfToFloat(floatToHalf(0.4999997913837433)), 0.5);
// overflow saturates to infinity, tiny values flush toward zero
assert.equal(floatToHalf(1e5), 0x7c00);   // finite-but-huge keeps no mantissa bits
assert.equal(halfToFloat(floatToHalf(1e5)), Infinity);
assert.equal(halfToFloat(floatToHalf(-1e5)), -Infinity);
assert.equal(halfToFloat(floatToHalf(1e-9)), 0);
assert.ok(Math.abs(halfToFloat(floatToHalf(6e-8))) < 1e-6);
// the largest normal half round-trips exactly
assert.equal(halfToFloat(floatToHalf(65504)), 65504);
assert.ok(Math.abs(halfToFloat(floatToHalf(60000)) - 60000) < 60);
// round-trip sweep: fp16 carries ~3 decimal digits
for (let e = -4; e <= 3; e++) {
  // near-powers-of-two round their mantissa UP across every exponent parity
  for (const m of [1, 2.5, 7.75, -0.5, -9.25, 1.9999, 0.99999]) {
    const v = m * 10 ** e;
    const rt = halfToFloat(floatToHalf(v));
    assert.ok(Math.abs(rt - v) <= 2e-3 * Math.abs(v) + 1e-6, `fp16 round-trip of ${v} got ${rt}`);
  }
}

// ---- v2 blob: half the bytes, float16-close ---------------------------------
const n = 5, shK = 3;
const data = new Float32Array(n * 16).map((_, i) => Math.sin(i * 0.7) * 2);
const sh = new Float32Array(n * shK * 3).map((_, i) => Math.cos(i * 1.3) * 0.5);
const ses = {
  exportRawState: async () => ({ data, n, sh, shK, dc: 'sigmoid' }),
  trainer: { iter: 4321, v2: true },
  opts: { trainer: { growRate: 0.05 }, refineEvery: 500 },
};
const blob = await packState(ses);
const headLen = new DataView(blob.buffer).getUint32(0, true);
const head = JSON.parse(new TextDecoder().decode(blob.subarray(4, 4 + headLen)));
assert.equal(head.version, 2);
assert.equal(blob.length, 4 + headLen + (n * 16 + n * shK * 3) * 2);
const st = parseState(blob);
assert.equal(st.iter, 4321);
assert.equal(st.engine, 'v2');
assert.equal(st.gaussians.dc, 'sigmoid');
assert.equal(st.gaussians.n, n);
assert.equal(st.gaussians.shK, shK);
assert.deepEqual(st.recipe, { growRate: 0.05, refineEvery: 500 });
const close = (a, b) => {
  assert.equal(a.length, b.length);
  for (let i = 0; i < a.length; i++) {
    assert.ok(Math.abs(a[i] - b[i]) <= 2e-3 * Math.abs(b[i]) + 1e-6, `word ${i}: ${a[i]} vs ${b[i]}`);
  }
};
close(st.gaussians.data, data);
close(st.gaussians.sh, sh);

// ---- SH-free runs: no SH section, null back ---------------------------------
const ses0 = {
  exportRawState: async () => ({ data: data.subarray(0, 32), n: 2, sh: null, shK: 0, dc: 'sigmoid' }),
  trainer: { iter: 7 },
};
const st0 = parseState(await packState(ses0));
assert.equal(st0.gaussians.sh, null);
assert.equal(st0.gaussians.n, 2);
assert.equal(st0.engine, 'v1');   // trainer.v2 absent -> the v1 default
close(st0.gaussians.data, data.subarray(0, 32));

// ---- odd header lengths leave the payload unaligned: must still parse ------
const base = JSON.stringify({ ...head, recipe: { pad: '' } });
const pad = 'x'.repeat(base.length % 2 === 0 ? 1 : 0);
const head2 = new TextEncoder().encode(JSON.stringify({ ...head, recipe: { pad } }));
assert.equal((4 + head2.length) % 2, 1);
const blob2 = new Uint8Array(4 + head2.length + (blob.length - 4 - headLen));
new DataView(blob2.buffer).setUint32(0, head2.length, true);
blob2.set(head2, 4);
blob2.set(blob.subarray(4 + headLen), 4 + head2.length);
const st2 = parseState(blob2);
close(st2.gaussians.data, data);
close(st2.gaussians.sh, sh);

// ---- v1 float32 blobs keep parsing byte-exact -------------------------------
const enc = new TextEncoder();
const h1 = enc.encode(JSON.stringify({ magic: 'splatjs-state', version: 1, n: 2, shK: 0, iter: 9 }));
const p1 = new Uint8Array(data.buffer, 0, 2 * 16 * 4);
const b1 = new Uint8Array(4 + h1.length + p1.length);
new DataView(b1.buffer).setUint32(0, h1.length, true);
b1.set(h1, 4);
b1.set(p1, 4 + h1.length);
const s1 = parseState(b1);
assert.deepEqual(s1.gaussians.data, data.subarray(0, 32));
assert.equal(s1.iter, 9);
assert.equal(s1.engine, 'v1');

// ---- future versions are refused, not misread -------------------------------
const hf = enc.encode(JSON.stringify({ magic: 'splatjs-state', version: 3, n: 1, shK: 0, iter: 0 }));
const bf = new Uint8Array(4 + hf.length + 32);
new DataView(bf.buffer).setUint32(0, hf.length, true);
bf.set(hf, 4);
assert.throws(() => parseState(bf), /supported/);
assert.throws(() => parseState(blob.subarray(0, blob.length - 1)), /length/);

console.log('test_state_fp16: OK');
