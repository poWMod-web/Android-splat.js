// packState/parseState round-trip — pure node, no browser. The state blob is
// the crash-safety currency: v2 stores float16 words (half the bytes of the
// v1 float32 layout), carries its dc-convention and engine tags, and old v1
// blobs — with or without the tags — keep parsing byte-exact.
import { test, expect } from '@playwright/test';
import { packState, parseState } from '../../app/js/session_io.js';

const n = 1234, shK = 15;
const data = new Float32Array(n * 16).map((_, i) => Math.sin(i * 0.7) * 2);
const sh = new Float32Array(n * shK * 3).map((_, i) => Math.cos(i * 1.3) * 0.5);

const bytesEqual = (a, b) =>
  a.byteLength === b.byteLength &&
  Buffer.compare(Buffer.from(a.buffer, a.byteOffset, a.byteLength),
    Buffer.from(b.buffer, b.byteOffset, b.byteLength)) === 0;

// float16-close: fp16 carries ~3 decimal digits, so a resume re-converges
// the quantization noise within a few hundred iterations
const closeEnough = (a, b) => {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) {
    if (Math.abs(a[i] - b[i]) > 2e-3 * Math.abs(b[i]) + 1e-6) return false;
  }
  return true;
};

for (const [engine, v2, dc] of [['v1', false, 'sigmoid'], ['v2', true, 'sh']]) {
  test(`round-trip float16-close, tags carried (${engine})`, async () => {
    const ses = {
      exportRawState: async () => ({ data, n, sh, shK, dc }),
      trainer: { iter: 4321, v2 },
    };
    const blob = await packState(ses);
    const headLen = new DataView(blob.buffer).getUint32(0, true);
    const head = JSON.parse(new TextDecoder().decode(blob.subarray(4, 4 + headLen)));
    expect(head.version).toBe(2);
    expect(blob.length).toBe(4 + headLen + (n * 16 + n * shK * 3) * 2);
    const st = parseState(blob);
    expect(st.iter).toBe(4321);
    expect(st.engine).toBe(engine);
    expect(st.gaussians.dc).toBe(dc);
    expect(st.gaussians.n).toBe(n);
    expect(st.gaussians.shK).toBe(shK);
    expect(closeEnough(st.gaussians.data, data)).toBe(true);
    expect(closeEnough(st.gaussians.sh, sh)).toBe(true);
  });
}

test('legacy v1 blob parses byte-exact with v1 defaults', async () => {
  const head = new TextEncoder().encode(
    JSON.stringify({ magic: 'splatjs-state', version: 1, n, shK, iter: 7 }));
  const params = new Uint8Array(data.buffer, 0, n * 16 * 4);
  const shb = new Uint8Array(sh.buffer, 0, n * shK * 3 * 4);
  const blob = new Uint8Array(4 + head.length + params.length + shb.length);
  new DataView(blob.buffer).setUint32(0, head.length, true);
  blob.set(head, 4);
  blob.set(params, 4 + head.length);
  blob.set(shb, 4 + head.length + params.length);
  const st = parseState(blob);
  expect(st.iter).toBe(7);
  expect(st.engine).toBe('v1');
  expect(st.gaussians.dc).toBe('sigmoid');
  expect(bytesEqual(st.gaussians.data, data)).toBe(true);
  expect(bytesEqual(st.gaussians.sh, sh)).toBe(true);
});

test('truncated, foreign or future blobs are rejected', async () => {
  expect(() => parseState(new Uint8Array(2))).toThrow();
  const junk = new TextEncoder().encode('not a state blob at all............');
  expect(() => parseState(junk)).toThrow();
  const future = new TextEncoder().encode(
    JSON.stringify({ magic: 'splatjs-state', version: 3, n: 1, shK: 0, iter: 0 }));
  const blob = new Uint8Array(4 + future.length + 2);
  new DataView(blob.buffer).setUint32(0, future.length, true);
  blob.set(future, 4);
  expect(() => parseState(blob)).toThrow(/supported/);
});
