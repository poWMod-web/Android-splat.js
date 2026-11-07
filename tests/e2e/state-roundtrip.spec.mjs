// packState/parseState round-trip — pure node, no browser. The state blob is
// the crash-safety currency: it must survive byte-exact, carry its
// dc-convention and engine tags, and old blobs without the tags must keep
// parsing as v1/sigmoid.
import { test, expect } from '@playwright/test';
import { packState, parseState } from '../../app/js/session_io.js';

const n = 1234, shK = 15;
const data = new Float32Array(n * 16).map((_, i) => Math.sin(i * 0.7) * 2);
const sh = new Float32Array(n * shK * 3).map((_, i) => Math.cos(i * 1.3) * 0.5);

const bytesEqual = (a, b) =>
  a.byteLength === b.byteLength &&
  Buffer.compare(Buffer.from(a.buffer, a.byteOffset, a.byteLength),
    Buffer.from(b.buffer, b.byteOffset, b.byteLength)) === 0;

for (const [engine, v2, dc] of [['v1', false, 'sigmoid'], ['v2', true, 'sh']]) {
  test(`round-trip byte-exact, tags carried (${engine})`, async () => {
    const ses = {
      exportRawState: async () => ({ data, n, sh, shK, dc }),
      trainer: { iter: 4321, v2 },
    };
    const st = parseState(await packState(ses));
    expect(st.iter).toBe(4321);
    expect(st.engine).toBe(engine);
    expect(st.gaussians.dc).toBe(dc);
    expect(st.gaussians.n).toBe(n);
    expect(st.gaussians.shK).toBe(shK);
    expect(bytesEqual(st.gaussians.data, data)).toBe(true);
    expect(bytesEqual(st.gaussians.sh, sh)).toBe(true);
  });
}

test('legacy blob without tags parses with v1 defaults', async () => {
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
});

test('truncated or foreign blobs are rejected', async () => {
  expect(() => parseState(new Uint8Array(2))).toThrow();
  const junk = new TextEncoder().encode('not a state blob at all............');
  expect(() => parseState(junk)).toThrow();
});
