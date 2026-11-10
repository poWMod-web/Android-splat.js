// browser/zip.js — DOM-free zip reader for user-supplied archives.
//
// The engine's own reader only takes stored entries; OS compressors deflate
// by default, so this one walks the central directory and inflates method 8
// via the platform DecompressionStream. No dependencies, runs in Node too
// (pinned by tests/unit/test_browser_zip.mjs).

async function inflateRaw(raw) {
  const stream = new Blob([raw]).stream().pipeThrough(new DecompressionStream('deflate-raw'));
  return new Uint8Array(await new Response(stream).arrayBuffer());
}

/** Read a whole zip into name → bytes (Uint8Array). Throws user-facing
 *  errors for corrupt archives and unsupported compression methods. */
export async function readZip(bytes) {
  const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  let eocd = -1;
  for (let o = bytes.length - 22; o >= Math.max(0, bytes.length - 65557); o--) {
    if (dv.getUint32(o, true) === 0x06054b50) { eocd = o; break; }
  }
  if (eocd < 0) throw new Error('not a zip file (end-of-central-directory not found)');
  const count = dv.getUint16(eocd + 10, true);
  let o = dv.getUint32(eocd + 16, true);
  const out = new Map();
  for (let i = 0; i < count; i++) {
    if (o + 46 > bytes.length || dv.getUint32(o, true) !== 0x02014b50) {
      throw new Error('zip central directory is corrupt');
    }
    const method = dv.getUint16(o + 10, true);
    const compSize = dv.getUint32(o + 20, true);
    const nameLen = dv.getUint16(o + 28, true);
    const extraLen = dv.getUint16(o + 30, true);
    const comLen = dv.getUint16(o + 32, true);
    const localOff = dv.getUint32(o + 42, true);
    const name = new TextDecoder().decode(bytes.subarray(o + 46, o + 46 + nameLen));
    o += 46 + nameLen + extraLen + comLen;
    if (name.endsWith('/')) continue;
    if (dv.getUint32(localOff, true) !== 0x04034b50) throw new Error(`zip entry ${name} has a corrupt local header`);
    const lnLen = dv.getUint16(localOff + 26, true);
    const leLen = dv.getUint16(localOff + 28, true);
    const start = localOff + 30 + lnLen + leLen;
    const raw = bytes.subarray(start, start + compSize);
    if (method === 0) out.set(name, raw.slice());
    else if (method === 8) out.set(name, await inflateRaw(raw));
    else throw new Error(`zip entry ${name} uses compression method ${method} — unzip it and drop the files instead`);
  }
  return out;
}
