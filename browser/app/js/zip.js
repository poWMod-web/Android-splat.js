// zip.js — minimal STORED (no compression) zip writer, plus a reader that
// also inflates (OS compressors deflate by default) — no dependencies.
// JPEGs don't recompress, so stored entries are the right choice anyway.

const CRC_TBL = (() => {
  const t = new Uint32Array(256);
  for (let n = 0; n < 256; n++) {
    let c = n;
    for (let k = 0; k < 8; k++) c = (c & 1) ? (0xEDB88320 ^ (c >>> 1)) : (c >>> 1);
    t[n] = c >>> 0;
  }
  return t;
})();

function crc32(data) {
  let c = 0xFFFFFFFF;
  for (let i = 0; i < data.length; i++) c = CRC_TBL[(c ^ data[i]) & 0xFF] ^ (c >>> 8);
  return (c ^ 0xFFFFFFFF) >>> 0;
}

/** entries: [{ name, data: Uint8Array }] -> zip Blob */
export function zipStore(entries) {
  const enc = new TextEncoder();
  const parts = [];
  const central = [];
  let offset = 0;

  for (const e of entries) {
    const name = enc.encode(e.name);
    const crc = crc32(e.data);
    const local = new DataView(new ArrayBuffer(30));
    local.setUint32(0, 0x04034b50, true);
    local.setUint16(4, 20, true);           // version needed
    local.setUint16(8, 0, true);            // method: stored
    local.setUint32(14, crc, true);
    local.setUint32(18, e.data.length, true);
    local.setUint32(22, e.data.length, true);
    local.setUint16(26, name.length, true);
    parts.push(local.buffer, name, e.data);

    const cd = new DataView(new ArrayBuffer(46));
    cd.setUint32(0, 0x02014b50, true);
    cd.setUint16(4, 20, true);
    cd.setUint16(6, 20, true);
    cd.setUint16(10, 0, true);
    cd.setUint32(16, crc, true);
    cd.setUint32(20, e.data.length, true);
    cd.setUint32(24, e.data.length, true);
    cd.setUint16(28, name.length, true);
    cd.setUint32(42, offset, true);
    central.push(cd.buffer, name);
    offset += 30 + name.length + e.data.length;
  }

  let cdSize = 0;
  for (const c of central) cdSize += c.byteLength ?? c.length;
  const end = new DataView(new ArrayBuffer(22));
  end.setUint32(0, 0x06054b50, true);
  end.setUint16(8, entries.length, true);
  end.setUint16(10, entries.length, true);
  end.setUint32(12, cdSize, true);
  end.setUint32(16, offset, true);
  return new Blob([...parts, ...central, end.buffer], { type: 'application/zip' });
}

async function inflateRaw(raw) {
  const stream = new Blob([raw]).stream().pipeThrough(new DecompressionStream('deflate-raw'));
  return new Uint8Array(await new Response(stream).arrayBuffer());
}

/** Read a whole zip into name → bytes (Uint8Array). Walks the central
 *  directory; stored entries pass through, method 8 inflates via the
 *  platform DecompressionStream. Throws user-facing errors for corrupt
 *  archives and unsupported compression methods. */
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
