import assert from 'node:assert/strict';
import { deflateRawSync } from 'node:zlib';
import { readFileSync } from 'node:fs';
import { readZip } from '../../app/js/zip.js';
import { zipStore } from '../../app/js/zip.js';

const zipSource = readFileSync(new URL('../../app/js/zip.js', import.meta.url), 'utf8');
assert.ok(!/\b(document|window|navigator|localStorage|indexedDB)\b/.test(zipSource),
  'the zip codec should stay DOM-free (Blob/Response/DecompressionStream are platform, Node-safe)');

const enc = new TextEncoder();
const same = (a, b) => a.length === b.length && a.every((v, i) => v === b[i]);

// 1. stored entries written by the repo's own writer round-trip
{
  const photo = new Uint8Array([137, 80, 78, 71, 13, 10, 26, 10, 1, 2, 3, 4]);
  const json = enc.encode(JSON.stringify({ frames: [] }));
  const blob = zipStore([
    { name: 'frame_0001.jpg', data: photo },
    { name: 'nested/transforms.json', data: json },
  ]);
  const got = await readZip(new Uint8Array(await blob.arrayBuffer()));
  assert.ok(same(got.get('frame_0001.jpg'), photo), 'stored photo bytes survive');
  assert.ok(same(got.get('nested/transforms.json'), json), 'stored json bytes survive');
}

// 2. deflated entries (what OS compressors emit) inflate — minimal writer
{
  const payload = enc.encode('x'.repeat(5000) + 'pose-data'.repeat(200));
  const comp = deflateRawSync(payload);
  const name = enc.encode('capture/frame_0002.jpg');
  const lh = new DataView(new ArrayBuffer(30));
  lh.setUint32(0, 0x04034b50, true);
  lh.setUint16(8, 8, true);                    // method: deflate
  lh.setUint32(18, comp.length, true);
  lh.setUint32(22, payload.length, true);
  lh.setUint16(26, name.length, true);
  const cd = new DataView(new ArrayBuffer(46));
  cd.setUint32(0, 0x02014b50, true);
  cd.setUint16(10, 8, true);
  cd.setUint32(20, comp.length, true);
  cd.setUint32(24, payload.length, true);
  cd.setUint16(28, name.length, true);
  cd.setUint32(42, 0, true);                   // local header offset
  const cdOff = 30 + name.length + comp.length;
  const eocd = new DataView(new ArrayBuffer(22));
  eocd.setUint32(0, 0x06054b50, true);
  eocd.setUint16(10, 1, true);
  eocd.setUint32(16, cdOff, true);
  const zip = new Uint8Array(cdOff + 46 + name.length + 22);
  zip.set(new Uint8Array(lh.buffer), 0);
  zip.set(name, 30);
  zip.set(comp, 30 + name.length);
  zip.set(new Uint8Array(cd.buffer), cdOff);
  zip.set(name, cdOff + 46);
  zip.set(new Uint8Array(eocd.buffer), cdOff + 46 + name.length);
  const got = await readZip(zip);
  assert.ok(same(got.get('capture/frame_0002.jpg'), payload), 'deflated bytes inflate exactly');
}

// 3. garbage is a user-facing error, not a crash
await assert.rejects(readZip(new Uint8Array([1, 2, 3, 4])), /not a zip file/);

// 4. an unsupported method names the entry and suggests unzip-and-drop
{
  const blob = zipStore([{ name: 'a.jpg', data: new Uint8Array([9, 9]) }]);
  const zip = new Uint8Array(await blob.arrayBuffer());
  const dv = new DataView(zip.buffer);
  for (let o = 0; o < zip.length - 4; o++) {   // rewrite both method fields
    if (dv.getUint32(o, true) === 0x04034b50) dv.setUint16(o + 8, 12, true);
    if (dv.getUint32(o, true) === 0x02014b50) dv.setUint16(o + 10, 12, true);
  }
  await assert.rejects(readZip(zip), /a\.jpg.*method 12.*unzip it and drop the files/);
}

console.log('zip reader (stored + deflated) + stored writer round-trip: passed');
