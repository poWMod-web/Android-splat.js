// EXIF focal reader (src/io/exif.js): JPEG APP1 parsing on a synthetic file
// (always) and on the iPhone statue photos when the set is present (skipped
// otherwise); the 35 mm-equivalent -> pixel conversion; graceful nulls.
import { readFileSync, existsSync, readdirSync } from 'node:fs';
import { readExifFocal, focalPxFrom35 } from '../../src/io/exif.js';

let failures = 0;
const check = (name, cond, detail = '') => {
  console.log(`${cond ? 'PASS' : 'FAIL'} ${name} ${detail}`);
  if (!cond) failures++;
};

// a minimal JPEG: SOI, APP1 Exif with IFD0 -> ExifIFD { FocalLength 6.86 mm, FocalLengthIn35mmFilm 24 }, SOS
function syntheticJpeg() {
  const tiff = [];
  const u16 = (v) => tiff.push(v & 255, v >> 8);            // little-endian
  const u32 = (v) => tiff.push(v & 255, (v >> 8) & 255, (v >> 16) & 255, (v >>> 24) & 255);
  tiff.push(0x49, 0x49); u16(42); u32(8);                   // II, 42, IFD0 at 8
  // IFD0: one entry (ExifIFD pointer) -> ExifIFD at 26
  u16(1); u16(0x8769); u16(4); u32(1); u32(26); u32(0);
  // ExifIFD at 26: two entries; rational value stored at 26 + 2 + 24 + 4 = 56
  u16(2);
  u16(0x920A); u16(5); u32(1); u32(56);                     // FocalLength rational at 56
  u16(0xA405); u16(3); u32(1); u16(24); u16(0);             // FocalLengthIn35mmFilm 24
  u32(0);                                                   // next IFD
  u32(686); u32(100);                                       // 6.86 mm
  const seg = [0x45, 0x78, 0x69, 0x66, 0, 0, ...tiff];
  const len = seg.length + 2;
  return new Uint8Array([0xFF, 0xD8, 0xFF, 0xE1, len >> 8, len & 255, ...seg, 0xFF, 0xDA, 0, 2, 0xFF, 0xD9]);
}

{
  const r = await readExifFocal(new Blob([syntheticJpeg()]));
  check('synthetic JPEG parses', !!r && r.f35 === 24 && Math.abs(r.fmm - 6.86) < 1e-9, JSON.stringify(r));
  check('35 mm -> px (diagonal definition)', Math.abs(focalPxFrom35(24, 8064, 6048) - 5591) < 2, focalPxFrom35(24, 8064, 6048).toFixed(1));
  const none = await readExifFocal(new Blob([new Uint8Array([0xFF, 0xD8, 0xFF, 0xDA, 0, 2, 0xFF, 0xD9])]));
  check('JPEG without APP1 -> null', none === null);
  const junk = await readExifFocal(new Blob([new Uint8Array(200)]));
  check('unknown bytes -> null', junk === null);
}

const dir = new URL('../../data/statue_ka/', import.meta.url);
const path = decodeURIComponent(dir.pathname.replace(/^\/([A-Za-z]:)/, '$1'));
if (existsSync(path)) {
  const files = readdirSync(path).filter((f) => /\.jpe?g$/i.test(f)).slice(0, 3);
  for (const f of files) {
    const r = await readExifFocal(new Blob([readFileSync(`${path}/${f}`)]));
    check(`statue ${f}: iPhone 14 Pro, 24 mm`, !!r && r.f35 === 24 && /iPhone/.test(r.model || ''), JSON.stringify(r));
  }
} else {
  console.log('SKIP statue photos (data/statue_ka absent)');
}

if (failures) { console.log(`${failures} failure(s)`); process.exit(1); }
console.log('EXIF TESTS PASSED');
