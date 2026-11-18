// featworker.js — SIFT extraction in a Web Worker (extraction is per-image
// independent; a pool of these turns ~100s of sequential CPU work into ~15s).
import { detectSift } from './sift.js';

self.onmessage = (e) => {
  const { id, gray, w, h, maxFeats, firstOctave, peakScale, spatialBalance } = e.data;
  const f = detectSift(gray, w, h, maxFeats, firstOctave, peakScale ?? 1, !!spatialBalance);
  self.postMessage(
    { id, n: f.n, x: f.x, y: f.y, scale: f.scale, angle: f.angle, desc: f.desc },
    [f.x.buffer, f.y.buffer, f.scale.buffer, f.angle.buffer, f.desc.buffer],
  );
};
