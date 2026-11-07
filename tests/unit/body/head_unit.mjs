import { readFileSync } from 'node:fs';
import { deformHead } from '../../../app/avatar/body/head.js';
const S = JSON.parse(readFileSync('scratch/lisa_anny_surface3_head.json', 'utf8'));
const corrJ = JSON.parse(readFileSync('scratch/anny_face_corr.json', 'utf8')).corr;
const face3 = JSON.parse(readFileSync('scratch/face_canon3d.json', 'utf8')).landmarks;
const hb = S.boneLabels.indexOf('head');
const headWeight = S.vertices.map((_, i) => S.boneIndices[i].reduce((s, b, k) => s + (b === hb ? S.boneWeights[i][k] : 0), 0));
const corr = new Map(Object.entries(corrJ).map(([k, v]) => [+k, v]));
const face = new Map(Object.entries(face3).map(([k, v]) => [+k, v]));
const t0 = Date.now();
deformHead(S, headWeight, corr, face, console.log);
console.log(`deformHead in ${((Date.now() - t0) / 1000).toFixed(1)}s`);
const ref = JSON.parse(readFileSync('scratch/lisa_anny_surface3_face.json', 'utf8'));
let s = 0, m = 0; for (let i = 0; i < S.vertices.length; i++) { const d = Math.hypot(...[0, 1, 2].map((a) => S.vertices[i][a] - ref.vertices[i][a])); s += d; m = Math.max(m, d); }
console.log(`vs python head_deform: mean ${(s / S.vertices.length * 1000).toFixed(2)} mm, max ${(m * 1000).toFixed(1)} mm`);
