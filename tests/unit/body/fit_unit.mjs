import { readFileSync } from 'node:fs';
import { parseAnny } from '../../../app/avatar/body/anny.js';
import { fitBody } from '../../../app/avatar/body/fit.js';
import { plyCenters } from '../../../app/avatar/rig/export-binding-core.js';
const buf = readFileSync('app/models/anny_game_engine.bin'); const anny = parseAnny(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength));
const src = JSON.parse(readFileSync('scratch/lisa_markers2.json', 'utf8'));
const ply = readFileSync('scratch/lisa_face_v3.ply'); const centers = plyCenters(ply);
const ref = JSON.parse(readFileSync('scratch/lisa_anny_markers3.json', 'utf8'));
const t0 = Date.now();
const fit = await fitBody(anny, { landmarks: src.landmarks, report: src.report, splat: centers, log: console.log, progress: (m, f) => { if (f === 1) console.log('  ' + m); } });
console.log(`fit in ${((Date.now() - t0) / 1000).toFixed(1)}s; scale ${fit.scale.toFixed(3)} (python ${ref.scale.toFixed(3)})`);
const res = Object.values(fit.residualsM); console.log('landmark residual mean', (res.reduce((a, b) => a + b, 0) / res.length * 100).toFixed(2), 'cm (python', (Object.values(ref.landmarkResidualM).reduce((a, b) => a + b, 0) / Object.keys(ref.landmarkResidualM).length * 100).toFixed(2), 'cm)');
let sum = 0, n = 0; for (const [k, v] of Object.entries(fit.markers)) { const r = ref.markers[k]; if (!r) continue; sum += Math.hypot(v[0] - r[0], v[1] - r[1], v[2] - r[2]); n++; }
console.log(`markers vs python fit: mean distance ${(sum / n * 100).toFixed(2)} cm over ${n}`);
console.log('surface verts', fit.surface.vertices.length, 'faces', fit.surface.faces.length);
