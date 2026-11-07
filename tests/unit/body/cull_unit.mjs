import { readFileSync } from 'node:fs';
import { cullHeadInterior } from '../../../app/avatar/body/anny.js';
const S = JSON.parse(readFileSync('scratch/lisa_anny_surface3_face.json', 'utf8'));
const corr = JSON.parse(readFileSync('scratch/anny_face_corr.json', 'utf8')).corr;   // 468 face landmarks -> triangles on this mesh
const hb = S.boneLabels.indexOf('head'); const hw = S.vertices.map((_, i) => S.boneIndices[i].reduce((s, b, k) => s + (b === hb ? S.boneWeights[i][k] : 0), 0));
const n0 = S.vertices.length; const t0 = Date.now(); cullHeadInterior(S, hw);
const remap = S.vertexRemap; let kept = 0; for (const c of Object.values(corr)) if (c.tri.every((v) => remap[v] >= 0)) kept++;
console.log(`cull (${Date.now() - t0} ms): ${n0} -> ${S.vertices.length} verts, removed`, S.interiorCulled, `; face-landmark triangles intact: ${kept}/${Object.keys(corr).length}`);
