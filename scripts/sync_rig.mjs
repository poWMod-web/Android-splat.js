// sync_rig.mjs — copy the rigger's pure core into the app (app/avatar/rig/).
//
// Avatar mode binds a splat to the rpm_std rig in the tab with the same code
// the rigger and its CLIs use (client_git/splat-rigger/src). Until that core
// is its own package, this snapshot is checked in here and refreshed with:
//
//   node scripts/sync_rig.mjs
import { copyFileSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const src = join(here, '..', '..', 'client_git', 'splat-rigger');
const dst = join(here, '..', 'app', 'avatar', 'rig');
mkdirSync(dst, { recursive: true });
const FILES = ['glb.js', 'align-solver.js', 'skin-transfer.js', 'autofit-core.js', 'export-binding-core.js'];
for (const f of FILES) copyFileSync(join(src, 'src', f), join(dst, f));
copyFileSync(join(src, 'data', 'glb_avatar.glb'), join(dst, 'glb_avatar.glb'));
// the walk cycle the finished avatar is previewed with (app/avatar/anim.js)
const CLIPS = ['walking_anim.glb'];
for (const f of CLIPS) copyFileSync(join(src, 'data', f), join(dst, f));
let head = '';
try { head = readFileSync(join(src, '..', '.git', 'HEAD'), 'utf8').trim(); } catch { /* no git */ }
writeFileSync(join(dst, 'SYNC.md'), `Snapshot of client_git/splat-rigger/src (${FILES.join(', ')}) + data/glb_avatar.glb + data/${CLIPS.join(', data/')}.\nDo not edit here — run node scripts/sync_rig.mjs after changing the rigger.\nsynced ${new Date().toISOString()} from ${head}\n`);
console.log(`synced ${FILES.length} modules + glb_avatar.glb + ${CLIPS.length} clip(s) -> app/avatar/rig/`);
