// node scripts/restamp_space.mjs <spaceId> <modelKey> [--desc "..."] [--title "..."]
//   [--splats N] [--iter N] [--minutes N] [--psnr N] [--res "979 × 546"] [--frames N]
//   [--sogmb N] [--dry]
// Points a published sample space at a different CDN model: the splat
// UserModelEntity's glbUrl, the space description, and the splatjs stamp the
// wall/share endpoint read (sogUrl/reconUrl/splats/iter/minutes/psnr...).
// Everything else in the stamp (thumb, pin, badge, fpv, framesLabel) is kept.
// Stats default to the model's <key>_recon.json on the CDN.
import { readFileSync } from 'node:fs';
const args = process.argv.slice(2);
const opt = (n, d) => { const i = args.indexOf('--' + n); return i >= 0 ? args[i + 1] : d; };
const [spaceId, key] = args.filter((a, i) => !a.startsWith('--') && (i === 0 || !args[i - 1].startsWith('--')));
if (!spaceId || !key) { console.error('usage: spaceId modelKey [--desc ...]'); process.exit(1); }
const DRY = args.includes('--dry');

const envText = readFileSync('C:/Dev/arrival.space/backend_git/backend_test/.env', 'utf8');
const env = Object.fromEntries(envText.split(/\r?\n/).filter((l) => l.includes('=') && !l.startsWith('#'))
  .map((l) => [l.slice(0, l.indexOf('=')).trim(), l.slice(l.indexOf('=') + 1).trim()]));
const BASE = 'https://api-live.arrival.space';
const CDN = 'https://ugc.arrival.space/splatjs/models';
const j = async (r) => { const t = await r.text(); try { return JSON.parse(t); } catch { return { raw: t.slice(0, 200) }; } };

// model facts from the CDN
const recon = await j(await fetch(`${CDN}/${key}_recon.json`));
if (!recon.version) { console.error('recon missing on CDN', key); process.exit(1); }
const head = await fetch(`${CDN}/${key}.sog`, { method: 'HEAD' });
const sogMb = +opt('sogmb', Math.round(+head.headers.get('content-length') / 1e6));
const st = recon.stats || {};
const psnr = +opt('psnr', st.psnrTest ? st.psnrTest.psnr : NaN);
const stamp = {
  sogUrl: `${CDN}/${key}.sog`, reconUrl: `${CDN}/${key}_recon.json`,
  splats: +opt('splats', recon.splats), iter: +opt('iter', recon.iter), minutes: +opt('minutes', st.minutes),
  psnrTrain: +(+opt('psnrtrain', st.psnrTrain)).toFixed(2),
  psnrTest: Number.isFinite(psnr) ? { psnr: +psnr.toFixed(3), frames: +opt('evalframes', st.psnrTest && st.psnrTest.frames || 0) || undefined } : null,
  frames: +opt('frames', (recon.frames || recon.cams || []).length), sogMb,
};
if (opt('res')) stamp.res = opt('res');
console.log('model facts:', JSON.stringify(stamp, null, 1));

// auth
const login = await j(await fetch(`${BASE}/loginUser/${encodeURIComponent(env.TEST_ADMIN_EMAIL)}/${encodeURIComponent(env.TEST_ADMIN_PW)}`));
if (login.status !== 'ok') { console.log('login failed', login); process.exit(1); }
const keyRes = await j(await fetch(`${BASE}/api/v1/auth/generate-key`, { method: 'POST', headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ userId: login.id, logonCertificate: login.logonCertificate }) }));
const H = { Authorization: `Bearer ${keyRes.data.apiKey}`, 'Content-Type': 'application/json' };

// current space + entities
const space = await j(await fetch(`${BASE}/api/v1/spaces/${spaceId}`, { headers: H }));
if (space.status !== 'ok') { console.log('space read failed', space); process.exit(1); }
const cur = space.data.data.splatjs || {};
console.log('current stamp:', JSON.stringify({ sog: cur.sogUrl, splats: cur.splats, iter: cur.iter, min: cur.minutes, psnrTrain: cur.psnrTrain, psnrTest: cur.psnrTest }));
const ents = await j(await fetch(`${BASE}/api/v1/spaces/${spaceId}/entities?limit=200`, { headers: H }));
const splatEnt = (ents.data.entities || []).find((e) => e.entity_type === 'UserModelEntity' && /\.sog$/.test(e.entity_data.glbUrl || ''));
if (!splatEnt) { console.log('no .sog entity found'); process.exit(1); }
console.log('splat entity:', splatEnt.entity_id, splatEnt.entity_data.glbUrl);

const next = { ...cur, ...stamp };
const body = { splatjs: next };
if (opt('desc')) body.description = opt('desc');
if (opt('descfile')) body.description = readFileSync(opt('descfile'), 'utf8').trim();
if (opt('title')) body.title = opt('title');
console.log('\nnew stamp:', JSON.stringify(next, null, 1));
if (body.description) console.log('new description:', body.description);
if (DRY) { console.log('(dry run — nothing written)'); process.exit(0); }

const e = await j(await fetch(`${BASE}/api/v1/spaces/${spaceId}/entities/${splatEnt.entity_id}`, { method: 'PUT', headers: H,
  body: JSON.stringify({ entity_data: { glbUrl: stamp.sogUrl } }) }));
console.log('entity:', e.status, e.message || '');
const put = await j(await fetch(`${BASE}/api/v1/spaces/${spaceId}`, { method: 'PUT', headers: H, body: JSON.stringify(body) }));
console.log('stamp:', put.status, put.message || '');
const share = await j(await fetch(`${BASE}/api/v1/splatjs/share/${spaceId}`));
const sj = share.data && share.data.splatjs;
console.log('share resolves:', share.status, sj ? `${sj.sogUrl} ${sj.splats} splats ${sj.psnrTest ? sj.psnrTest.psnr + ' holdout' : sj.psnrTrain + ' train'}` : share.message);
console.log(`https://arrival.space/${spaceId}`);
