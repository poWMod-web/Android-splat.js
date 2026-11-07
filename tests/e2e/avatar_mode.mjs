// Avatar-mode end-to-end through the real app (node tests/e2e/avatar_mode.mjs --train --video=scratch/lisa_test_4k.mp4;
// the .mov must be transcoded to H.264 first: headless Chrome has no HEVC). Stops at the sign-in click; downloads the package.: upload a clip, accept the
// frames, tick "This is a person" on the start-training card, start the run and
// wait for the avatar stages (their bar lives in the dock since 2026-09-18:
// #avbar; the run has no confirmations — --extra=avreview=1 brings them back).
import { chromium, firefox } from 'playwright';
const args = process.argv.slice(2);
const opt = (n, d) => { const a = args.find((x) => x.startsWith(`--${n}=`)); return a ? a.slice(n.length + 3) : d; };
const VIDEO = opt('video', 'scratch/lisa_test_1080.mp4');
const OUT = opt('out', 'scratch/avatar_e2e');
const TRAIN = args.includes('--train');
// --headed: a visible window (the user's tab, not headless); --plainflags: no ANGLE/GPU flags beyond WebGPU;
// --swdecode: force software video decode — the three knobs that separate a headed tab from this harness
const HEADED = args.includes('--headed'), PLAIN = args.includes('--plainflags'), SWDEC = args.includes('--swdecode');
const BROWSER = opt('browser', 'chrome');   // --browser=firefox: Playwright's Firefox with WebGPU enabled (the user's browser, 2026-09-17)
const browser = BROWSER === 'firefox' ? await firefox.launch({ headless: !HEADED, firefoxUserPrefs: { 'dom.webgpu.enabled': true, 'gfx.webgpu.ignore-blocklist': true, 'dom.media.webcodecs.enabled': true, 'media.hardware-video-decoding.force-enabled': true } }) : await chromium.launch({ channel: 'chrome', headless: !HEADED, args: [
  '--enable-unsafe-webgpu', ...(PLAIN ? [] : ['--use-angle=d3d11', '--ignore-gpu-blocklist']), '--autoplay-policy=no-user-gesture-required',
  ...(SWDEC ? ['--disable-accelerated-video-decode'] : []) ] });
const page = await browser.newPage({ viewport: { width: 1280, height: 900 } });
const t0 = Date.now(); const log = (m) => console.log(`[${((Date.now() - t0) / 1000).toFixed(0)}s] ${m}`);
import { appendFileSync, writeFileSync } from 'node:fs';
writeFileSync(`${OUT}_console.log`, '');
page.on('console', (m) => { const t = m.text(); appendFileSync(`${OUT}_console.log`, `[${((Date.now() - t0) / 1000).toFixed(0)}s] ${t.slice(0, 20000)}
`); if (/\[avatar\]|\[video\]|\[session\]|matte|PAGEERROR/i.test(t) && !/favicon|GPU stall/i.test(t)) log('console: ' + t.slice(0, 220)); });
page.on('pageerror', (e) => log('PAGEERROR ' + String(e).slice(0, 300)));
// --faceiters=N runs the face pass (off in the app since 2026-09-14; the e2e used to pass 1500 by default — every 09-15 ladder run had it)
// --settings='{"feat":2048,...}' seeds the persisted settings panel before the page loads (the app's localStorage)
if (opt('settings', '')) await page.addInitScript((v) => { try { localStorage.setItem('splatjs_settings', v); } catch {} }, opt('settings'));
await page.goto(`http://localhost:${opt('port', '8734')}/app/index.html?iters=${opt('iters', '3000')}${opt('faceiters', '') ? '&faceiters=' + opt('faceiters') : ''}${opt('extra', '') ? '&' + opt('extra') : ''}`, { waitUntil: 'load' });
await page.waitForTimeout(2500);
await page.setInputFiles('#file-input', VIDEO);
log('video set');
await page.waitForSelector('#vid-use', { timeout: 300000 });
log('review card up');
const NOAVATAR = args.includes('--noavatar');   // a plain scene run of the same clip (solver A/B against main)
await page.screenshot({ path: `${OUT}_review.png` });
await page.click('#vid-use');
await page.waitForTimeout(1500);
await page.screenshot({ path: `${OUT}_detail.png` });
log('detail card: ' + (await page.textContent('#detail').catch(() => '')).replace(/\s+/g, ' ').slice(0, 160));
// the avatar route is chosen HERE since 2026-09-18: the box on the start-training
// card, with the matte and the "is this the person?" checkpoint after Start training
if (!NOAVATAR) { await page.waitForSelector('#set-avatar', { timeout: 60000 }); await page.check('#set-avatar'); log('avatar box ticked on the start card'); }
if (opt('dumpframes', '')) {   // --dumpframes=dir: save the picked frames as JPEGs (a COLMAP reference solve needs the same pictures)
  const { mkdirSync } = await import('node:fs'); const dir = opt('dumpframes'); mkdirSync(dir, { recursive: true });
  await page.exposeFunction('__saveFrame', (name, b64) => writeFileSync(`${dir}/${name}`, Buffer.from(b64, 'base64')));
  // 208 frames of 4K are still being captured onto the strip when the detail card appears — wait until the count settles
  await page.waitForFunction(() => { const S = window.__splat || {}; const n = (S.photos || []).filter((p) => p && p.url).length; const same = n > 0 && n === window.__lastN; window.__lastN = n; return same; }, null, { timeout: 300000, polling: 3000 });
  const n = await page.evaluate(async () => {
    const S = window.__splat || {}; const photos = (S.photos || []).filter((p) => p && p.url);   // the strip: one object URL per picked frame
    for (const p of photos) { const buf = new Uint8Array(await (await fetch(p.url)).arrayBuffer()); let s = ''; for (let i = 0; i < buf.length; i += 0x8000) s += String.fromCharCode.apply(null, buf.subarray(i, i + 0x8000)); await window.__saveFrame(p.name, btoa(s)); }
    if (!photos.length) console.log('[dump] nothing on S.photos — ' + JSON.stringify({ photos: (S.photos || []).length, preset: S.preset && Object.keys(S.preset), files: S.preset && S.preset.files && S.preset.files.length, urls: S.preset && S.preset.urls && S.preset.urls.length, loaded: S.loadedFiles && S.loadedFiles.length, own: S.ownUrls && S.ownUrls.length }));
    return photos.length;
  });
  log(`${n} frames written to ${dir}`); await browser.close(); process.exit(0);
}
if (TRAIN) {
  await page.click('#btn-go');
  log('run started');
  // the matte runs first now. Its cut-outs gate only exists with --extra=avreview=1
  // (nothing interrupts a normal run since 2026-09-18) — click it when it is there
  if (!NOAVATAR) {
    await page.waitForTimeout(4000);
    if (await page.$('#av-yes')) {
      await page.screenshot({ path: `${OUT}_cutouts.png` });
      log('cutouts gate up (avreview)');
      await page.click('#av-yes');
    }
  }
  if (args.includes('--solveonly')) {   // diagnostics: stop once the solve has seeded the model; the cameras go to OUT_cams.json
    // poll the page state, not the console: a plain scene run only logs the solve with ?sessionlog=1
    await page.waitForFunction(() => { const s = window.__splat && window.__splat.session; return !!(s && s.recon && s.recon.cams && s.recon.cams.length && s.trainer) || /failed|error/i.test((document.querySelector('#failcard') || {}).textContent || ''); }, null, { timeout: 900000, polling: 2000 });
    await page.waitForTimeout(500);
    const cams = await page.evaluate(() => (window.__splat && window.__splat.session && window.__splat.session.recon) ? window.__splat.session.recon.cams.filter((c) => !c.crop).map(({ imgIdx, name, R, t, f, fy, cx, cy }) => ({ imgIdx, name, R: Array.from(R), t: Array.from(t), f, fy, cx, cy })) : null).catch(() => null);
    const ring = await page.evaluate(() => (window.__splat && window.__splat.regCams) ? window.__splat.regCams.length : -1).catch(() => -1);
    if (cams) { writeFileSync(`${OUT}_cams.json`, JSON.stringify(cams)); log(`solve done — ${cams.length} cameras written; overlay ring holds ${ring} frustums`); } else log('solve done — no cams reachable');
    await browser.close(); process.exit(0);
  }
  await page.waitForSelector('#avbar', { timeout: 1800000 });   // the stages' bar in the dock (a card only appears for a decision)
  log('avatar stages started: ' + (await page.textContent('#avbar').catch(() => '?')).replace(/\s+/g, ' ').slice(0, 160));
  const tEnd = Date.now() + 25 * 60000; let last = '', lastCap = '', shotN = 0;
  while (Date.now() < tEnd) {
    await page.waitForTimeout(5000);
    const txt = ((await page.textContent('#avbar').catch(() => '')) + ' | ' + (await page.textContent('#avcard').catch(() => ''))).replace(/\s+/g, ' ').slice(0, 300);
    if (txt !== last) { log('card: ' + txt); last = txt; }
    const dbg = await page.evaluate(() => window.__avatarDebug ? JSON.stringify(window.__avatarDebug) : null).catch(() => null);
    if (dbg && !globalThis.__dbgShown) { globalThis.__dbgShown = true; log('DEBUG ' + dbg.slice(0, 1600)); writeFileSync(`${OUT}_debug.json`, dbg); }
    // the card's picture (joints, body model): screenshot each new one — they
    // are shown, never confirmed, and the last one stays while a stage has none
    const cap = (await page.textContent('#av-shotcap').catch(() => '')) || '';
    if (cap && cap !== lastCap) { lastCap = cap; await page.screenshot({ path: `${OUT}_shot${++shotN}.png` }); log('picture: ' + cap.replace(/\s+/g, ' ').slice(0, 130)); }
    // the step bar goes away at the last stage — the action panel takes over
    if (await page.$('#av-go') || /failed|stopped here/.test(txt) || !((await page.$('#avbar')) || (await page.$('#avact')))) break;
  }
  await page.screenshot({ path: `${OUT}_avatar.png` });
  // the walk preview: two frames apart, so a still shows it is actually moving
  if (await page.$('#c-walk')) {
    await page.waitForTimeout(450);
    await page.screenshot({ path: `${OUT}_walk2.png` });
    log('walk frames: ' + `${OUT}_avatar.png` + ' / ' + `${OUT}_walk2.png`);
  }
  if (await page.$('#av-dl')) {
    const [dl] = await Promise.all([page.waitForEvent('download', { timeout: 120000 }), page.click('#av-dl')]);
    await dl.saveAs(`${OUT}_package.zip`); log('package saved: ' + `${OUT}_package.zip`);
  }
}
await browser.close();
