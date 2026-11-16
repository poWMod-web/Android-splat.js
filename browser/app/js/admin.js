// admin.js — a small moderation sheet for the community wall (index.html?admin).
//
// Lists every scene the public gallery shows (presets aside) with its sharer
// and date, and lets a signed-in admin unlist one: the space's privacy goes
// to "Link Only", so the share link keeps resolving but the wall no longer
// shows it. Relist puts it back to "Open". The public endpoints cannot list
// Link Only spaces, so the sheet remembers what it unlisted (this browser)
// and takes a space id by hand for anything else.
//
// Rights come from the platform: the stored sign-in must be allowed to
// update other users' spaces (an admin account); everyone else gets the
// API's refusal as a message.

import { getToken, hasToken, API_BASE } from './arrival.js';
import { fetchGallery, shareLink, setSharePrivacy } from './share.js';

const LS_UNLISTED = 'splatjs.admin.unlisted';
const esc = (s) => String(s ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const fmt = (n) => Math.round(n || 0).toLocaleString('en-US');
const when = (d) => { const t = new Date(d); return isNaN(t) ? '' : t.toLocaleDateString(undefined, { day: 'numeric', month: 'short', year: 'numeric' }); };
const loadUnlisted = () => { try { return JSON.parse(localStorage.getItem(LS_UNLISTED) || '[]'); } catch { return []; } };
const saveUnlisted = (a) => { try { localStorage.setItem(LS_UNLISTED, JSON.stringify(a)); } catch {} };

/**
 * @param {object} o
 * @param {(msg: string, ms?: number) => void} o.flash
 * @param {(ownerId: string) => Promise<{name: string, avatar: string}|null>} o.ownerInfo
 * @param {Set<string>} o.presetIds   official demo scenes (listed apart, never unlisted here)
 * @param {() => void} o.onClose
 */
export async function adminDashboard({ flash, ownerInfo, presetIds, onClose }) {
  document.getElementById('admincard')?.remove();
  const card = document.createElement('div');
  card.className = 'upcard';
  card.id = 'admincard';
  card.innerHTML = `
    <button class="card-x" id="adm-x" aria-label="Close">&times;</button>
    <div class="adm-head"><b>Community shares</b><span class="prep-sub" id="adm-sub">loading …</span></div>
    <div class="adm-tools">
      <label class="adm-manual">Space id <input id="adm-id" type="text" spellcheck="false" placeholder="12345678_1234"></label>
      <button class="btn btn-outline" id="adm-unlist-id">Unlist</button>
      <button class="btn btn-outline" id="adm-relist-id">Relist</button>
      <label class="adm-check"><input type="checkbox" id="adm-presets"> show presets</label>
      <span class="adm-spacer"></span>
      <button class="btn btn-accent" id="adm-signin" hidden>Sign in with Arrival.Space</button>
    </div>
    <div class="adm-list" id="adm-list"></div>`;
  document.getElementById('stage').appendChild(card);
  const $ = (id) => card.querySelector('#' + id);
  const close = () => { card.remove(); onClose && onClose(); };
  $('adm-x').onclick = close;
  const onKey = (e) => { if (e.key === 'Escape') { window.removeEventListener('keydown', onKey); close(); } };
  window.addEventListener('keydown', onKey);

  let busy = false;
  const signedIn = () => { $('adm-signin').hidden = hasToken(); };
  $('adm-signin').onclick = async () => {
    const popup = window.open('', 'arrival-oauth', 'width=480,height=720');
    if (!popup) { flash('The sign-in window was blocked — allow popups for this site.', 6000); return; }
    try { await getToken((m) => flash(m, 60000), popup); flash('Signed in.', 2500); signedIn(); }
    catch (e) { flash(`Sign-in failed: ${e.message}`, 8000); }
  };
  signedIn();

  /** Flip a space's stored privacy; the sheet's memory follows. */
  async function setPrivacy(it, privacy) {
    if (busy) return;
    if (!hasToken()) { flash('Sign in first (an admin account).', 5000); return; }
    busy = true;
    try {
      await setSharePrivacy(it.id, privacy);
      const mem = loadUnlisted().filter((u) => u.id !== it.id);
      if (privacy !== 'Open') mem.unshift({ id: it.id, title: it.title || '', when: Date.now(), owner: String(it.id).split('_')[0], thumb: it.thumb || '' });
      saveUnlisted(mem);
      flash(`${it.title || it.id}: ${privacy === 'Open' ? 'listed again' : 'unlisted (link still works)'}`, 5000);
      await render();
    } catch (e) {
      flash(`${it.id}: ${e.message}`, 9000);
    } finally { busy = false; }
  }
  $('adm-unlist-id').onclick = () => { const id = $('adm-id').value.trim(); if (id) setPrivacy({ id, title: id }, 'Link Only'); };
  $('adm-relist-id').onclick = () => { const id = $('adm-id').value.trim(); if (id) setPrivacy({ id, title: id }, 'Open'); };
  $('adm-presets').onchange = () => render();

  function row(it, kind) {
    const el = document.createElement('div');
    el.className = 'adm-row';
    el.dataset.kind = kind;
    const thumb = it.thumb || (it.splatjs && it.splatjs.thumbUrl) || it.screenshotUrl || '';
    const dB = it.splatjs && (it.splatjs.psnrTest ? it.splatjs.psnrTest.psnr : it.splatjs.psnrTrain);
    const stats = it.splatjs ? `${fmt(it.splatjs.splats)} splats${dB ? ` · ${(+dB).toFixed(1)} dB` : ''}${it.splatjs.frames ? ` · ${it.splatjs.frames} photos` : ''}` : '';
    el.innerHTML = `
      <a class="adm-thumb" href="${esc(shareLink(it.id))}" target="_blank" rel="noopener">${thumb ? `<img src="${esc(thumb)}" alt="" loading="lazy">` : ''}</a>
      <div class="adm-main">
        <b>${esc(it.title || 'Untitled')}</b>
        <span class="adm-meta"><span class="adm-by"><i class="galav"></i><em></em></span> · ${esc(when(it.createdDate || it.when))} · <code>${esc(it.id)}</code></span>
        <span class="adm-meta">${esc(stats)}</span>
      </div>
      <span class="adm-state" data-state="${kind}">${kind === 'listed' ? 'listed' : kind === 'preset' ? 'preset' : 'unlisted'}</span>
      <span class="adm-acts">
        ${kind === 'listed' ? '<button class="btn btn-outline" data-act="unlist">Unlist</button>' : ''}
        ${kind === 'unlisted' ? '<button class="btn btn-accent" data-act="relist">Relist</button>' : ''}
        <a class="btn btn-quiet" href="${esc(shareLink(it.id))}" target="_blank" rel="noopener">Open</a>
      </span>`;
    el.querySelector('[data-act="unlist"]')?.addEventListener('click', () => setPrivacy({ ...it, thumb }, 'Link Only'));
    el.querySelector('[data-act="relist"]')?.addEventListener('click', () => setPrivacy(it, 'Open'));
    const ownerId = String(it.id).split('_')[0];
    ownerInfo(ownerId).then((u) => {
      const by = el.querySelector('.adm-by');
      if (!u || !u.name) { by.textContent = ownerId; return; }
      by.querySelector('em').textContent = u.name;
      const av = by.querySelector('.galav');
      if (u.avatar) av.style.backgroundImage = `url("${u.avatar}")`; else av.textContent = u.name.trim().charAt(0).toUpperCase();
      by.title = `profile ${ownerId}`;
      by.style.cursor = 'pointer';
      by.onclick = () => window.open(`https://profile.arrival.space/${encodeURIComponent(ownerId)}`, '_blank', 'noopener');
    });
    return el;
  }

  async function render() {
    const list = $('adm-list');
    list.innerHTML = '';
    let items = [];
    try { items = (await fetchGallery({ count: 48 })).items || []; }
    catch (e) { $('adm-sub').textContent = `gallery unavailable: ${e.message}`; return; }
    const listedIds = new Set(items.map((i) => String(i.id)));
    const shares = items.filter((i) => !presetIds.has(String(i.id))).sort((a, b) => new Date(b.createdDate) - new Date(a.createdDate));
    const presets = items.filter((i) => presetIds.has(String(i.id)));
    const unlisted = loadUnlisted().filter((u) => !listedIds.has(String(u.id)));
    saveUnlisted(unlisted);   // anything listed again elsewhere drops out of the memory
    $('adm-sub').textContent = `${shares.length} listed by people · ${unlisted.length} unlisted here · ${presets.length} presets`;
    const section = (label) => { const h = document.createElement('div'); h.className = 'adm-section'; h.textContent = label; list.appendChild(h); };
    if (shares.length) { section('Listed in the gallery'); for (const it of shares) list.appendChild(row(it, 'listed')); }
    if (unlisted.length) { section('Unlisted from this browser'); for (const it of unlisted) list.appendChild(row(it, 'unlisted')); }
    if ($('adm-presets').checked && presets.length) { section('Presets'); for (const it of presets) list.appendChild(row(it, 'preset')); }
    if (!shares.length && !unlisted.length) { const p = document.createElement('p'); p.className = 'adm-empty'; p.textContent = 'Nothing shared by people right now.'; list.appendChild(p); }
  }
  await render();
  return { close, render };
}
