// share.js — the social layer on top of the trainer.
//
// A shared creation IS an arrival space: the SOG is its walkable model, and
// a `splatjs` block in the space data (written through the same authed API
// the upload button already uses) carries everything the splat.js viewer
// needs to present it — sogUrl, recon, thumbnail, training stats. Reading
// is public: two read-only endpoints resolve share links and page the
// gallery, honoring the space's privacy ("Open" and "Link Only" resolve,
// only "Open" is listed).
//
//   Public scenes are also registered through the platform's publish route
//   (feed order, and the creator's followers are told only when they chose so).
//
//   share link:  <app>/index.html?space=<spaceId>
//   enter space: https://arrival.space/<spaceId>

import { getToken, api, uploadFile, forgetRevokedToken, storedToken, API_BASE } from './arrival.js';
import { buildReconJson } from './session_io.js';
import { zipStore } from './zip.js';

const API = `${API_BASE}/api/v1`;

/** Publish the current run as a shared arrival space.
 *  Returns { spaceId, spaceUrl, link }. */
export async function shareCreation(S, sogBlob, {
  title, privacy = 'Link Only', includePhotos = false, informFollowers = false, thumbBlob = null,
  popup = null, onStatus = () => {}, onProgress = () => {}, recon: reconOverride = null,
  stats: statsOverride = null, device = null,
} = {}) {
  const token = await getToken(onStatus, popup);
  const slug = (title || 'splat').toLowerCase().replace(/\W+/g, '_');

  try {
    // 1) the walkable space, with the SOG as its model
    const { resourceKey, fileUrl: sogUrl } = await uploadFile(sogBlob, `${slug}.sog`, { token, onStatus, onProgress });
    onStatus('Creating the space …');
    const space = await api('/user/create-space', token, {
      space_data: { title: title || 'Splat.js scene', description: 'Trained in the browser with Splat.js', resource_key: resourceKey },
    });
    const spaceUrl = space.data.space_url;
    const spaceId = String(spaceUrl).split('/').pop();

    // 2) the recon (tour, compare, stats) — plus the photographs when the
    //    creator opted in (preset runs already reference public URLs).
    //    Photos upload in parallel, four in flight; a stray 429 retries.
    const recon = reconOverride || buildReconJson(S); // records share without a live session
    if (includePhotos && S.loadedFiles && S.loadedFiles.length) {
      const total = S.loadedFiles.length;
      let done = 0;
      let next = 0;
      const worker = async () => {
        for (;;) {
          const i = next++;
          if (i >= total) return;
          const f = S.loadedFiles[i];
          const up = await uploadResilient(f.source || f, `${slug}_${f.name}`, token, onStatus);
          recon.source.urls[i] = up.fileUrl;
          onStatus(`Uploading photos … ${++done}/${total}`);
        }
      };
      await Promise.all(Array.from({ length: 4 }, worker));
    }
    // 2b) strip thumbnails, packed into ONE stored zip: the viewer's film
    //     strip draws 140px cards — without this it pulls every full-size
    //     training photograph just to paint them (reported: minutes of
    //     loading on a share link). Best-effort — a share without the pack
    //     simply falls back to the old per-photo loads.
    try {
      const files = S.loadedFiles || [];
      const urls = recon.source && recon.source.urls;
      if (!reconOverride && urls && urls.length && urls.every(Boolean) && files.length === urls.length) {
        const entries = [];
        for (let i = 0; i < files.length; i++) {
          onStatus(`Packing strip thumbnails … ${i + 1}/${files.length}`);
          const bm = await createImageBitmap(files[i].source || files[i], { resizeWidth: 280, resizeQuality: 'medium' });
          const cv = document.createElement('canvas');
          cv.width = bm.width; cv.height = bm.height;
          cv.getContext('2d').drawImage(bm, 0, 0);
          bm.close();
          const blob = await new Promise((res) => cv.toBlob(res, 'image/jpeg', 0.7));
          if (!blob) throw new Error('thumb encode failed');
          entries.push({ name: `${i}.jpg`, data: new Uint8Array(await blob.arrayBuffer()) });
        }
        const tz = await uploadFile(zipStore(entries), `${slug}_thumbs.zip`, { token, contentType: 'application/zip', onStatus });
        recon.source.thumbs = tz.fileUrl;
      }
    } catch (e) { console.warn('thumbnail pack skipped:', e); }
    const rz = await uploadFile(
      new Blob([JSON.stringify(recon)], { type: 'application/json' }),
      `${slug}_recon.json`, { token, contentType: 'application/json', onStatus });

    // 3) the thumbnail for the gallery tile
    let thumbUrl = null;
    if (thumbBlob) {
      const th = await uploadFile(thumbBlob, `${slug}_thumb.webp`, { token, contentType: 'image/webp', onStatus });
      thumbUrl = th.fileUrl;
    }

    // 4) stamp the space: description + the splatjs block the public
    //    endpoints resolve
    onStatus('Publishing …');
    // a stored run shares WITHOUT a live session — off the wall there is no
    // S.session to read the cycle count from, and S itself is empty
    const st = statsOverride || {
      splats: S.splats,
      iter: S.session.trainer.iter,
      minutes: S.minutes || 0,
      psnrTrain: S.psnrTrain ?? null,
      psnrTest: S.psnrTest ? { psnr: S.psnrTest.psnr, frames: S.psnrTest.frames.length } : null,
    };
    const dB = st.psnrTest ? st.psnrTest.psnr : st.psnrTrain;
    await api(`/spaces/${spaceId}`, token, {
      description: `${title} — ${Number(st.splats || 0).toLocaleString('en-US')} splats trained in the browser by Splat.js` +
        (dB ? ` · ${dB.toFixed(1)} dB` : ''),
      privacy,
      splatjs: {
        version: 1,
        sogUrl,
        reconUrl: rz.fileUrl,
        thumbUrl,
        splats: st.splats,
        iter: st.iter,
        minutes: st.minutes || 0,
        device,   // the machine it was trained on, short: "PC, RTX 5080", "iPhone"
        // input facts for the pre-start detail card
        frames: (recon.source && recon.source.names && recon.source.names.length) || null,
        res: (recon.frames && recon.frames[0] && recon.frames[0].tw)
          ? `${recon.frames[0].tw} × ${recon.frames[0].th}` : null,
        psnrTrain: st.psnrTrain,
        psnrTest: st.psnrTest,
      },
    }, 'PUT');

    // 4b) the space opens on the capture's own flight (WEB-7704). Best-effort:
    //     the share is complete and correct without an intro.
    try {
      onStatus('Building the intro flight …');
      const { attachIntroCam } = await import('./introcam.js');
      const intro = await attachIntroCam(spaceId, recon.cams, { token, slug, onStatus });
      if (intro) console.log(`intro cutscene: ${intro.keys} keys, ${intro.seconds}s`);
    } catch (e) { console.warn('intro cutscene skipped:', e.message); }

    // 5) a public scene goes through the platform's publish route as well:
    //    it registers the space as published (feed order) and tells the
    //    creator's followers only when they asked for it. Best-effort — the
    //    share itself is complete without it.
    if (privacy === 'Open') {
      try {
        await api('/spaces/update-privacy', token, { spaceId, roomPrivacy: 'Public', informFollowers: !!informFollowers });
      } catch (e) { console.warn('publish registration skipped:', e.message); }
    }

    return { spaceId, spaceUrl, link: shareLink(spaceId) };
  } catch (e) {
    if (forgetRevokedToken(e)) {
      throw new Error('your Arrival.Space key was revoked — press Share again to sign in');
    }
    throw e;
  }
}

/** uploadFile that shrugs off a stray 429 or network blip with a short
 *  retry — resilience, not pacing. */
async function uploadResilient(blob, fileName, token, onStatus, tries = 4) {
  for (let a = 1; ; a++) {
    try {
      return await uploadFile(blob, fileName, { token, contentType: 'image/jpeg' });
    } catch (e) {
      if (a >= tries || e.auth) throw e;
      await new Promise((r) => setTimeout(r, 3000 * a));
    }
  }
}

export const shareLink = (spaceId) => {
  // always target index.html: the extensionless /splat-js URL 301s through
  // the CDN and loses its query string on the way
  const path = location.pathname.endsWith('.html')
    ? location.pathname
    : location.pathname.replace(/\/$/, '') + '/index.html';
  return `${location.origin}${path}?space=${spaceId}`;
};

/** Resolve a share link (public — no account, no key). */
export async function resolveShare(spaceId) {
  const res = await fetch(`${API}/splatjs/share/${encodeURIComponent(spaceId)}`);
  if (res.status === 404) throw new Error('this space has no shared splat (or it is private)');
  if (!res.ok) throw new Error(`share lookup failed (${res.status})`);
  const data = await res.json();
  return data.data;
}

/** One page of the public gallery. */
export async function fetchGallery({ count = 12, before = null } = {}) {
  const u = new URL(`${API}/splatjs/gallery`);
  u.searchParams.set('count', count);
  if (before) u.searchParams.set('before', before);
  const res = await fetch(u);
  if (!res.ok) return { items: [], nextBefore: null };
  const data = await res.json();
  return data.data || { items: [], nextBefore: null };
}

// ---------------------------------------------------------------------------
// managing your own shares (needs the stored sign-in — no popup here)
// ---------------------------------------------------------------------------

/** The signed-in user's shares, privacy included (management view). */
export async function fetchMine() {
  const token = storedToken();
  if (!token) return null;
  // no-store: this list changes the moment the visitor shares something
  const res = await fetch(`${API}/splatjs/mine`, { headers: { Authorization: `Bearer ${token}` }, cache: 'no-store' });
  if (!res.ok) return null;
  const data = await res.json();
  return (data.data && data.data.items) || [];
}

/** Flip a share's visibility — "Closed" is the kill-switch: the link stops
 *  resolving instantly. Takes stored privacy values (Open/Link Only/Closed). */
export function setSharePrivacy(spaceId, privacy) {
  return api(`/spaces/${encodeURIComponent(spaceId)}`, storedToken(), { privacy }, 'PUT');
}

/** Rename a share: the space's title, which is also the gallery tile's
 *  caption and the name the viewer shows. */
export function renameShare(spaceId, title) {
  return api(`/spaces/${encodeURIComponent(spaceId)}`, storedToken(), { title }, 'PUT');
}

/** Delete the share's space entirely. */
export function deleteShare(spaceId) {
  return api(`/spaces/${encodeURIComponent(spaceId)}`, storedToken(), undefined, 'DELETE');
}
