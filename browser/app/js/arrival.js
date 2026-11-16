// arrival.js — send the trained splat into the visitor's Arrival.Space.
//
// Auth is OAuth 2.0 Authorization Code + PKCE against the Arrival.Space API
// (dynamic client registration, public client, S256). The access token IS the
// user's API key (it does not expire), so it is kept in localStorage and every
// later export is one click. The OAuth redirect happens in a POPUP — the app
// page holds a trained model in memory that a top-level redirect would destroy.
//
// Upload is the documented three-step flow (codex.arrival.space/api/
// file-upload-api): presigned URL -> S3 PUT -> upload-complete (with async job
// polling), then user/create-space with the resource key.

const API_BASE = new URLSearchParams(location.search).get('api') === 'dev'
  ? 'https://api-dev.arrival.space'
  : 'https://api-live.arrival.space';
const API = `${API_BASE}/api/v1`;

const LS_CLIENT = `arrival_client:${API_BASE}`;
const LS_TOKEN = `arrival_token:${API_BASE}`;

/** The OAuth popup redirects back to THIS page with ?code=. Call first thing
 *  at boot: in the popup it reports back to the opener and closes; returns
 *  true so the popup skips booting the whole app. */
export function handleOAuthCallback() {
  const q = new URLSearchParams(location.search);
  if (!q.has('code') || !window.opener) return false;
  window.opener.postMessage(
    { type: 'arrival-oauth', code: q.get('code'), state: q.get('state') },
    location.origin);
  document.body.innerHTML =
    '<p style="font: 14px system-ui; color: #cfd8dc; padding: 40px">Signed in — you can close this window.</p>';
  setTimeout(() => window.close(), 400);
  return true;
}

const b64url = (buf) => btoa(String.fromCharCode(...new Uint8Array(buf)))
  .replaceAll('+', '-').replaceAll('/', '_').replaceAll('=', '');

function randomString(n) {
  const a = new Uint8Array(n);
  crypto.getRandomValues(a);
  return [...a].map((v) => 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789'[v % 62]).join('');
}

const redirectUri = () => location.origin + location.pathname;

/** RFC 7591 dynamic registration, once per origin+API host. */
async function getClient() {
  const cached = localStorage.getItem(LS_CLIENT);
  if (cached) {
    const c = JSON.parse(cached);
    if (c.redirect_uris && c.redirect_uris.includes(redirectUri())) return c;
  }
  const res = await fetch(`${API_BASE}/register`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      client_name: 'Splat.js',
      client_uri: location.origin + location.pathname,
      redirect_uris: [redirectUri()],
      grant_types: ['authorization_code'],
      response_types: ['code'],
      token_endpoint_auth_method: 'none',
      scope: 'mcp:tools',
    }),
  });
  if (!res.ok) throw new Error(`client registration failed (${res.status})`);
  const client = await res.json();
  localStorage.setItem(LS_CLIENT, JSON.stringify(client));
  return client;
}

/** Is a (never-expiring) token already stored? Callers use this to decide
 *  whether the click needs to open a sign-in window SYNCHRONOUSLY — a
 *  window.open after any await is popup-blocked. */
export function hasToken() {
  return !!localStorage.getItem(LS_TOKEN);
}

/** The stored token itself (null when signed out) — for calls that must
 *  never trigger a sign-in popup, like the "Mine" management list. */
export function storedToken() {
  return localStorage.getItem(LS_TOKEN);
}

/** Who the key in this browser belongs to — the name and picture the header
 *  chip wears, and whether the account moderates. Null when signed out; a key
 *  the server no longer honours is forgotten here, so the corner falls back to
 *  Login instead of lying. Memoised: the header and the wall both ask. */
let mePromise = null;
export function whoAmI() {
  if (!mePromise) mePromise = loadMe();
  return mePromise;
}

async function loadMe() {
  const token = localStorage.getItem(LS_TOKEN);
  if (!token) return null;
  try {
    const res = await fetch(`${API}/user/me`, { headers: { Authorization: `Bearer ${token}` }, cache: 'no-store' });
    if (res.status === 401) { localStorage.removeItem(LS_TOKEN); return null; }
    if (!res.ok) return null;
    const u = (await res.json()).data;
    return u ? {
      id: u.id,
      name: u.name || u.uniqueName || 'Signed in',
      avatar: u.profileImageUrl || u.avatarUrl || '',
      isAdmin: !!u.isAdmin,
    } : null;
  } catch { return null; }   // offline: the key is still there, say nothing
}

/** Forget the key held in this browser. Nothing server-side changes — what
 *  was published stays published. */
export function signOut() {
  localStorage.removeItem(LS_TOKEN);
  mePromise = null;
}

// the sign-in window of a round-trip still waiting for the visitor
let pendingSignIn = null;

/** Bring an unfinished sign-in's window back to the front; false when no
 *  sign-in is waiting. The login page's "Register on Arrival.Space" link
 *  opens a tab of ITS own, so someone who leaves to make an account leaves
 *  this window behind them — finding it again is the whole way back. */
export function focusSignIn() {
  if (!pendingSignIn || pendingSignIn.closed) { pendingSignIn = null; return false; }
  try { pendingSignIn.focus(); } catch { /* the window may refuse focus */ }
  return true;
}

/** Full PKCE round-trip; resolves to the access token. `popup` must be a
 *  window opened synchronously inside the user's click (about:blank is
 *  fine — it gets navigated to the login page here). */
async function signIn(onStatus, popup) {
  if (!popup || popup.closed) {
    throw new Error('sign-in window unavailable — press Upload again');
  }
  pendingSignIn = popup;
  try {
    return await signInRoundTrip(onStatus, popup);
  } finally {
    if (pendingSignIn === popup) pendingSignIn = null;
  }
}

async function signInRoundTrip(onStatus, popup) {
  const client = await getClient();
  const verifier = randomString(64);
  const challenge = b64url(await crypto.subtle.digest('SHA-256', new TextEncoder().encode(verifier)));
  const state = randomString(32);

  const auth = new URL(`${API_BASE}/authorize`);
  auth.searchParams.set('client_id', client.client_id);
  auth.searchParams.set('redirect_uri', redirectUri());
  auth.searchParams.set('response_type', 'code');
  auth.searchParams.set('code_challenge', challenge);
  auth.searchParams.set('code_challenge_method', 'S256');
  auth.searchParams.set('state', state);
  auth.searchParams.set('scope', 'mcp:tools');

  popup.location.href = auth.href;
  onStatus('Sign in to Arrival.Space in the popup …');

  const code = await new Promise((resolve, reject) => {
    const timer = setTimeout(() => { cleanup(); reject(new Error('sign-in timed out')); }, 5 * 60_000);
    const watch = setInterval(() => {
      if (popup.closed) { cleanup(); reject(new Error('sign-in window was closed')); }
    }, 800);
    const onMsg = (e) => {
      if (e.origin !== location.origin || !e.data || e.data.type !== 'arrival-oauth') return;
      cleanup();
      if (e.data.state !== state) { reject(new Error('OAuth state mismatch')); return; }
      resolve(e.data.code);
    };
    const cleanup = () => { clearTimeout(timer); clearInterval(watch); removeEventListener('message', onMsg); };
    addEventListener('message', onMsg);
  });

  const res = await fetch(`${API_BASE}/token`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
    body: new URLSearchParams({
      grant_type: 'authorization_code',
      code,
      redirect_uri: redirectUri(),
      client_id: client.client_id,
      code_verifier: verifier,
    }),
  });
  if (!res.ok) throw new Error(`token exchange failed (${res.status})`);
  const tok = await res.json();
  if (!tok.access_token) throw new Error('no access token in response');
  localStorage.setItem(LS_TOKEN, tok.access_token);
  return tok.access_token;
}

async function getToken(onStatus, popup) {
  return localStorage.getItem(LS_TOKEN) || signIn(onStatus, popup);
}

async function api(path, token, body, method) {
  const res = await fetch(`${API}${path}`, {
    method: method || (body ? 'POST' : 'GET'),
    headers: {
      Authorization: `Bearer ${token}`,
      ...(body ? { 'Content-Type': 'application/json' } : {}),
    },
    body: body ? JSON.stringify(body) : undefined,
  });
  if (res.status === 401) { const e = new Error('unauthorized'); e.auth = true; throw e; }
  const data = await res.json().catch(() => ({}));
  if (!res.ok || data.status === 'error') {
    throw new Error(data.message || `${path} failed (${res.status})`);
  }
  return data;
}

// the share layer builds on the same auth + upload machinery
export { getToken, api, API_BASE };

function putWithProgress(params, blob, onProgress) {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.upload.onprogress = (e) => {
      if (e.lengthComputable) onProgress(Math.round((e.loaded / e.total) * 100));
    };
    xhr.onload = () => (xhr.status >= 200 && xhr.status < 300)
      ? resolve() : reject(new Error(`upload failed (${xhr.status})`));
    xhr.onerror = () => reject(new Error('network error during upload'));
    xhr.open(params.method, params.url);
    for (const [k, v] of Object.entries(params.headers || {})) xhr.setRequestHeader(k, v);
    xhr.send(blob);
  });
}

async function pollJob(jobId, token, onStatus) {
  for (;;) {
    await new Promise((r) => setTimeout(r, 2000));
    const data = await api(`/jobs/${jobId}`, token);
    const job = data.data;
    onStatus(`Processing on Arrival.Space … ${job.progress ?? 0}%`);
    if (job.job_status === 'completed') return job.result.resource_key;
    if (job.job_status === 'failed') throw new Error(job.error || 'processing failed');
  }
}

/** One file through the documented three-step flow: presign -> S3 PUT ->
 *  upload-complete (+ async job polling). Returns { resourceKey, fileUrl } —
 *  fileUrl is the file's public CDN address. */
export async function uploadFile(blob, fileName, { token, contentType = 'application/octet-stream', onStatus = () => {}, onProgress = () => {} }) {
  onStatus(`Preparing ${fileName} …`);
  const up = await api('/files/upload', token, {
    file_name: fileName,
    file_size: blob.size,
    content_type: contentType,
  });
  const params = up.data.params;

  onStatus(`Uploading ${fileName} …`);
  await putWithProgress(params, blob, onProgress);

  onStatus('Confirming upload …');
  const fileUrl = params.url.split('?')[0];
  const done = await api('/files/upload-complete', token, {
    status: 'success',
    extra_info: { file_url: fileUrl },
  });
  const resourceKey = (done.status === 'processing' && done.data.job_id)
    ? await pollJob(done.data.job_id, token, onStatus)
    : done.data.resource_key;
  return { resourceKey, fileUrl: done.data.url || done.data.file_url || fileUrl };
}

/** Sign-in errors from a revoked stored key: forget the key so the next
 *  click starts a fresh sign-in (a popup can only open inside a click). */
export function forgetRevokedToken(e) {
  if (!e.auth) return false;
  localStorage.removeItem(LS_TOKEN);
  return true;
}

/** The whole thing: sign in (once), upload the .ply, make a space.
 *  Returns the new space's URL (spaceUrl) and its id. `popup`: a
 *  synchronously opened window for the first sign-in (null when
 *  hasToken()). */
export async function sendToArrival(blob, title, { ext = 'ply', onStatus = () => {}, onProgress = () => {}, popup = null, cams = null } = {}) {
  const token = await getToken(onStatus, popup);
  const slug = (title || 'splat').toLowerCase().replace(/\W+/g, '_');
  const fileName = `${slug}.${ext}`;

  try {
    const { resourceKey } = await uploadFile(blob, fileName, { token, onStatus, onProgress });
    onStatus('Creating your space …');
    const space = await api('/user/create-space', token, {
      space_data: { title: title || 'Splat.js scene', description: 'Trained in the browser with Splat.js', resource_key: resourceKey },
    });
    const spaceUrl = space.data.space_url;
    const spaceId = String(spaceUrl).split('/').pop();
    // the space opens on the capture's own flight (WEB-7704) — best-effort,
    // the upload stands without it
    if (cams && cams.length > 1) {
      try {
        const { attachIntroCam } = await import('./introcam.js');
        await attachIntroCam(spaceId, cams, { token, slug, onStatus });
      } catch (e) { console.warn('intro cutscene skipped:', e.message); }
    }
    return { spaceUrl, spaceId };
  } catch (e) {
    if (forgetRevokedToken(e)) {
      throw new Error('your Arrival.Space key was revoked — press Upload again to sign in');
    }
    throw e;
  }
}
