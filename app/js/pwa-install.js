// SplatXR PWA install affordance.
//
// The browser owns whether an install prompt is available. This module only
// captures the prompt event, gives the two app buttons a click handler, and
// explains the browser-menu path when no prompt is exposed. It does not claim
// offline support or WebGPU support.

const INSTALL_BUTTON_IDS = ['btn-install', 'xr-install-app'];

const getWindow = () => (typeof window === 'undefined' ? null : window);
const getDocument = () => (typeof document === 'undefined' ? null : document);

function isStandaloneWindow() {
  const win = getWindow();
  const nav = win?.navigator || (typeof navigator === 'undefined' ? null : navigator);
  let displayModeStandalone = false;
  try {
    displayModeStandalone = Boolean(win?.matchMedia?.('(display-mode: standalone)')?.matches);
  } catch {
    // Some embedded browsers expose matchMedia but reject display-mode queries.
  }
  return displayModeStandalone || nav?.standalone === true;
}

function installButtons() {
  const doc = getDocument();
  if (!doc) return [];
  return INSTALL_BUTTON_IDS
    .map((id) => doc.getElementById(id))
    .filter((button, index, buttons) => button && buttons.indexOf(button) === index);
}

function showInstalledState(installed) {
  for (const button of installButtons()) {
    // Hiding preserves any icon/label markup supplied by the host page while
    // making it impossible to offer an install action from the installed app.
    button.hidden = installed;
    button.disabled = installed;
    button.setAttribute('aria-disabled', installed ? 'true' : 'false');
    if (installed) button.dataset.pwaInstalled = 'true';
    else delete button.dataset.pwaInstalled;
  }
}

function appendText(parent, tagName, text, className = '') {
  const element = parent.ownerDocument.createElement(tagName);
  if (className) element.className = className;
  element.textContent = text;
  parent.append(element);
  return element;
}

function showManualInstallGuide() {
  const doc = getDocument();
  if (!doc?.body) return null;

  const existing = doc.querySelector('dialog[data-pwa-install-guide]');
  if (existing) {
    try {
      if (!existing.open && typeof existing.showModal === 'function') existing.showModal();
      else if (!existing.open) existing.setAttribute('open', '');
    } catch {
      existing.setAttribute('open', '');
    }
    return existing;
  }

  const dialog = doc.createElement('dialog');
  dialog.className = 'xr-dialog xr-help';
  dialog.dataset.pwaInstallGuide = 'true';
  dialog.setAttribute('aria-labelledby', 'pwa-install-guide-title');

  const closeButton = doc.createElement('button');
  closeButton.className = 'xr-dialog-close';
  closeButton.type = 'button';
  closeButton.setAttribute('aria-label', 'Close install guide');
  closeButton.textContent = '×';
  dialog.append(closeButton);

  appendText(dialog, 'p', 'INSTALL THE APP', 'eyebrow');
  const heading = appendText(dialog, 'h2', 'Add SplatXR to your home screen.');
  heading.id = 'pwa-install-guide-title';
  appendText(dialog, 'p', 'This browser did not offer an automatic install prompt. On a compatible Android device, use the browser menu to add this HTTPS page to your home screen.');
  appendText(dialog, 'h3', 'Android in Chrome');

  const steps = doc.createElement('ol');
  for (const step of [
    'Open SplatXR in a full Chrome browser tab, not an embedded preview.',
    'Open the Chrome menu (⋮).',
    'Choose “Install app” or “Add to Home screen”; the wording varies by device and browser version.',
    'Confirm the install, then launch SplatXR from the home screen when you want the standalone layout.',
  ]) appendText(steps, 'li', step);
  dialog.append(steps);

  appendText(dialog, 'div', 'Installation adds a home-screen/full-screen entry. It does not add WebGPU support, and this project’s service worker intentionally does not cache the app for offline training.', 'xr-notice');

  const actions = doc.createElement('div');
  actions.className = 'xr-help-actions';
  const doneButton = doc.createElement('button');
  doneButton.className = 'btn btn-outline';
  doneButton.type = 'button';
  doneButton.textContent = 'Got it';
  actions.append(doneButton);
  dialog.append(actions);

  const dismiss = () => {
    try {
      if (typeof dialog.close === 'function' && dialog.open) dialog.close();
      else dialog.remove();
    } catch {
      dialog.remove();
    }
  };
  closeButton.addEventListener('click', dismiss);
  doneButton.addEventListener('click', dismiss);
  dialog.addEventListener('click', (event) => {
    if (event.target === dialog) dismiss();
  });
  dialog.addEventListener('close', () => dialog.remove(), { once: true });

  doc.body.append(dialog);
  try {
    if (typeof dialog.showModal === 'function') dialog.showModal();
    else dialog.setAttribute('open', '');
  } catch {
    dialog.setAttribute('open', '');
  }
  return dialog;
}

/**
 * Wire the header and home-footer install buttons.
 *
 * @param {{flash?: (message: string, duration?: number) => void}} options
 * @returns {{isInstalled: () => boolean, hasDeferredPrompt: () => boolean, showManualInstallGuide: () => Element|null, destroy: () => void}}
 */
export function initPWAInstall({ flash } = {}) {
  const win = getWindow();
  let deferredPrompt = null;
  let installed = isStandaloneWindow();
  const boundButtons = new Set();

  const notify = (message, duration = 5000) => {
    if (typeof flash !== 'function') return;
    try { flash(message, duration); } catch { /* UI feedback must not break install. */ }
  };

  const onInstallClick = async () => {
    if (installed) return;

    const promptEvent = deferredPrompt;
    if (!promptEvent || typeof promptEvent.prompt !== 'function') {
      showManualInstallGuide();
      return;
    }

    // The browser permits a deferred prompt to be consumed only once.
    deferredPrompt = null;
    try {
      await promptEvent.prompt();
      const choice = await promptEvent.userChoice;
      if (choice?.outcome === 'accepted') notify('Install requested. SplatXR will appear on your home screen.', 6000);
    } catch {
      // A prompt can become invalid after navigation or browser UI changes.
      // The manual browser-menu instructions are still useful in that case.
      showManualInstallGuide();
    }
  };

  const bindButtons = () => {
    for (const button of installButtons()) {
      if (!boundButtons.has(button)) {
        button.type = 'button';
        button.addEventListener('click', onInstallClick);
        boundButtons.add(button);
      }
    }
    showInstalledState(installed);
  };

  const onBeforeInstallPrompt = (event) => {
    if (typeof event.preventDefault === 'function') event.preventDefault();
    deferredPrompt = event;
    bindButtons();
  };

  const onAppInstalled = () => {
    deferredPrompt = null;
    installed = true;
    showInstalledState(true);
    notify('SplatXR was added to your home screen.', 5000);
  };

  bindButtons();
  if (win?.addEventListener) {
    win.addEventListener('beforeinstallprompt', onBeforeInstallPrompt);
    win.addEventListener('appinstalled', onAppInstalled);
  }

  return {
    isInstalled: () => installed,
    hasDeferredPrompt: () => Boolean(deferredPrompt),
    showManualInstallGuide,
    destroy: () => {
      if (win?.removeEventListener) {
        win.removeEventListener('beforeinstallprompt', onBeforeInstallPrompt);
        win.removeEventListener('appinstalled', onAppInstalled);
      }
      for (const button of boundButtons) button.removeEventListener('click', onInstallClick);
      boundButtons.clear();
      deferredPrompt = null;
    },
  };
}
