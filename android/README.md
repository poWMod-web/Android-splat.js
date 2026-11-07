# Android packaging notes

SplatXR is a web app, not a separate native or Expo app. There are two Android delivery modes:

- **PWA/home-screen install:** use the deployed HTTPS site directly. Android Chrome may expose **Install app** or **Add to Home screen**. The installed entry uses `display: "standalone"` from `app/manifest.webmanifest`.
- **Verified Trusted Web Activity:** after a real deployment exists, wrap that same site with the official Bubblewrap CLI. A TWA is still the site running in the browser’s fullscreen surface; it does not bundle WebGPU or turn unsupported hardware into a WebGPU device.

## Start from a deployed site

This repository deliberately has no production URL, release package name, signing certificate, or keystore. Do not generate a random release key or pretend that a signed APK has been tested. First deploy the app over HTTPS and confirm that the browser can fetch:

- `https://your-domain.example/splatxr/index.html`
- `https://your-domain.example/splatxr/manifest.webmanifest`
- the existing icon paths under `icons/`

Then run the repository helper with the deployed directory URL (or its `index.html` URL):

```sh
node android/package-android.mjs https://your-domain.example/splatxr/
```

The helper resolves the relative manifest URL and invokes the maintained package without inventing Bubblewrap configuration:

```sh
npx --yes @bubblewrap/cli init --manifest=https://your-domain.example/splatxr/manifest.webmanifest
```

Review the generated Bubblewrap project and its current CLI output. Once the real Android package ID and signing setup are chosen, build with the official command from that project:

```sh
bubblewrap build
```

The helper stops after `init`; it does not build, sign, install, upload, or publish an APK.

## Verification before release

A TWA needs Digital Asset Links ownership verification. Publish `/.well-known/assetlinks.json` on the exact web origin with the final Android package name and the SHA-256 fingerprint for the certificate that signs the installed package. Those values must come from the actual release credentials. If verification is missing or incorrect, the Android experience can fall back to a Custom Tab instead of an owned fullscreen TWA.

After signing, test the release package on physical Android hardware against the final HTTPS deployment. Check the fullscreen launch, navigation back to the site, camera permission flow, WebXR availability, and WebGPU capability separately. A successful TWA launch does not imply WebXR or WebGPU support.

## App limitations

- `id: "./"`, `start_url: "index.html"`, `scope: "./"`, and the existing 192px/512px/maskable icons are the stable manifest contract.
- The app’s install controller uses `beforeinstallprompt` when the browser supplies it, otherwise it explains the manual browser-menu path. In standalone display mode or after `appinstalled`, the install buttons are hidden.
- `app/sw.js` intentionally has an empty fetch handler and no cache. Do not describe the installed app as an offline training package.
- Splat training remains dependent on browser WebGPU and device resources. Android packaging cannot force a missing WebGPU implementation.

## Official references

- [Trusted Web Activity quick start (Chrome Developers)](https://developer.chrome.com/docs/android/trusted-web-activity/quick-start)
- [Trusted Web Activities (Android Developers)](https://developer.android.com/develop/ui/views/layout/webapps/trusted-web-activities)
- [GoogleChromeLabs Bubblewrap](https://github.com/GoogleChromeLabs/bubblewrap)
