# Android installation

SplatXR remains one web app. The source in this repository can be installed as an Android home-screen web app when it is served from a compatible HTTPS deployment; this does not create a second Expo or native application.

## Install from Android Chrome

1. Deploy the built `app/` (or the project’s published equivalent) at its final HTTPS URL.
2. Open that URL in a full Chrome tab. Embedded previews and in-app browsers can omit install and camera capabilities.
3. If Chrome exposes the install prompt, use it. Otherwise open the Chrome menu (`⋮`) and choose **Install app** or **Add to Home screen**. The wording varies by Android device and browser version.
4. Launch the new home-screen entry to use the manifest’s standalone display mode.

The app UI has an **Install** action in the header and on the home footer. When the browser exposes `beforeinstallprompt`, that action invokes the browser prompt. If it does not, the action opens an accessible in-app guide with the manual Chrome-menu steps. In standalone display mode, the actions are hidden because the app is already installed.

### What installation does not guarantee

- A home-screen install does not add WebGPU support. Training still needs a browser/device with the required WebGPU implementation and enough GPU memory; SplatXR cannot force WebGPU support.
- WebXR/AR camera support is also browser- and device-dependent. Use the existing device guide and permissions flow.
- The service worker in this project intentionally performs no caching. Installation is not an offline training bundle, and this project does not advertise offline operation.
- Use HTTPS and a full browser tab for camera permissions and install eligibility.

## A signed Android package is a later deployment step

This repository does **not** contain a production URL, final Android application ID, signing certificate, or release keystore. Therefore it is not safe to claim a verified signed APK from the source tree, and no random signing key is generated here.

For a production URL, the supported route is a **Trusted Web Activity (TWA)** using the official Bubblewrap CLI:

```sh
# Run from the repository root, after the site is deployed over HTTPS.
node android/package-android.mjs https://your-domain.example/splatxr/

# In the Bubblewrap project created by `init`, review the generated config,
# choose the real package ID/signing setup, then run the official command:
bubblewrap build
```

The helper only runs the official `@bubblewrap/cli init --manifest=...` command against the deployed `manifest.webmanifest`. It does not install an Android SDK, choose an application ID, create a keystore, sign an APK, or publish anything. If the CLI needs local prerequisites, follow its current official setup guidance rather than copying an unverified SDK path into this repository.

Before shipping a TWA, the deployment owner must:

1. Keep the final HTTPS origin/path and manifest stable (`id: "./"`, `start_url: "index.html"`, `scope: "./"`).
2. Choose and retain the final Android package name and signing key.
3. Publish `/.well-known/assetlinks.json` on the web origin with that package name and the SHA-256 fingerprint of the certificate that signs the installed app.
4. Build and sign the release package, then test the installed app on real Android hardware.

The Digital Asset Links verification is what lets the TWA open the owned site fullscreen. Without a valid association, Android can fall back to a browser Custom Tab rather than a verified fullscreen TWA. The exact signing and asset-links values must come from the actual release key; placeholders are not sufficient.

## Official references

- [Trusted Web Activity quick start (Chrome Developers)](https://developer.chrome.com/docs/android/trusted-web-activity/quick-start)
- [Trusted Web Activities (Android Developers)](https://developer.android.com/develop/ui/views/layout/webapps/trusted-web-activities)
- [GoogleChromeLabs Bubblewrap](https://github.com/GoogleChromeLabs/bubblewrap)
