import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

const app = readFileSync(new URL('../../app/js/app.js', import.meta.url), 'utf8');
const manifest = readFileSync(new URL('../../android/app/src/main/AndroidManifest.xml', import.meta.url), 'utf8');
const main = readFileSync(new URL('../../android/app/src/main/java/space/arrival/splatxr/MainActivity.java', import.meta.url), 'utf8');
const service = readFileSync(new URL('../../android/app/src/main/java/space/arrival/splatxr/TrainingService.java', import.meta.url), 'utf8');

// The manifest declares the background-training service and its permissions.
assert.match(manifest, /<uses-permission android:name="android\.permission\.FOREGROUND_SERVICE" \/>/,
  'the manifest should allow foreground services');
assert.match(manifest, /<uses-permission android:name="android\.permission\.FOREGROUND_SERVICE_SPECIAL_USE" \/>/,
  'the manifest should allow the special-use training service');
assert.match(manifest, /<uses-permission android:name="android\.permission\.WAKE_LOCK" \/>/,
  'the manifest should allow the training wake lock');
assert.match(manifest, /<uses-permission android:name="android\.permission\.POST_NOTIFICATIONS" \/>/,
  'the manifest should allow progress notifications');
assert.match(manifest, /<service android:name="\.TrainingService" android:exported="false" android:process=":training" android:foregroundServiceType="specialUse">/,
  'the manifest should declare the training service as special-use in the :training process');
assert.match(manifest, /PROPERTY_SPECIAL_USE_FGS_SUBTYPE/,
  'the service should declare its special-use subtype property');

// The service: an ongoing progress notification, foreground priority, and a
// partial wake lock; swiping the app away leaves nothing stale behind.
assert.match(service, /new NotificationChannel\(\s*CHANNEL_ID, "Training progress", NotificationManager\.IMPORTANCE_LOW\)/,
  'the service should post progress on a low-importance channel');
assert.match(service, /startForeground\(NOTIF_ID, notification, ServiceInfo\.FOREGROUND_SERVICE_TYPE_SPECIAL_USE\)/,
  'the service should start foreground as special-use on Android 14+');
assert.match(service, /setOngoing\(ongoing\)/,
  'training should keep a persistent, non-dismissible ongoing notification');
assert.match(service, /setForegroundServiceBehavior\(Notification\.FOREGROUND_SERVICE_IMMEDIATE\)/,
  'Android 12+ should expose the foreground-service notification immediately');
assert.match(service, /PARTIAL_WAKE_LOCK, "SplatXR:training"/,
  'the service should hold a partial wake lock while training');
assert.match(service, /case "start":[\s\S]*case "update":[\s\S]*case "finish":[\s\S]*default:/,
  'the service should handle start, update, finish and cancel commands');
assert.match(service, /if \(foreground\) \{/,
  'stray updates after a cancel should not resurrect the notification');
assert.match(service, /setProgress\(total, Math\.min\(done, total\), false\)/,
  'updates should carry the deterministic training progress bar');
assert.match(service, /notifications\(\)\.notify\(DONE_ID/,
  'a finish should leave one dismissible summary notification');
assert.match(service, /public void onTaskRemoved\(Intent rootIntent\) \{/,
  'swiping the app away should reach the takeover-or-teardown branch');
assert.match(service, /String active = RunFiles\.readActive\(this\);[\s\S]*?if \(active == null\) \{[\s\S]*?teardown\(\);[\s\S]*?sendBroadcast\(handoff\);[\s\S]*?HANDOFF_ACK_MS/,
  'a swipe with no outlived run should tear down; with one, the service should ask for the pass, settling on a timer');
assert.match(service, /checkSelfPermission\(Manifest\.permission\.POST_NOTIFICATIONS\)[\s\S]*?!= PackageManager\.PERMISSION_GRANTED\) return;/,
  'the finish summary should be skipped when notifications are denied instead of crashing');
assert.match(service, /catch \(Exception ignored\) \{ \/\* progress display must never crash training \*\/ \}/,
  'progress posts should never crash the app');
assert.match(service, /catch \(Exception ignored\) \{ \/\* a summary must never crash the app \*\/ \}/,
  'the finish summary should never crash the app');
assert.match(service, /startForeground\(NOTIF_ID, notification[\s\S]*?catch \(Exception e\) \{[\s\S]*?teardown\(\);/,
  'a refused foreground start should stop quietly instead of crashing');
assert.match(service, /if \(intent == null\) \{[\s\S]*?resurrectIfOurs\(active\)[\s\S]*?stopSelf\(\);/,
  'a restart without a command should resurrect only a service-owned run, else stop instead of going zombie');
assert.match(service, /releaseWakeLock\(\);/,
  'every teardown path should release the wake lock');
assert.match(service, /case "paused":/,
  'the service should handle the paused-state command');
assert.match(service, /builder\.addAction\(lastPaused \? android\.R\.drawable\.ic_media_play : android\.R\.drawable\.ic_media_pause,/,
  'the ongoing notification should carry a Pause/Resume action');
assert.match(service, /PendingIntent\.getBroadcast\(this, 1, toggle,/,
  'the shade button should fire a broadcast instead of opening the app');
assert.match(service, /if \(lastPaused\) releaseWakeLock\(\); else acquireWakeLock\(\);/,
  'a shade pause should free the CPU while keeping the notification up');

// The bridge: four feature-detected entry points, a runtime notification
// permission request, and cleanup when the activity dies.
assert.match(main, /startTrainingNotification\(String title, String text\)/,
  'the bridge should start the training notification');
assert.match(main, /updateTrainingNotification\(String title, String text, int done, int total\)/,
  'the bridge should update the notification with stats and progress');
assert.match(main, /finishTrainingNotification\(String title, String text\)/,
  'the bridge should post the completion summary');
assert.match(main, /cancelTrainingNotification\(\)/,
  'the bridge should cancel the notification');
assert.match(main, /Manifest\.permission\.POST_NOTIFICATIONS/,
  'starting training should request the notification permission on Android 13+');
assert.match(main, /if \("start"\.equals\(action\) && Build\.VERSION\.SDK_INT >= 26\) startForegroundService\(intent\);\s*else startService\(intent\);/,
  'only the start command should raise a foreground service; the rest are plain starts');
assert.match(main, /stopTrainingService\(\);/,
  'destroying the activity should stop the training service');
assert.match(main, /ACTION_TOGGLE_TRAINING = "space\.arrival\.splatxr\.TOGGLE_TRAINING"/,
  'the activity should own the shade-toggle broadcast action');
assert.match(main, /setTrainingPaused\(boolean paused\)/,
  'the bridge should flip the notification paused state');
assert.match(main, /ContextCompat\.registerReceiver\(this, toggleTrainingReceiver, toggleFilter,\s*ContextCompat\.RECEIVER_NOT_EXPORTED\)/,
  'the shade listener should register NOT_EXPORTED via ContextCompat (bare registerReceiver fails Android 14+ lint)');
assert.match(main, /unregisterReceiver\(toggleTrainingReceiver\)/,
  'the activity should drop the shade listener on destroy');
assert.match(main, /evaluateJavascript\("window\.__splatxrNotifToggle&&window\.__splatxrNotifToggle\(\)", null\)/,
  'a shade toggle should reach the page as a JS call');

// The web side: start with training, tick live stats throttled, pause sheds
// the service, finish and failure settle the shade. No bridge, no-op.
assert.match(app, /typeof b\.updateTrainingNotification === 'function' \? b : null/,
  'notification calls should be feature-detected for desktop browsers');
assert.match(app, /function androidNotifStart\(\)/,
  'training start should post the notification');
assert.match(app, /function androidNotifTick\(\)/,
  'training progress should tick the notification');
assert.match(app, /function androidNotifStop\(finished\)/,
  'training end should settle the notification');
assert.match(app, /S\.runId = keepId \|\| `run_[\s\S]*?`;\s*androidNotifStart\(\);/,
  'every training run should be background-safe from the first cycle');
assert.match(app, /S\.itersPerSec = m\.itersPerSec;\s*androidNotifTick\(\);/,
  'every metrics beat should offer the notification fresh stats');
assert.match(app, /now - \(S\._notifT \|\| 0\) < 2500/,
  'notification updates should be throttled');
assert.match(app, /\$\{fmt\(S\.splats\)\} splats · \$\{fmt\(S\.iter\)\} \/ \$\{fmt\(S\.maxIters\)\}/,
  'the notification should carry splats and the cycle count');
assert.match(app, /function notifScore\(\) \{[\s\S]*?S\.scoreScale === 'similarity'[\s\S]*?psnrToApproxSsimPercent\(psnr\)\.toFixed\(1\)\} %/,
  'the notification score should follow the dB/% switch');
assert.match(app, /\$\{notifScore\(\)\}\$\{fmt\(S\.splats\)\} splats/,
  'progress ticks and the finish summary should share the unit-aware score');
assert.match(app, /if \(on\) \{ if \(!notifSetPaused\(false\)\) androidNotifStart\(\); \}/,
  'resuming should flip the shade button, restarting the service only on old bridges');
assert.match(app, /else if \(!notifSetPaused\(true\)\) androidNotifStop\(false\);/,
  'pausing should keep the shade up with Resume, shedding only on old bridges');
assert.match(app, /function notifSetPaused\(paused\) \{[\s\S]*?typeof b\.setTrainingPaused !== 'function'/,
  'the shade flip should be feature-detected per method for old APKs');
assert.match(app, /globalThis\.__splatxrNotifToggle = \(\) => \{ if \(S\.state === 'train'\) toggleTrain\(\); \};/,
  'the shade button should toggle training through the same path as the in-app button');
assert.match(app, /androidNotifStop\(true\);   \/\/ the shade keeps the final score/,
  'finishing should leave the final score in the shade');
assert.match(app, /function solveFailed\(why\) \{\s*androidNotifStop\(false\);/,
  'a failed run should clear the notification');

// A wedged frame surfaces train-stall as a flash AND a stuck notification:
// the user never watches "Starting" forever again.
assert.match(app, /if \(e\.kind === 'train-stall'\) \{/,
  'the stall event should reach the user');
assert.match(app, /Training stalled: a frame has run/,
  'the stall should name the wedged frame');
assert.match(app, /Stalled at \$\{fmt\(e\.iter\)\} — GPU overwhelmed/,
  'the notification should say stalled, not hang at Starting');

// The snapshot view: when the worker canvas path is dead, worker renders
// still reach the screen as blitted pixels (no canvas needed).
assert.match(app, /function engageCaptureView\(why\) \{/,
  'exhausting remounts should engage the snapshot view');
assert.match(app, /cv-model-cap/,
  'the snapshot view should own its 2D canvas in the stage');
assert.match(app, /S\.session\.view\.capture\(\)\.then\(\(shot\) => \{/,
  'the snapshot view should blit worker renderCapture pixels');
assert.match(app, /snapshot view instead/,
  'the fallback should say so plainly');

console.log('background training notification: passed');
