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
assert.match(manifest, /<service android:name="\.TrainingService" android:exported="false" android:foregroundServiceType="specialUse">/,
  'the manifest should declare the training service as special-use');
assert.match(manifest, /PROPERTY_SPECIAL_USE_FGS_SUBTYPE/,
  'the service should declare its special-use subtype property');

// The service: an ongoing progress notification, foreground priority, and a
// partial wake lock; swiping the app away leaves nothing stale behind.
assert.match(service, /new NotificationChannel\(\s*CHANNEL_ID, "Training progress", NotificationManager\.IMPORTANCE_LOW\)/,
  'the service should post progress on a low-importance channel');
assert.match(service, /startForeground\(NOTIF_ID, notification, ServiceInfo\.FOREGROUND_SERVICE_TYPE_SPECIAL_USE\)/,
  'the service should start foreground as special-use on Android 14+');
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
  'swiping the app away should tear the service down');
assert.match(service, /if \(intent == null\) \{ stopSelf\(\); return START_NOT_STICKY; \}/,
  'a restart without a command should stop instead of going zombie');
assert.match(service, /releaseWakeLock\(\);/,
  'every teardown path should release the wake lock');

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
assert.match(main, /if \(Build\.VERSION\.SDK_INT >= 26\) startForegroundService\(intent\);\s*else startService\(intent\);/,
  'commands should start foreground where the platform requires it');
assert.match(main, /stopTrainingService\(\);/,
  'destroying the activity should stop the training service');

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
assert.match(app, /S\.runId = resumeId \|\| `run_[\s\S]*?`;\s*androidNotifStart\(\);/,
  'every training run should be background-safe from the first cycle');
assert.match(app, /S\.itersPerSec = m\.itersPerSec;\s*androidNotifTick\(\);/,
  'every metrics beat should offer the notification fresh stats');
assert.match(app, /now - \(S\._notifT \|\| 0\) < 2500/,
  'notification updates should be throttled');
assert.match(app, /\$\{fmt\(S\.splats\)\} splats · \$\{fmt\(S\.iter\)\} \/ \$\{fmt\(S\.maxIters\)\}/,
  'the notification should carry splats and the cycle count');
assert.match(app, /if \(on\) androidNotifStart\(\); else androidNotifStop\(false\);/,
  'pausing should shed the service and resuming should restart it');
assert.match(app, /androidNotifStop\(true\);   \/\/ the shade keeps the final score/,
  'finishing should leave the final score in the shade');
assert.match(app, /function solveFailed\(why\) \{\s*androidNotifStop\(false\);/,
  'a failed run should clear the notification');

console.log('background training notification: passed');
