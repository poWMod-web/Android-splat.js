package space.arrival.splatxr;

import android.Manifest;
import android.app.Notification;
import android.app.NotificationChannel;
import android.app.NotificationManager;
import android.app.PendingIntent;
import android.app.Service;
import android.content.BroadcastReceiver;
import android.content.Context;
import android.content.Intent;
import android.content.IntentFilter;
import android.content.pm.PackageManager;
import android.content.pm.ServiceInfo;
import android.net.Uri;
import android.os.Build;
import android.os.Handler;
import android.os.IBinder;
import android.os.Looper;
import android.os.PowerManager;
import android.util.Log;
import android.webkit.ConsoleMessage;
import android.webkit.JavascriptInterface;
import android.webkit.WebChromeClient;
import android.webkit.WebResourceRequest;
import android.webkit.WebView;
import android.webkit.WebViewClient;

import androidx.core.content.ContextCompat;
import androidx.webkit.WebViewAssetLoader;

import org.json.JSONObject;

import java.io.File;

/** Foreground service behind training. Two jobs:
 *
 *  1. The notification: a live score + splat count in the shade, process
 *     priority while the app runs in the background, and a partial wake lock
 *     so the GPU work is not suspended with the screen. Driven best-effort
 *     from the Activity's WebView through MainActivity's bridge.
 *
 *  2. Migrate-on-swipe (design B): this service runs in the :training process
 *     (see the manifest). While the app is open it only carries the
 *     notification; when the task is swiped away it boots a HEADLESS WebView
 *     (own data dir, same asset handlers) that resumes the run from the last
 *     file checkpoint and keeps training. The reopened app monitors the run
 *     through the files-dir bus (RunFiles — no binder, no parcel limits):
 *     progress + snapshots out, pause/resume/stop/snapshot commands in.
 *
 *  The Activity's viewer path is never touched: while the app is open the run
 *  is local and live; headless training only ever owns a run the Activity has
 *  demonstrably left (task removed + stale Activity heartbeat).
 */
public final class TrainingService extends Service {
    private static final String TAG = "SplatXRHeadless";
    private static final String CHANNEL_ID = "splatxr-training";
    private static final int NOTIF_ID = 101;
    private static final int DONE_ID = 102;
    /** Swipe-away to headless boot: let the dying Activity process release
     *  the GPU first (two live WebGPU devices briefly would OOM small phones). */
    private static final long TAKEOVER_DELAY_MS = 8000;
    /** The handoff handshake: acked in 5 s (alive and working — the verdict
     *  then gets 90 s for a large checkpoint) or never (gone — settle from
     *  whatever the handoff files hold). */
    private static final long HANDOFF_ACK_MS = 5000;
    private static final long HANDOFF_VERDICT_MS = 90000;
    /** A boot that posts nothing (no status flip off 'booting') in this long
     *  is wedged: mark interrupted and stand down. */
    private static final long BOOT_WATCHDOG_MS = 180000;
    /** The Activity is alive if it beat in the last 15 s — the takeover and
     *  every resurrect refuse while it has. */
    private static final long ACTIVITY_FRESH_MS = 15000;
    /** A crashing boot must not resurrect-loop: 3 attempts per 10 minutes. */
    private static final int MAX_BOOTS = 3;
    private static final long BOOT_WINDOW_MS = 600000;

    private boolean foreground;
    private PowerManager.WakeLock wakeLock;
    // the last-shown progress, so the paused flag can rebuild the same
    // notification with its button flipped
    private String lastTitle = "";
    private String lastText = "";
    private int lastDone, lastTotal;
    private boolean lastPaused;

    private static boolean sSuffixSet;
    private Handler main;
    private WebView headless;
    private WebViewAssetLoader assetLoader;
    private String headlessRunId;
    private boolean headlessActive;
    private String handoffRunId;
    private boolean handoffAcked;
    private boolean handoffSettled;
    private int handoffGen;   // the pass timers only settle the pass that armed them
    private final Runnable bootWatchdog = () -> {
        if (!headlessActive || headlessRunId == null) return;
        if ("booting".equals(statusOf(headlessRunId))) {
            Log.w(TAG, "headless boot timed out for " + headlessRunId);
            markInterrupted(headlessRunId, "the background trainer never came up");
            teardown();
        }
    };
    /** The shade's Pause/Resume while headless: the Activity is gone, so the
     *  service turns the broadcast into a headless toggle itself. (When the
     *  Activity is alive its own receiver answers and this one stands down.) */
    private final BroadcastReceiver toggleReceiver = new BroadcastReceiver() {
        @Override public void onReceive(Context ctx, Intent intent) {
            if (headlessActive && headlessRunId != null) headlessEval("window.__headlessToggle&&window.__headlessToggle()");
        }
    };

    @Override public IBinder onBind(Intent intent) { return null; }

    @Override public void onCreate() {
        super.onCreate();
        main = new Handler(Looper.getMainLooper());
        try {
            IntentFilter toggleFilter = new IntentFilter(MainActivity.ACTION_TOGGLE_TRAINING);
            ContextCompat.registerReceiver(this, toggleReceiver, toggleFilter,
                    ContextCompat.RECEIVER_NOT_EXPORTED);
        } catch (Exception ignored) { /* shade control is best-effort */ }
    }

    @Override public int onStartCommand(Intent intent, int flags, int startId) {
        // A pressure-killed service restarts with a null intent (sticky): come
        // back only for a run this service owned — a locally-owned run belongs
        // to its Activity, and anything else lingers nothing.
        if (intent == null) {
            String active = RunFiles.readActive(this);
            if (active != null && resurrectIfOurs(active)) return START_STICKY;
            stopSelf();
            return START_NOT_STICKY;
        }
        String action = intent.getStringExtra("action");
        if (action == null) action = "";
        switch (action) {
            case "start":
                lastTitle = intent.getStringExtra("title");
                lastText = intent.getStringExtra("text");
                lastDone = 0; lastTotal = 0; lastPaused = false;
                startForegroundTraining(lastTitle, lastText);
                break;
            case "update":
                if (foreground) {
                    lastTitle = intent.getStringExtra("title");
                    lastText = intent.getStringExtra("text");
                    lastDone = intent.getIntExtra("done", 0);
                    lastTotal = intent.getIntExtra("total", 0);
                    showOngoing(lastTitle, lastText, lastDone, lastTotal, false);
                } else {
                    stopSelf(); // delivered to a dead service: linger nothing
                }
                break;
            case "paused":
                // the shade (or the app) flipped training state: keep the
                // foreground notification so the other button stays one tap
                // away, but only hold the CPU awake while running
                lastPaused = intent.getBooleanExtra("paused", false);
                if (foreground) {
                    if (lastPaused) releaseWakeLock(); else acquireWakeLock();
                    showOngoing(lastTitle, lastText, lastDone, lastTotal, false);
                } else {
                    stopSelf();
                }
                break;
            case "finish":
                showDone(intent.getStringExtra("title"), intent.getStringExtra("text"));
                teardown();
                break;
            case "headless-pause":
                headlessCommand(intent, "window.__headlessPause&&window.__headlessPause()");
                break;
            case "headless-resume":
                headlessCommand(intent, "window.__headlessResume&&window.__headlessResume()");
                break;
            case "headless-snapshot":
                headlessCommand(intent, "window.__headlessSnapshot&&window.__headlessSnapshot()");
                break;
            case "headless-ack": {
                String runId = intent.getStringExtra("runId");
                if (runId != null && runId.equals(handoffRunId) && !handoffSettled) {
                    handoffAcked = true;
                    final String id = runId;
                    final int gen = handoffGen;
                    main.postDelayed(() -> {
                        if (gen == handoffGen) settleHandoff(id, "the handoff verdict never came");
                    }, HANDOFF_VERDICT_MS);
                }
                break;
            }
            case "headless-ready": {
                String runId = intent.getStringExtra("runId");
                if (runId != null && runId.equals(handoffRunId) && !handoffSettled) {
                    if (intent.getBooleanExtra("ok", false)) {
                        // secured: boot after the GPU delay (the boot
                        // re-validates the marker, the files and the Activity
                        // heartbeat before touching anything)
                        handoffSettled = true;
                        final String id = runId;
                        main.postDelayed(() -> bootHeadlessNow(id), TAKEOVER_DELAY_MS);
                    } else {
                        settleHandoff(runId, "the handoff checkpoint failed");
                    }
                }
                break;
            }
            case "headless-ping": {
                // liveness probe for a service-owned live marker (the Start
                // guard): answered from Java, which no page throttle silences —
                // a frozen-but-alive twin still owns its run and must block.
                // Owned = the live twin, or a handoff still settling for this
                // run; anything else lingers nothing.
                String runId = intent.getStringExtra("runId");
                boolean owned = runId != null &&
                        ((headlessActive && runId.equals(headlessRunId)) ||
                                (!handoffSettled && runId.equals(handoffRunId)));
                if (owned) {
                    try {
                        JSONObject beat = new JSONObject();
                        beat.put("alive", true);
                        beat.put("at", System.currentTimeMillis());
                        RunFiles.writeJson(this, runId, "service.json", beat.toString());
                    } catch (Exception ignored) { /* the guard treats silence as dead */ }
                } else {
                    stopSelf(); // delivered to a dead service: linger nothing
                }
                break;
            }
            case "headless-stop":
                // never boots a trainer just to stop it: with no live twin the
                // status file is the whole story
                if (headlessActive) {
                    headlessEval("window.__headlessStop&&window.__headlessStop()");
                } else {
                    String runId = intent.getStringExtra("runId");
                    if (runId != null) markInterrupted(runId, "stopped from the app");
                    teardown();
                }
                break;
            default: // "cancel" and anything unknown: leave nothing behind
                teardown();
                break;
        }
        return START_STICKY;
    }

    @Override public void onTaskRemoved(Intent rootIntent) {
        if (headlessActive) {
            super.onTaskRemoved(rootIntent);
            return;   // already training headless — the swipe changes nothing
        }
        String active = RunFiles.readActive(this);
        if (active == null) {
            // no outlived run: training died with the WebView, so the
            // notification must not linger as a stale ongoing entry
            teardown();
            super.onTaskRemoved(rootIntent);
            return;
        }
        // a run outlived the Activity: ask the dying process for the pass —
        // a fresh file checkpoint it writes before it goes. Acked in 5 s or
        // never (settled from the files on disk either way: the Back-leave
        // copy arms exactly the swipe-during-reload case).
        handoffRunId = active;
        handoffAcked = false;
        handoffSettled = false;
        handoffGen++;
        Intent handoff = new Intent(MainActivity.ACTION_HANDOFF_TRAINING);
        handoff.setPackage(getPackageName());
        handoff.putExtra("runId", active);
        try { sendBroadcast(handoff); } catch (Exception ignored) { /* nobody home: the timer settles */ }
        final String runId = active;
        final int gen = handoffGen;
        main.postDelayed(() -> {
            if (gen == handoffGen && !handoffAcked) settleHandoff(runId, "the app was gone before the pass");
        }, HANDOFF_ACK_MS);
        super.onTaskRemoved(rootIntent);
    }

    @Override public void onDestroy() {
        try { unregisterReceiver(toggleReceiver); } catch (Exception ignored) { /* never registered */ }
        if (headlessActive && headlessRunId != null) {
            // destroyed mid-takeover (only a crash or a force-stop lands here;
            // orderly ends clear headlessActive first): leave a recoverable
            // status, never a live one with no trainer behind it
            markInterrupted(headlessRunId, "the background trainer was destroyed");
        }
        destroyHeadless();
        notifications().cancel(NOTIF_ID);
        releaseWakeLock();
        foreground = false;
        super.onDestroy();
    }

    // ── migrate-on-swipe: the headless trainer ──────────────────────────────

    /** A monitor command for the headless twin: evaluate now when live, else
     *  boot for a live marker and retry the (guarded, no-op-safe) call while
     *  the driver comes up. */
    private void headlessCommand(Intent intent, String script) {
        String runId = intent.getStringExtra("runId");
        if (headlessActive) {
            if (runId == null || runId.equals(headlessRunId)) headlessEval(script);
            return;
        }
        if (runId == null || !runId.equals(RunFiles.readActive(this))) return;
        if (!isLive(statusOf(runId))) return;
        bootHeadlessNow(runId);
        for (int i = 1; i <= 6; i++) main.postDelayed(() -> headlessEval(script), i * 10000L);
    }

    private void headlessEval(String script) {
        WebView w = headless;
        if (w == null) return;
        if (Looper.myLooper() == Looper.getMainLooper()) {
            try { w.evaluateJavascript(script, null); } catch (Exception ignored) { /* a racing boot */ }
        } else {
            main.post(() -> {
                try { w.evaluateJavascript(script, null); } catch (Exception ignored) { /* a racing boot */ }
            });
        }
    }

    /** The sticky-restart path: only a run this service owned comes back. */
    private boolean resurrectIfOurs(String runId) {
        String status = statusOf(runId);
        if (!isLive(status) || !"service".equals(ownerOf(runId)) || activityFresh(runId)) return false;
        bootHeadlessNow(runId);
        return true;
    }

    /** Boot the headless trainer for runId. Every precondition re-validates
     *  here (marker, handoff files, Activity silence, boot budget): the call
     *  sites are delayed and racing by nature. */
    private void bootHeadlessNow(String runId) {
        if (headlessActive) return;
        if (Build.VERSION.SDK_INT < 28) {
            // setDataDirectorySuffix needs Android 9+: older phones keep
            // today's behavior (the run interrupts on swipe-away)
            markInterrupted(runId, "background training needs Android 9 or newer");
            teardown();
            return;
        }
        if (runId == null || !runId.equals(RunFiles.readActive(this))) return;
        if (!isLive(statusOf(runId))) return;
        if (activityFresh(runId)) return;   // the Activity owns it — stand down
        if (!handoffReady(runId)) {
            markInterrupted(runId, "the swipe-away handoff was incomplete");
            teardown();
            return;
        }
        if (!bootBudget(runId)) {
            markInterrupted(runId, "the background trainer kept failing to start");
            teardown();
            return;
        }
        // exactly once per process: a repeat boot in a long-lived :training
        // process must not trip on its own suffix
        if (!sSuffixSet) {
            try {
                WebView.setDataDirectorySuffix("training");
                sSuffixSet = true;
            } catch (Exception e) {
                // another WebView already lives in this process (never
                // expected: nothing else runs in :training) — refuse rather
                // than corrupt it
                Log.e(TAG, "data dir suffix refused", e);
                markInterrupted(runId, "the background trainer could not start");
                teardown();
                return;
            }
        }
        // claimed BEFORE the load: races (a fast reopen's Continue-here, a
        // second task-removed) resolve against owner=service from here on
        markBooting(runId);
        try {
            File runsRoot = RunFiles.runsDir(this);
            File captureRoot = new File(getFilesDir(), "captures");
            if (!captureRoot.exists()) captureRoot.mkdirs();
            assetLoader = new WebViewAssetLoader.Builder()
                    .addPathHandler("/assets/", new WebViewAssetLoader.AssetsPathHandler(this))
                    .addPathHandler("/captures/", new WebViewAssetLoader.InternalStoragePathHandler(this, captureRoot))
                    .addPathHandler("/runs/", new WebViewAssetLoader.InternalStoragePathHandler(this, runsRoot))
                    .build();
            WebView w = new WebView(this);
            w.getSettings().setJavaScriptEnabled(true);
            w.getSettings().setDomStorageEnabled(true);
            w.getSettings().setAllowFileAccess(false);
            w.getSettings().setAllowContentAccess(true);
            w.getSettings().setMediaPlaybackRequiresUserGesture(false);
            w.addJavascriptInterface(new HeadlessBridge(), "AndroidHeadless");
            w.setWebViewClient(new WebViewClient() {
                @Override public android.webkit.WebResourceResponse shouldInterceptRequest(
                        WebView view, WebResourceRequest request) {
                    return assetLoader.shouldInterceptRequest(request.getUrl());
                }
                @Override public boolean shouldOverrideUrlLoading(WebView view, WebResourceRequest request) {
                    // the driver page itself loads; nothing ever navigates away from it
                    Uri uri = request.getUrl();
                    if ("https".equals(uri.getScheme()) && "appassets.androidplatform.net".equals(uri.getHost())) return false;
                    return true;
                }
            });
            w.setWebChromeClient(new WebChromeClient() {
                @Override public boolean onConsoleMessage(ConsoleMessage msg) {
                    Log.d(TAG, (msg != null ? msg.message() : "(null console message)"));
                    return true;
                }
            });
            headless = w;
            headlessRunId = runId;
            headlessActive = true;
            WebView.setWebContentsDebuggingEnabled(BuildConfig.DEBUG);
            w.loadUrl("https://appassets.androidplatform.net/assets/app/train-headless.html?run=" + runId);
            main.postDelayed(bootWatchdog, BOOT_WATCHDOG_MS);
            Log.i(TAG, "headless trainer booting for " + runId);
        } catch (Exception e) {
            Log.e(TAG, "headless boot failed", e);
            markInterrupted(runId, "the background trainer could not start");
            teardown();
        }
    }

    private void destroyHeadless() {
        headlessActive = false;
        headlessRunId = null;
        try {
            if (main != null) main.removeCallbacks(bootWatchdog);
        } catch (Exception ignored) { /* never armed */ }
        WebView w = headless;
        headless = null;
        if (w == null) return;
        Runnable kill = () -> {
            try { w.stopLoading(); } catch (Exception ignored) { }
            try { w.loadUrl("about:blank"); } catch (Exception ignored) { }
            try { w.destroy(); } catch (Exception ignored) { }
        };
        if (Looper.myLooper() == Looper.getMainLooper()) kill.run();
        else main.post(kill);
    }

    /** Settle the pass: boot the twin off whatever the handoff files hold
     *  (the fresh handoff copy, or the Back-leave copy when the app died
     *  mid-reload), else interrupt with a summary. Once only; a stale settle
     *  (marker moved on, Activity alive and training) stands down. */
    private void settleHandoff(String runId, String why) {
        if (handoffSettled) return;
        handoffSettled = true;
        if (runId == null || !runId.equals(RunFiles.readActive(this))) return;
        if (activityFresh(runId)) return;   // the Activity owns it — stand down
        if (handoffReady(runId)) {
            final String id = runId;
            main.postDelayed(() -> bootHeadlessNow(id), TAKEOVER_DELAY_MS);
            return;
        }
        markInterrupted(runId, why);
        showDone("Training interrupted", "The background pass failed — reopen to continue from the last save.");
        teardown();
    }

    /** The takeover preconditions: a live status, staged photos and a state
     *  checkpoint from the pass (or the Back-leave copy). Nothing to resume
     *  from (resume-from-state only, V1) interrupts instead. */
    private boolean handoffReady(String runId) {
        try {
            String photos = RunFiles.readJson(this, runId, "photos.json");
            if (photos == null || !new JSONObject(photos).optBoolean("complete", false)) return false;
            File state = RunFiles.runFile(this, runId, "state.bin");
            if (state == null || !state.isFile() || state.length() < 16) return false;
            return RunFiles.readJson(this, runId, "recon.json") != null;
        } catch (Exception e) {
            return false;
        }
    }

    private boolean activityFresh(String runId) {
        try {
            String beat = RunFiles.readJson(this, runId, "activity.json");
            if (beat == null) return false;
            JSONObject o = new JSONObject(beat);
            return o.optBoolean("alive", false) &&
                    System.currentTimeMillis() - o.optLong("at", 0) < ACTIVITY_FRESH_MS;
        } catch (Exception e) {
            return false;
        }
    }

    /** The resurrect budget: boots.json counts attempts inside a 10-minute
     *  window. False spends nothing (the caller interrupts instead). */
    private boolean bootBudget(String runId) {
        try {
            long now = System.currentTimeMillis();
            int count = 0;
            long first = now;
            String raw = RunFiles.readJson(this, runId, "boots.json");
            if (raw != null) {
                JSONObject o = new JSONObject(raw);
                count = o.optInt("count", 0);
                first = o.optLong("first", now);
                if (now - first > BOOT_WINDOW_MS) {
                    count = 0;
                    first = now;
                }
            }
            if (count >= MAX_BOOTS) return false;
            JSONObject next = new JSONObject();
            next.put("count", count + 1);
            next.put("first", first);
            return RunFiles.writeJson(this, runId, "boots.json", next.toString());
        } catch (Exception e) {
            return false;
        }
    }

    private String statusFile(String runId, String key) {
        try {
            String raw = RunFiles.readJson(this, runId, "status.json");
            if (raw == null) return "";
            return new JSONObject(raw).optString(key, "");
        } catch (Exception e) {
            return "";
        }
    }

    private String statusOf(String runId) { return statusFile(runId, "status"); }
    private String ownerOf(String runId) { return statusFile(runId, "owner"); }

    private static boolean isLive(String status) {
        return "staging".equals(status) || "booting".equals(status) ||
                "active".equals(status) || "paused".equals(status);
    }

    /** Flip to booting/owner=service, keeping the last-known iter. */
    private void markBooting(String runId) {
        writeStatus(runId, "booting", "service", -1, null);
    }

    /** Flip a LIVE status to interrupted (finished and already-dead statuses
     *  are never touched). */
    private void markInterrupted(String runId, String error) {
        if (!isLive(statusOf(runId))) return;
        writeStatus(runId, "interrupted", "service", -1, error);
    }

    private void writeStatus(String runId, String status, String owner, int iter, String error) {
        try {
            int keep = iter;
            if (keep < 0) {
                try {
                    String raw = RunFiles.readJson(this, runId, "status.json");
                    if (raw != null) keep = new JSONObject(raw).optInt("iter", 0);
                } catch (Exception ignored) { keep = 0; }
            }
            JSONObject o = new JSONObject();
            o.put("status", status);
            o.put("owner", owner);
            o.put("iter", Math.max(0, keep));
            o.put("at", System.currentTimeMillis());
            if (error != null) o.put("error", error);
            RunFiles.writeJson(this, runId, "status.json", o.toString());
        } catch (Exception ignored) { /* status is best-effort */ }
    }

    // ── the headless page's bridge ──────────────────────────────────────────
    private final class HeadlessBridge {
        @JavascriptInterface public void log(String msg) { Log.d(TAG, msg == null ? "(null)" : msg); }

        /** Live progress from the driver: the files-dir copy for the monitor,
         *  the heartbeat, and the shade notification. */
        @JavascriptInterface public void postProgress(String runId, String json) {
            if (runId == null || json == null) return;
            try {
                RunFiles.writeJson(TrainingService.this, runId, "progress.json", json);
                JSONObject beat = new JSONObject();
                beat.put("alive", true);
                beat.put("at", System.currentTimeMillis());
                RunFiles.writeJson(TrainingService.this, runId, "service.json", beat.toString());
                if (!foreground) return;
                JSONObject p = new JSONObject(json);
                lastTitle = p.optString("title", lastTitle);
                lastText = p.optString("text", lastText);
                lastDone = p.optInt("done", lastDone);
                lastTotal = p.optInt("total", lastTotal);
                boolean training = p.optBoolean("training", !lastPaused);
                if (training == lastPaused) {
                    lastPaused = !training;
                    if (lastPaused) releaseWakeLock(); else acquireWakeLock();
                }
                showOngoing(lastTitle, lastText, lastDone, lastTotal, false);
            } catch (Exception ignored) { /* progress display must never crash training */ }
        }

        @JavascriptInterface public boolean writeRunJson(String runId, String name, String json) {
            return RunFiles.writeJson(TrainingService.this, runId, name, json);
        }

        @JavascriptInterface public boolean writeRunChunk(String runId, String name, String base64, boolean last) {
            return RunFiles.writeChunk(TrainingService.this, runId, name, base64, last);
        }

        @JavascriptInterface public boolean writeSnapshotChunk(String runId, int w, int h, String base64, boolean last) {
            return RunFiles.writeSnapshotChunk(TrainingService.this, runId, w, h, base64, last);
        }

        // The three orderly ends: the driver secured the status file first, so
        // the service only posts the summary and stands down.
        @JavascriptInterface public void finishRun(String runId, String title, String text) {
            endRun(title, text);
        }

        @JavascriptInterface public void stopRun(String runId, String title, String text) {
            endRun(title, text);
        }

        @JavascriptInterface public void abortRun(String runId, String title, String text) {
            endRun(title, text);
        }

        private void endRun(String title, String text) {
            showDone(title, text);
            main.post(() -> teardown());
        }
    }

    // ── the notification (unchanged job) ────────────────────────────────────

    private void startForegroundTraining(String title, String text) {
        ensureChannel();
        acquireWakeLock();
        Notification notification = buildNotification(title, text, 0, 0, true);
        // On-device model training is not one of the platform's fixed
        // foreground-service types, so this service runs as specialUse on
        // Android 14+ (declared in the manifest with its subtype property).
        try {
            if (Build.VERSION.SDK_INT >= 34) {
                startForeground(NOTIF_ID, notification, ServiceInfo.FOREGROUND_SERVICE_TYPE_SPECIAL_USE);
            } else {
                startForeground(NOTIF_ID, notification);
            }
        } catch (Exception e) {
            // The platform refused the foreground start (a background launch
            // on Android 12+, a missing declaration): stop quietly instead of
            // crashing — training itself runs in the WebView either way.
            teardown();
            return;
        }
        foreground = true;
    }

    private void showOngoing(String title, String text, int done, int total, boolean alert) {
        try {
            notifications().notify(NOTIF_ID, buildNotification(title, text, done, total, true));
        } catch (Exception ignored) { /* progress display must never crash training */ }
    }

    private void showDone(String title, String text) {
        // Finished training leaves one dismissible summary in the shade —
        // unless notifications were denied, where posting throws a
        // SecurityException and takes the app down with it (Stop & keep
        // crashed exactly here on permission-denied devices).
        if (Build.VERSION.SDK_INT >= 33 && checkSelfPermission(Manifest.permission.POST_NOTIFICATIONS)
                != PackageManager.PERMISSION_GRANTED) return;
        try {
            notifications().notify(DONE_ID, buildNotification(title, text, 0, 0, false));
        } catch (Exception ignored) { /* a summary must never crash the app */ }
    }

    private void teardown() {
        handoffSettled = true;
        destroyHeadless();
        try { notifications().cancel(NOTIF_ID); } catch (Exception ignored) { /* already gone */ }
        try { stopForeground(true); } catch (Exception ignored) { /* never foregrounded */ }
        releaseWakeLock();
        foreground = false;
        stopSelf();
    }

    private NotificationManager notifications() {
        return (NotificationManager) getSystemService(NOTIFICATION_SERVICE);
    }

    private void ensureChannel() {
        if (Build.VERSION.SDK_INT < 26) return;
        NotificationChannel channel = new NotificationChannel(
                CHANNEL_ID, "Training progress", NotificationManager.IMPORTANCE_LOW);
        channel.setDescription("Live score and splat count while a model trains.");
        notifications().createNotificationChannel(channel);
    }

    private Notification buildNotification(String title, String text, int done, int total, boolean ongoing) {
        Notification.Builder builder = Build.VERSION.SDK_INT >= 26
                ? new Notification.Builder(this, CHANNEL_ID)
                : new Notification.Builder(this);
        Intent open = new Intent(this, MainActivity.class)
                .setAction(Intent.ACTION_MAIN)
                .addCategory(Intent.CATEGORY_LAUNCHER);
        PendingIntent tap = PendingIntent.getActivity(this, 0, open,
                PendingIntent.FLAG_UPDATE_CURRENT | PendingIntent.FLAG_IMMUTABLE);
        builder.setSmallIcon(R.mipmap.icon180)
                .setContentTitle(title == null || title.isEmpty() ? "Training model" : title)
                .setContentText(text == null ? "" : text)
                .setContentIntent(tap)
                .setOngoing(ongoing)
                .setAutoCancel(!ongoing)
                .setOnlyAlertOnce(true);
        if (Build.VERSION.SDK_INT < 26) builder.setPriority(Notification.PRIORITY_LOW);
        if (ongoing && total > 0 && done >= 0) builder.setProgress(total, Math.min(done, total), false);
        if (ongoing) {
            // Pause / Resume without opening the app: a broadcast the
            // activity turns into a WebView toggle call while the app lives,
            // and this service turns into a headless toggle once it owns the
            // run — whichever side is alive answers, the other stands down
            Intent toggle = new Intent(MainActivity.ACTION_TOGGLE_TRAINING);
            toggle.setPackage(getPackageName());
            PendingIntent togglePi = PendingIntent.getBroadcast(this, 1, toggle,
                    PendingIntent.FLAG_UPDATE_CURRENT | PendingIntent.FLAG_IMMUTABLE);
            builder.addAction(lastPaused ? android.R.drawable.ic_media_play : android.R.drawable.ic_media_pause,
                    lastPaused ? "Resume" : "Pause", togglePi);
            if (lastPaused && text != null && !text.isEmpty()) builder.setContentText(text + " · Paused");
        }
        return builder.build();
    }

    private void acquireWakeLock() {
        try {
            if (wakeLock == null) {
                PowerManager pm = (PowerManager) getSystemService(POWER_SERVICE);
                wakeLock = pm.newWakeLock(PowerManager.PARTIAL_WAKE_LOCK, "SplatXR:training");
                wakeLock.setReferenceCounted(false);
            }
            if (!wakeLock.isHeld()) wakeLock.acquire();
        } catch (Exception ignored) { /* the service still raises our priority */ }
    }

    private void releaseWakeLock() {
        try {
            if (wakeLock != null && wakeLock.isHeld()) wakeLock.release();
        } catch (Exception ignored) { /* already released */ }
    }
}
