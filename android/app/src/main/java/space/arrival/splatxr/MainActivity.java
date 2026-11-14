package space.arrival.splatxr;

import android.app.Activity;
import android.Manifest;
import android.content.BroadcastReceiver;
import android.content.Context;
import android.content.Intent;
import android.content.IntentFilter;
import android.content.pm.PackageManager;
import android.util.Base64;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.view.WindowInsets;
import android.webkit.JavascriptInterface;
import android.webkit.WebResourceRequest;
import android.webkit.WebChromeClient;
import android.webkit.ValueCallback;
import android.webkit.WebView;
import android.webkit.WebViewClient;
import android.widget.FrameLayout;

import androidx.core.content.ContextCompat;
import androidx.webkit.WebViewAssetLoader;

import org.json.JSONArray;
import org.json.JSONObject;

import java.io.File;
import java.io.IOException;
import java.io.FileOutputStream;
import java.io.OutputStream;
import java.util.HashMap;
import java.util.Map;

public final class MainActivity extends Activity {
    private static final int REQUEST_CAPTURE = 41;
    private static final int REQUEST_FILE_PICKER = 42;
    private static final int REQUEST_SAVE_FILE = 43;
    private static final int REQUEST_NOTIFY_PERMISSION = 44;
    /** The training notification's Pause/Resume button: the service fires it,
     *  the activity turns it into a WebView toggle call. */
    public static final String ACTION_TOGGLE_TRAINING = "space.arrival.splatxr.TOGGLE_TRAINING";
    /** Notification/app foreground: return service-owned training to this Activity. */
    public static final String ACTION_RETURN_TRAINING = "space.arrival.splatxr.RETURN_TRAINING";
    /** The background handoff: the foreground service fires it when this Activity
     *  stops (or its task is removed), the activity acks and checkpoints — the
     *  verdict follows as headless-ready (or the service times out). */
    public static final String ACTION_HANDOFF_TRAINING = "space.arrival.splatxr.HANDOFF_TRAINING";
    private ValueCallback<Uri[]> fileChooserCallback;
    private WebView webView;
    private WebViewAssetLoader assetLoader;
    private final Object fileSaveLock = new Object();
    private OutputStream fileSaveStream;
    private boolean fileSavePickerPending;
    /** Suppress onStop's normal Activity→service handoff while replacing a
     *  renderer whose WebGPU process was lost during service return. */
    private boolean gpuRendererRestart;
    private final BroadcastReceiver toggleTrainingReceiver = new BroadcastReceiver() {
        @Override public void onReceive(Context ctx, Intent intent) {
            if (webView == null) return;
            webView.post(() -> {
                try { webView.evaluateJavascript("window.__splatxrNotifToggle&&window.__splatxrNotifToggle()", null); }
                catch (Exception ignored) { /* the shade toggle must never crash the app */ }
            });
        }
    };
    private final BroadcastReceiver handoffReceiver = new BroadcastReceiver() {
        @Override public void onReceive(Context ctx, Intent intent) {
            // ack FIRST (the service stops its absence timer), then run the
            // page's handoff — the verdict follows as headless-ready
            try {
                startService(new Intent(MainActivity.this, TrainingService.class)
                        .putExtra("action", "headless-ack")
                        .putExtra("runId", intent.getStringExtra("runId")));
            } catch (Exception ignored) { /* the service times out instead */ }
            if (webView == null) return;
            webView.post(() -> {
                try { webView.evaluateJavascript("window.__splatxrHandoff&&window.__splatxrHandoff()", null); }
                catch (Exception ignored) { /* the service times out instead */ }
            });
        }
    };

    @Override protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        getWindow().setStatusBarColor(android.graphics.Color.rgb(14, 17, 18));
        getWindow().setNavigationBarColor(android.graphics.Color.rgb(14, 17, 18));
        getWindow().getDecorView().setSystemUiVisibility(0);

        File captureRoot = new File(getFilesDir(), "captures");
        if (!captureRoot.exists()) captureRoot.mkdirs();
        File runsRoot = RunFiles.runsDir(this);
        assetLoader = new WebViewAssetLoader.Builder()
                .addPathHandler("/assets/", new WebViewAssetLoader.AssetsPathHandler(this))
                .addPathHandler("/captures/", new WebViewAssetLoader.InternalStoragePathHandler(this, captureRoot))
                .addPathHandler("/runs/", new WebViewAssetLoader.InternalStoragePathHandler(this, runsRoot))
                .build();

        webView = new WebView(this);
        webView.setBackgroundColor(android.graphics.Color.rgb(14, 17, 18));
        webView.getSettings().setJavaScriptEnabled(true);
        webView.getSettings().setDomStorageEnabled(true);
        webView.getSettings().setAllowFileAccess(false);
        webView.getSettings().setAllowContentAccess(true);
        webView.getSettings().setMediaPlaybackRequiresUserGesture(false);
        webView.addJavascriptInterface(new AndroidARBridge(), "AndroidARBridge");
        webView.setWebViewClient(new WebViewClient() {
            @Override public android.webkit.WebResourceResponse shouldInterceptRequest(WebView view, WebResourceRequest request) {
                return assetLoader.shouldInterceptRequest(request.getUrl());
            }
            @Override public boolean shouldOverrideUrlLoading(WebView view, WebResourceRequest request) {
                Uri uri = request.getUrl();
                if ("https".equals(uri.getScheme()) && "appassets.androidplatform.net".equals(uri.getHost())) return false;
                if ("https".equals(uri.getScheme()) || "http".equals(uri.getScheme())) {
                    try { startActivity(new Intent(Intent.ACTION_VIEW, uri)); } catch (Exception ignored) { }
                }
                return true;
            }
        });
        webView.setWebChromeClient(new WebChromeClient() {
            @Override public boolean onShowFileChooser(WebView view, ValueCallback<Uri[]> callback, FileChooserParams params) {
                if (fileChooserCallback != null) fileChooserCallback.onReceiveValue(null);
                fileChooserCallback = callback;
                Intent picker = params.createIntent();
                try { startActivityForResult(picker, REQUEST_FILE_PICKER); return true; }
                catch (Exception e) { fileChooserCallback = null; callback.onReceiveValue(null); return false; }
            }
        });
        FrameLayout host = new FrameLayout(this);
        setContentView(host);
        host.addView(webView, new FrameLayout.LayoutParams(
                FrameLayout.LayoutParams.MATCH_PARENT, FrameLayout.LayoutParams.MATCH_PARENT));
        applySafeAreaInsets(host);
        WebView.setWebContentsDebuggingEnabled(BuildConfig.DEBUG);
        webView.loadUrl("https://appassets.androidplatform.net/assets/app/index.html");

        // the notification's Pause/Resume button (a dynamic receiver: nothing
        // to declare, nothing exported). ContextCompat carries the
        // NOT_EXPORTED flag back to minSdk — a bare registerReceiver fails
        // Android 14+ lint (UnspecifiedRegisterReceiverFlag) on every path.
        try {
            IntentFilter toggleFilter = new IntentFilter(ACTION_TOGGLE_TRAINING);
            ContextCompat.registerReceiver(this, toggleTrainingReceiver, toggleFilter,
                    ContextCompat.RECEIVER_NOT_EXPORTED);
        } catch (Exception ignored) { /* shade control is best-effort */ }
        try {
            IntentFilter handoffFilter = new IntentFilter(ACTION_HANDOFF_TRAINING);
            ContextCompat.registerReceiver(this, handoffReceiver, handoffFilter,
                    ContextCompat.RECEIVER_NOT_EXPORTED);
        } catch (Exception ignored) { /* the handoff falls back to the files on disk */ }
    }

    private void applySafeAreaInsets(FrameLayout content) {
        final int baseLeft = content.getPaddingLeft();
        final int baseTop = content.getPaddingTop();
        final int baseRight = content.getPaddingRight();
        final int baseBottom = content.getPaddingBottom();
        content.setOnApplyWindowInsetsListener((view, insets) -> {
            int left, top, right, bottom;
            if (Build.VERSION.SDK_INT >= 30) {
                android.graphics.Insets bars = insets.getInsets(
                        WindowInsets.Type.systemBars() | WindowInsets.Type.displayCutout());
                left = bars.left; top = bars.top; right = bars.right; bottom = bars.bottom;
            } else {
                left = insets.getSystemWindowInsetLeft();
                top = insets.getSystemWindowInsetTop();
                right = insets.getSystemWindowInsetRight();
                bottom = insets.getSystemWindowInsetBottom();
            }
            view.setPadding(baseLeft + left, baseTop + top,
                    baseRight + right, baseBottom + bottom);
            return insets;
        });
        content.requestApplyInsets();
    }

    @Override protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (requestCode == REQUEST_FILE_PICKER) {
            if (fileChooserCallback != null) {
                Uri[] values = null;
                if (resultCode == RESULT_OK && data != null) {
                    if (data.getClipData() != null) {
                        android.content.ClipData clip = data.getClipData();
                        values = new Uri[clip.getItemCount()];
                        for (int i = 0; i < clip.getItemCount(); i++) values[i] = clip.getItemAt(i).getUri();
                    } else if (data.getData() != null) values = new Uri[]{data.getData()};
                }
                fileChooserCallback.onReceiveValue(values);
                fileChooserCallback = null;
            }
        } else if (requestCode == REQUEST_SAVE_FILE) {
            if (resultCode != RESULT_OK || data == null || data.getData() == null) {
                synchronized (fileSaveLock) { fileSavePickerPending = false; }
                dispatchFileSaveEvent("cancelled", null);
                return;
            }
            try {
                OutputStream output = getContentResolver().openOutputStream(data.getData(), "w");
                if (output == null) throw new IOException("Could not open the selected destination.");
                synchronized (fileSaveLock) {
                    fileSaveStream = output;
                    fileSavePickerPending = false;
                }
                dispatchFileSaveEvent("ready", null);
            } catch (Exception e) {
                synchronized (fileSaveLock) { fileSavePickerPending = false; }
                dispatchFileSaveEvent("error", e.getMessage());
            }
        } else if (requestCode == REQUEST_CAPTURE && resultCode == RESULT_OK && webView != null) {
            String captureId = data == null ? "" : data.getStringExtra("captureId");
            String script = "window.dispatchEvent(new CustomEvent('splatxr-native-capture',{detail:{captureId:" +
                    JSONObject.quote(captureId == null ? "" : captureId) + "}}))";
            webView.post(() -> webView.evaluateJavascript(script, null));
        }
    }

    private void startFileSave(String name, String mimeType) {
        synchronized (fileSaveLock) {
            if (fileSavePickerPending || fileSaveStream != null) {
                dispatchFileSaveEvent("error", "Another file save is already in progress.");
                return;
            }
            fileSavePickerPending = true;
        }
        Intent picker = new Intent(Intent.ACTION_CREATE_DOCUMENT);
        picker.addCategory(Intent.CATEGORY_OPENABLE);
        picker.setType(mimeType == null || mimeType.isEmpty() ? "application/octet-stream" : mimeType);
        picker.putExtra(Intent.EXTRA_TITLE, safeFileName(name));
        try {
            startActivityForResult(picker, REQUEST_SAVE_FILE);
        } catch (Exception e) {
            synchronized (fileSaveLock) { fileSavePickerPending = false; }
            dispatchFileSaveEvent("error", e.getMessage());
        }
    }

    private static String safeFileName(String name) {
        String safe = name == null ? "splat-export.bin" : name;
        safe = safe.replace('/', '_').replace('\\', '_').replaceAll("[\\r\\n\\p{Cntrl}]", "_").trim();
        return safe.isEmpty() ? "splat-export.bin" : safe;
    }

    private boolean writeFileSaveChunk(String encoded) {
        synchronized (fileSaveLock) {
            if (fileSaveStream == null) return false;
            try {
                fileSaveStream.write(Base64.decode(encoded, Base64.DEFAULT));
                return true;
            } catch (Exception e) {
                try { fileSaveStream.close(); } catch (Exception ignored) { }
                fileSaveStream = null;
                dispatchFileSaveEvent("error", e.getMessage());
                return false;
            }
        }
    }

    private void finishFileSave() {
        OutputStream output;
        synchronized (fileSaveLock) {
            output = fileSaveStream;
            fileSaveStream = null;
        }
        if (output == null) {
            dispatchFileSaveEvent("error", "No destination is open for this export.");
            return;
        }
        try {
            output.close();
            dispatchFileSaveEvent("complete", null);
        } catch (Exception e) {
            dispatchFileSaveEvent("error", e.getMessage());
        }
    }

    private void cancelFileSave() {
        OutputStream output;
        synchronized (fileSaveLock) {
            output = fileSaveStream;
            fileSaveStream = null;
            fileSavePickerPending = false;
        }
        if (output != null) try { output.close(); } catch (Exception ignored) { }
    }

    private void dispatchFileSaveEvent(String type, String message) {
        WebView view = webView;
        if (view == null) return;
        JSONObject detail = new JSONObject();
        try {
            detail.put("type", type);
            if (message != null) detail.put("message", message);
        } catch (Exception ignored) { }
        String script = "window.dispatchEvent(new CustomEvent('splatxr-native-file-save',{detail:" + detail + "}));";
        view.post(() -> view.evaluateJavascript(script, null));
    }

    private void sendTrainingCommand(String action, String title, String text, int done, int total) {
        try {
            Intent intent = new Intent(this, TrainingService.class)
                    .putExtra("action", action)
                    .putExtra("title", title == null ? "" : title)
                    .putExtra("text", text == null ? "" : text)
                    .putExtra("done", done)
                    .putExtra("total", total);
            // Only "start" raises a foreground service. Updates, the finish
            // summary and cancels go to the already-foreground service with
            // a plain start — if it died mid-training they no-op instead of
            // hanging a foreground start the service never answers (which
            // the platform kills the app for).
            if ("start".equals(action) && Build.VERSION.SDK_INT >= 26) startForegroundService(intent);
            else startService(intent);
        } catch (Exception ignored) { /* best-effort progress display */ }
    }

    /** Flip the notification's Pause/Resume button (the service stays up so
     *  the other button is always one tap away in the shade). */
    private void sendTrainingPaused(boolean paused) {
        try {
            Intent intent = new Intent(this, TrainingService.class)
                    .putExtra("action", "paused")
                    .putExtra("paused", paused);
            // the service is already foreground (start raised it): a plain
            // start, like updates — never a second foreground start
            startService(intent);
        } catch (Exception ignored) { /* shade control is best-effort */ }
    }

    private void stopTrainingService() {
        try { stopService(new Intent(this, TrainingService.class)); }
        catch (Exception ignored) { /* the service was never started */ }
    }

    /** System Back goes through the page's history first: Back closes pushed UI
     *  layers and checkpoints out of a live run (the page's popstate leave —
     *  same effect as the header Back) instead of killing the app. Only with
     *  no history left does Back finish the activity as before. */
    @SuppressWarnings("deprecation")
    @Override public void onBackPressed() {
        if (webView != null) {
            try {
                if (webView.canGoBack()) {
                    webView.goBack();
                    return;
                }
            } catch (Exception ignored) { /* fall through to the default */ }
        }
        super.onBackPressed();
    }

    /** Move a live run into the foreground service whenever this Activity is
     *  genuinely backgrounded. Previously the service owned only the ongoing
     *  notification until onTaskRemoved; Home/app-switch therefore left the
     *  actual GPU loop in a killable background WebView. */
    @Override protected void onStop() {
        super.onStop();
        if (isChangingConfigurations() || gpuRendererRestart) return;
        String active = RunFiles.readActive(this);
        if (active == null) return;
        try {
            startService(new Intent(this, TrainingService.class)
                    .putExtra("action", "activity-background")
                    .putExtra("runId", active));
        } catch (Exception ignored) { /* the already-foreground service still has swipe fallback */ }
    }

    /** Returning quickly may cancel only the service's debounce. If its
     *  trainer already owns the run, leave it untouched until Continue here. */
    @Override protected void onStart() {
        super.onStart();
        notifyActivityVisible();
    }

    /** A notification tap only brings the monitor to the front. It must not
     *  stop the healthy service trainer behind the user's back: ownership moves
     *  only after the user presses Continue here in that monitor. */
    @Override protected void onNewIntent(Intent intent) {
        super.onNewIntent(intent);
        setIntent(intent);
        notifyActivityVisible();
        if (intent != null && ACTION_RETURN_TRAINING.equals(intent.getAction()) && webView != null) {
            webView.post(() -> {
                try { webView.evaluateJavascript(
                        "window.__splatxrOpenMonitor&&window.__splatxrOpenMonitor()", null); }
                catch (Exception ignored) { /* boot already checks the durable marker */ }
            });
        }
    }

    private void notifyActivityVisible() {
        String active = RunFiles.readActive(this);
        if (active == null) return;
        try {
            startService(new Intent(this, TrainingService.class)
                    .putExtra("action", "activity-visible")
                    .putExtra("runId", active));
        } catch (Exception ignored) { /* the monitor reads the durable files */ }
    }

    @Override protected void onDestroy() {
        try { unregisterReceiver(toggleTrainingReceiver); } catch (Exception ignored) { /* never registered */ }
        try { unregisterReceiver(handoffReceiver); } catch (Exception ignored) { /* never registered */ }
        if (RunFiles.readActive(this) == null) stopTrainingService();
        cancelFileSave();
        if (fileChooserCallback != null) {
            fileChooserCallback.onReceiveValue(null);
            fileChooserCallback = null;
        }
        // Explicitly kill the renderer: Activity recreation alone can leave a
        // lost WebGPU WebView attached until GC, and the replacement then
        // inherits the same poisoned graphics process.
        WebView oldWebView = webView;
        webView = null;
        if (oldWebView != null) {
            try { oldWebView.stopLoading(); } catch (Exception ignored) { }
            try { oldWebView.removeJavascriptInterface("AndroidARBridge"); } catch (Exception ignored) { }
            try { oldWebView.loadUrl("about:blank"); } catch (Exception ignored) { }
            try { oldWebView.destroy(); } catch (Exception ignored) { }
        }
        super.onDestroy();
    }

    private final class AndroidARBridge {
        private final Map<String, OutputStream> captureImports = new HashMap<>();

        @JavascriptInterface public boolean isAvailable() { return true; }

        @JavascriptInterface public void startCapture() {
            runOnUiThread(() -> startActivityForResult(new Intent(MainActivity.this, NativeCaptureActivity.class), REQUEST_CAPTURE));
        }

        @JavascriptInterface public void captureKeyframe() {
            NativeCaptureActivity.requestKeyframe();
        }

        @JavascriptInterface public void beginFileSave(String name, String mimeType) {
            runOnUiThread(() -> startFileSave(name, mimeType));
        }

        @JavascriptInterface public boolean writeFileChunk(String base64) {
            return writeFileSaveChunk(base64);
        }

        @JavascriptInterface public void finishFileSave() {
            MainActivity.this.finishFileSave();
        }

        @JavascriptInterface public void cancelFileSave() {
            MainActivity.this.cancelFileSave();
        }

        @JavascriptInterface public void startTrainingNotification(String title, String text) {
            runOnUiThread(() -> {
                if (Build.VERSION.SDK_INT >= 33 &&
                        checkSelfPermission(Manifest.permission.POST_NOTIFICATIONS) != PackageManager.PERMISSION_GRANTED) {
                    requestPermissions(new String[]{Manifest.permission.POST_NOTIFICATIONS}, REQUEST_NOTIFY_PERMISSION);
                }
                sendTrainingCommand("start", title, text, 0, 0);
            });
        }

        @JavascriptInterface public void updateTrainingNotification(String title, String text, int done, int total) {
            sendTrainingCommand("update", title, text, done, total);
        }

        @JavascriptInterface public void finishTrainingNotification(String title, String text) {
            sendTrainingCommand("finish", title, text, 0, 0);
        }

        @JavascriptInterface public void cancelTrainingNotification() {
            sendTrainingCommand("cancel", null, null, 0, 0);
        }

        @JavascriptInterface public void setTrainingPaused(boolean paused) {
            sendTrainingPaused(paused);
        }

        // ── migrate-on-swipe: the files-dir bus (RunFiles). Small atomic
        // JSON writes, chunked large writes, the active marker, the native
        // photo retain, and commands to the headless twin. Reads go through
        // the /runs/ path handler as plain fetches, not through here.
        @JavascriptInterface public boolean writeRunJson(String runId, String name, String json) {
            return RunFiles.writeJson(MainActivity.this, runId, name, json);
        }

        @JavascriptInterface public boolean writeRunChunk(String runId, String name, String base64, boolean last) {
            return RunFiles.writeChunk(MainActivity.this, runId, name, base64, last);
        }

        @JavascriptInterface public boolean setActiveRun(String runId) {
            return RunFiles.writeActive(MainActivity.this, runId);
        }

        @JavascriptInterface public boolean clearActiveRun(String runId) {
            return RunFiles.clearActive(MainActivity.this, runId);
        }

        /** Restore a transaction marker lost by an older handback build. */
        @JavascriptInterface public String recoverActiveRun() {
            String runId = RunFiles.recoverActive(MainActivity.this);
            return runId == null ? "" : runId;
        }

        /** All complete Android-file runs, for rebuilding Home after IDB eviction. */
        @JavascriptInterface public String listRecoverableRuns() {
            return new JSONArray(RunFiles.listDurableRuns(MainActivity.this)).toString();
        }

        /** Replace the visible WebView after a handback device loss. The run's
         *  durable marker must still match, and onStop must not bounce it back
         *  into the service while this renderer-only restart is in flight. */
        @JavascriptInterface public void restartForGpuRecovery(String runId) {
            if (!RunFiles.validRunId(runId) || !runId.equals(RunFiles.readActive(MainActivity.this))) return;
            runOnUiThread(() -> {
                gpuRendererRestart = true;
                recreate();
            });
        }

        /** The handoff verdict: the page secured the run to files (or failed
         *  to) — the service boots the headless twin off it, or settles. */
        @JavascriptInterface public void handoffDone(String runId, boolean ok) {
            if (runId == null || !RunFiles.validRunId(runId)) return;
            try {
                startService(new Intent(MainActivity.this, TrainingService.class)
                        .putExtra("action", "headless-ready")
                        .putExtra("runId", runId)
                        .putExtra("ok", ok));
            } catch (Exception ignored) { /* the service times out instead */ }
        }

        /** A command to the headless twin (pause/resume/stop/snapshot/prune/grow/ping). Plain
         *  start: the service is already foreground while a run lives. */
        @JavascriptInterface public void headlessCommand(String action, String runId) {
            if (action == null || runId == null) return;
            if (!action.equals("headless-pause") && !action.equals("headless-resume") &&
                    !action.equals("headless-stop") && !action.equals("headless-snapshot") &&
                    !action.equals("headless-prune") && !action.equals("headless-grow") &&
                    !action.equals("headless-ping")) return;
            if (!RunFiles.validRunId(runId)) return;
            try {
                startService(new Intent(MainActivity.this, TrainingService.class)
                        .putExtra("action", action)
                        .putExtra("runId", runId));
            } catch (Exception ignored) { /* the twin is gone; the files say so */ }
        }

        /** Live service GPU duty, as a validated whole percentage. */
        @JavascriptInterface public void headlessGpu(String runId, int percent) {
            if (!RunFiles.validRunId(runId) || percent < 10 || percent > 100) return;
            try {
                startService(new Intent(MainActivity.this, TrainingService.class)
                        .putExtra("action", "headless-gpu")
                        .putExtra("runId", runId)
                        .putExtra("gpuPercent", percent));
            } catch (Exception ignored) { /* progress files remain authoritative */ }
        }

        /** Numeric prune argument is a separate validated bridge method: never
         * interpolate arbitrary page strings into service JavaScript. */
        @JavascriptInterface public void headlessPrune(String runId, int percent) {
            if (!RunFiles.validRunId(runId) || percent < 1 || percent > 99) return;
            try {
                startService(new Intent(MainActivity.this, TrainingService.class)
                        .putExtra("action", "headless-prune")
                        .putExtra("runId", runId)
                        .putExtra("prunePercent", percent));
            } catch (Exception ignored) { /* progress files remain authoritative */ }
        }

        private File captureArchive(String requested) {
            File root = new File(getFilesDir(), "captures");
            if (requested != null && requested.matches("[A-Za-z0-9_-]{1,80}")) {
                File exact = new File(root, requested);
                if (exact.isDirectory()) return exact;
            }
            File[] dirs = root.listFiles(File::isDirectory);
            if (dirs == null || dirs.length == 0) return null;
            java.util.Arrays.sort(dirs, (a, b) -> Long.compare(b.lastModified(), a.lastModified()));
            for (File dir : dirs) {
                if (new File(dir, "transforms.json").isFile() && NativeCaptureActivity.frameCount(dir) >= 3) return dir;
            }
            return null;
        }

        /** Native scan archives are the image ground truth. Return references,
         *  never image bytes; WebViews and service trainers fetch these URLs. */
        @JavascriptInterface public String exportDataset(String requestedId) {
            try {
                File current = captureArchive(requestedId);
                if (current == null) return "{\"ready\":false}";
                File transformFile = new File(current, "transforms.json");
                File pointsFile = new File(current, "seed_points.json");
                String captureId = current.getName();
                String base = "https://appassets.androidplatform.net/captures/" + captureId + "/";
                JSONObject result = new JSONObject();
                result.put("captureId", captureId);
                result.put("created", current.lastModified());
                result.put("frameCount", NativeCaptureActivity.frameCount(current));
                result.put("pointCount", NativeCaptureActivity.pointCount(current));
                result.put("transformsUri", base + "transforms.json");
                result.put("pointsUri", base + "seed_points.json");
                result.put("metadataUri", base + "capture.json");
                result.put("seedPlyUri", base + "seed.ply");
                JSONArray frames = new JSONArray();
                File[] files = current.listFiles((dir, name) -> name.matches("frame_[0-9]{4}\\.jpg"));
                if (files != null) {
                    java.util.Arrays.sort(files, java.util.Comparator.comparing(File::getName));
                    for (File file : files) {
                        JSONObject item = new JSONObject();
                        item.put("name", file.getName());
                        item.put("uri", base + file.getName());
                        frames.put(item);
                    }
                }
                result.put("frames", frames);
                result.put("ready", transformFile.isFile() && pointsFile.isFile());
                return result.toString();
            } catch (Exception e) {
                return "{\"ready\":false,\"error\":" + JSONObject.quote(e.getMessage() == null ? "export unavailable" : e.getMessage()) + "}";
            }
        }

        /** Explicitly delete a durable run. Live runs are protected. */
        @JavascriptInterface public boolean deleteRunArchive(String runId) {
            if (!RunFiles.validRunId(runId)) return false;
            if (runId.equals(RunFiles.readActive(MainActivity.this))) return false;
            return deleteArchiveTree(RunFiles.runDir(MainActivity.this, runId));
        }

        /** Explicit scan deletion is reference-safe: any durable run manifest
         * naming this capture keeps the canonical photos alive. */
        @JavascriptInterface public String deleteCaptureArchive(String captureId) {
            JSONObject result = new JSONObject();
            try {
                if (captureId == null || !captureId.matches("[A-Za-z0-9_-]{1,80}")) {
                    result.put("deleted", false); result.put("reason", "invalid"); return result.toString();
                }
                File[] runs = RunFiles.runsDir(MainActivity.this).listFiles(File::isDirectory);
                if (runs != null) for (File run : runs) {
                    String raw = RunFiles.readJson(MainActivity.this, run.getName(), "manifest.json");
                    if (raw != null && captureId.equals(new JSONObject(raw).optString("srcKey"))) {
                        result.put("deleted", false); result.put("reason", "referenced");
                        result.put("runId", run.getName()); return result.toString();
                    }
                }
                File archive = new File(new File(getFilesDir(), "captures"), captureId);
                result.put("deleted", archive.isDirectory() && deleteArchiveTree(archive));
                if (!result.optBoolean("deleted")) result.put("reason", "missing");
            } catch (Exception e) {
                try { result.put("deleted", false); result.put("reason", "error"); } catch (Exception ignored) { }
            }
            return result.toString();
        }

        private boolean deleteArchiveTree(File file) {
            if (file == null || !file.exists()) return true;
            File[] children = file.listFiles();
            if (children != null) for (File child : children) if (!deleteArchiveTree(child)) return false;
            return file.delete();
        }

        /** One-time migration for captures saved by older builds only in
         * IndexedDB. Each file publishes atomically into a canonical native
         * archive; JavaScript deletes the old blobs only after verification. */
        @JavascriptInterface public synchronized boolean writeCaptureArchiveChunk(
                String captureId, String name, String base64, boolean last) {
            if (captureId == null || !captureId.matches("legacy_[A-Za-z0-9_-]{1,72}") ||
                    name == null || base64 == null) return false;
            boolean allowed = name.matches("frame_[0-9]{4}\\.jpg") ||
                    name.equals("transforms.json") || name.equals("seed_points.json") ||
                    name.equals("capture.json");
            if (!allowed) return false;
            File dir = new File(new File(getFilesDir(), "captures"), captureId);
            if (!dir.exists() && !dir.mkdirs()) return false;
            File target = new File(dir, name);
            File tmp = new File(dir, name + ".tmp");
            String key = target.getAbsolutePath();
            try {
                OutputStream out = captureImports.get(key);
                if (out == null) {
                    try { tmp.delete(); } catch (Exception ignored) { }
                    out = new FileOutputStream(tmp, false);
                    captureImports.put(key, out);
                }
                out.write(Base64.decode(base64, Base64.DEFAULT));
                if (!last) return true;
                out.close(); captureImports.remove(key);
                if (target.exists() && !target.delete()) return false;
                return tmp.renameTo(target);
            } catch (Exception e) {
                OutputStream out = captureImports.remove(key);
                if (out != null) try { out.close(); } catch (Exception ignored) { }
                try { tmp.delete(); } catch (Exception ignored) { }
                return false;
            }
        }

        /** One Android-app-data catalog for Home. ARCore archives are the
         * canonical image/pose source; run directories contain only their own
         * model checkpoints and references to those archives. */
        @JavascriptInterface public String listHomeLibrary() {
            JSONObject out = new JSONObject();
            JSONArray captures;
            try { captures = new JSONArray(listCaptureArchives()); }
            catch (Exception ignored) { captures = new JSONArray(); }
            JSONArray runs = new JSONArray();
            for (String runId : RunFiles.listDurableRuns(MainActivity.this)) {
                try {
                    JSONObject row = new JSONObject();
                    row.put("runId", runId);
                    String manifestRaw = RunFiles.readJson(MainActivity.this, runId, "manifest.json");
                    JSONObject manifest = manifestRaw == null ? new JSONObject() : new JSONObject(manifestRaw);
                    row.put("name", manifest.optString("name", "Training run"));
                    row.put("srcKey", manifest.optString("srcKey", ""));
                    row.put("photoCount", manifest.optJSONArray("photos") == null ? 0 : manifest.optJSONArray("photos").length());
                    String statusRaw = RunFiles.readJson(MainActivity.this, runId, "status.json");
                    JSONObject status = statusRaw == null ? new JSONObject() : new JSONObject(statusRaw);
                    row.put("status", status.optString("status", "interrupted"));
                    row.put("owner", status.optString("owner", ""));
                    row.put("at", status.optLong("at", 0));
                    row.put("stopped", status.optBoolean("stopped", false));
                    String progressRaw = RunFiles.readJson(MainActivity.this, runId, "progress.json");
                    JSONObject progress = progressRaw == null ? new JSONObject() : new JSONObject(progressRaw);
                    // The service publishes high-frequency numbers separately
                    // from lifecycle status. Merge only when that beat is at
                    // least as new: a stale service progress file must not
                    // overwrite a later Activity handback/checkpoint.
                    boolean progressNewer = progress.optLong("at", 0) >= status.optLong("at", 0);
                    row.put("iter", progressNewer
                            ? Math.max(status.optInt("iter", 0), progress.optInt("iter", 0))
                            : status.optInt("iter", 0));
                    row.put("splats", progressNewer && progress.has("splats")
                            ? progress.optInt("splats", status.optInt("splats", 0))
                            : status.optInt("splats", 0));
                    if (progressNewer && progress.has("psnr")) row.put("psnr", progress.optDouble("psnr"));
                    else if (status.has("psnr")) row.put("psnr", status.optDouble("psnr"));
                    File thumb = new File(RunFiles.runDir(MainActivity.this, runId), "thumb.webp");
                    File snapshot = new File(RunFiles.runDir(MainActivity.this, runId), "snapshot.png");
                    if (thumb.isFile()) row.put("previewUri", "https://appassets.androidplatform.net/runs/" + runId + "/thumb.webp");
                    else if (snapshot.isFile()) row.put("previewUri", "https://appassets.androidplatform.net/runs/" + runId + "/snapshot.png");
                    else {
                        String captureId = manifest.optString("srcKey", "").replaceFirst("^native:", "");
                        File capture = new File(new File(getFilesDir(), "captures"), captureId);
                        File[] frames = capture.listFiles((d, name) -> name.matches("frame_[0-9]{4}\\.jpg"));
                        if (frames != null && frames.length > 0) {
                            java.util.Arrays.sort(frames, java.util.Comparator.comparing(File::getName));
                            row.put("previewUri", "https://appassets.androidplatform.net/captures/" +
                                    captureId + "/" + frames[0].getName());
                        }
                    }
                    runs.put(row);
                } catch (Exception ignored) { /* one malformed run cannot empty Home */ }
            }
            try { out.put("captures", captures); out.put("runs", runs); }
            catch (Exception ignored) { }
            return out.toString();
        }

        @JavascriptInterface public String listCaptureArchives() {
            JSONArray out = new JSONArray();
            try {
                File root = new File(getFilesDir(), "captures");
                File[] dirs = root.listFiles(File::isDirectory);
                if (dirs == null) return out.toString();
                java.util.Arrays.sort(dirs, (a, b) -> Long.compare(b.lastModified(), a.lastModified()));
                for (File dir : dirs) {
                    if (!dir.getName().matches("[A-Za-z0-9_-]{1,80}")) continue;
                    int count = NativeCaptureActivity.frameCount(dir);
                    if (count < 3 || !new File(dir, "transforms.json").isFile()) continue;
                    JSONObject item = new JSONObject();
                    item.put("captureId", dir.getName());
                    item.put("created", dir.lastModified());
                    item.put("frameCount", count);
                    File[] frames = dir.listFiles((d, name) -> name.matches("frame_[0-9]{4}\\.jpg"));
                    if (frames != null && frames.length > 0) {
                        java.util.Arrays.sort(frames, java.util.Comparator.comparing(File::getName));
                        item.put("previewUri", "https://appassets.androidplatform.net/captures/" +
                                dir.getName() + "/" + frames[0].getName());
                    }
                    out.put(item);
                }
            } catch (Exception ignored) { }
            return out.toString();
        }
    }
}
