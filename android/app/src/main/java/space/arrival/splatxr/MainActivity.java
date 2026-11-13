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
import java.io.OutputStream;

public final class MainActivity extends Activity {
    private static final int REQUEST_CAPTURE = 41;
    private static final int REQUEST_FILE_PICKER = 42;
    private static final int REQUEST_SAVE_FILE = 43;
    private static final int REQUEST_NOTIFY_PERMISSION = 44;
    /** The training notification's Pause/Resume button: the service fires it,
     *  the activity turns it into a WebView toggle call. */
    public static final String ACTION_TOGGLE_TRAINING = "space.arrival.splatxr.TOGGLE_TRAINING";
    /** The swipe-away handoff: the service fires it from onTaskRemoved, the
     *  activity acks at once and runs the page's file checkpoint — the
     *  verdict follows as headless-ready (or the service times out). */
    public static final String ACTION_HANDOFF_TRAINING = "space.arrival.splatxr.HANDOFF_TRAINING";
    private ValueCallback<Uri[]> fileChooserCallback;
    private WebView webView;
    private WebViewAssetLoader assetLoader;
    private final Object fileSaveLock = new Object();
    private OutputStream fileSaveStream;
    private boolean fileSavePickerPending;
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
            webView.post(() -> webView.evaluateJavascript(
                    "window.dispatchEvent(new CustomEvent('splatxr-native-capture'))", null));
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

    @Override protected void onDestroy() {
        try { unregisterReceiver(toggleTrainingReceiver); } catch (Exception ignored) { /* never registered */ }
        try { unregisterReceiver(handoffReceiver); } catch (Exception ignored) { /* never registered */ }
        if (RunFiles.readActive(this) == null) stopTrainingService();
        cancelFileSave();
        if (fileChooserCallback != null) {
            fileChooserCallback.onReceiveValue(null);
            fileChooserCallback = null;
        }
        super.onDestroy();
    }

    private final class AndroidARBridge {
        @JavascriptInterface public boolean isAvailable() { return true; }

        @JavascriptInterface public void startCapture() {
            runOnUiThread(() -> startActivityForResult(new Intent(MainActivity.this, NativeCaptureActivity.class), REQUEST_CAPTURE));
        }

        @JavascriptInterface public void captureKeyframe() {
            NativeCaptureActivity.requestKeyframe();
        }

        @JavascriptInterface public void resetCaptureSession() {
            NativeCaptureActivity.resetCaptureSession(MainActivity.this);
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

        @JavascriptInterface public void cleanRunDirs(String keepId) {
            RunFiles.cleanRunDirs(MainActivity.this, keepId);
        }

        @JavascriptInterface public String retainStatus(String runId) {
            String done = RunFiles.readJson(MainActivity.this, runId, "photos.json");
            return done != null ? done : "{\"pending\":true}";
        }

        /** Copy the native capture's frames into the run dir (background
         *  thread; the page polls retainStatus). Zero bytes cross the bridge. */
        @JavascriptInterface public void retainCapture(final String runId) {
            new Thread(() -> {
                try {
                    java.util.List<String> kept = RunFiles.retainCaptureFrames(MainActivity.this, runId);
                    JSONObject status = new JSONObject();
                    if (kept.isEmpty()) {
                        status.put("error", "no capture frames left to retain");
                    } else {
                        status.put("names", new JSONArray(kept));
                        status.put("complete", true);
                    }
                    status.put("at", System.currentTimeMillis());
                    RunFiles.writeJson(MainActivity.this, runId, "photos.json", status.toString());
                } catch (Exception e) {
                    try {
                        JSONObject status = new JSONObject();
                        status.put("error", e.getMessage() == null ? "retain failed" : e.getMessage());
                        RunFiles.writeJson(MainActivity.this, runId, "photos.json", status.toString());
                    } catch (Exception ignored) { /* status is best-effort */ }
                }
            }).start();
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

        /** A command to the headless twin (pause/resume/stop/snapshot). Plain
         *  start: the service is already foreground while a run lives. */
        @JavascriptInterface public void headlessCommand(String action, String runId) {
            if (action == null || runId == null) return;
            if (!action.equals("headless-pause") && !action.equals("headless-resume") &&
                    !action.equals("headless-stop") && !action.equals("headless-snapshot")) return;
            if (!RunFiles.validRunId(runId)) return;
            try {
                startService(new Intent(MainActivity.this, TrainingService.class)
                        .putExtra("action", action)
                        .putExtra("runId", runId));
            } catch (Exception ignored) { /* the twin is gone; the files say so */ }
        }

        @JavascriptInterface public String exportDataset() {
            try {
                File current = new File(getFilesDir(), "captures/current");
                File transformFile = new File(current, "transforms.json");
                File pointsFile = new File(current, "seed_points.json");
                JSONObject result = new JSONObject();
                result.put("frameCount", NativeCaptureActivity.frameCount(current));
                result.put("pointCount", NativeCaptureActivity.pointCount(current));
                result.put("transformsUri", "https://appassets.androidplatform.net/captures/current/transforms.json");
                result.put("pointsUri", "https://appassets.androidplatform.net/captures/current/seed_points.json");
                result.put("seedPlyUri", "https://appassets.androidplatform.net/captures/current/seed.ply");
                JSONArray frames = new JSONArray();
                File[] files = current.listFiles((dir, name) -> name.matches("frame_[0-9]{4}\\.jpg"));
                if (files != null) {
                    java.util.Arrays.sort(files, java.util.Comparator.comparing(File::getName));
                    for (File file : files) {
                        JSONObject item = new JSONObject();
                        item.put("name", file.getName());
                        item.put("uri", "https://appassets.androidplatform.net/captures/current/" + file.getName());
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
    }
}
