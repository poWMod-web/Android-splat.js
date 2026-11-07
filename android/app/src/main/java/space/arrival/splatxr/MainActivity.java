package space.arrival.splatxr;

import android.app.Activity;
import android.content.Intent;
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
    private ValueCallback<Uri[]> fileChooserCallback;
    private WebView webView;
    private WebViewAssetLoader assetLoader;
    private final Object fileSaveLock = new Object();
    private OutputStream fileSaveStream;
    private boolean fileSavePickerPending;

    @Override protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        getWindow().setStatusBarColor(android.graphics.Color.rgb(14, 17, 18));
        getWindow().setNavigationBarColor(android.graphics.Color.rgb(14, 17, 18));
        getWindow().getDecorView().setSystemUiVisibility(0);

        File captureRoot = new File(getFilesDir(), "captures");
        if (!captureRoot.exists()) captureRoot.mkdirs();
        assetLoader = new WebViewAssetLoader.Builder()
                .addPathHandler("/assets/", new WebViewAssetLoader.AssetsPathHandler(this))
                .addPathHandler("/captures/", new WebViewAssetLoader.InternalStoragePathHandler(this, captureRoot))
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

    @Override protected void onDestroy() {
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
