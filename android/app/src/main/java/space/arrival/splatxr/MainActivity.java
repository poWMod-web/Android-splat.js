package space.arrival.splatxr;

import android.app.Activity;
import android.content.Intent;
import android.net.Uri;
import android.os.Bundle;
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

public final class MainActivity extends Activity {
    private static final int REQUEST_CAPTURE = 41;
    private static final int REQUEST_FILE_PICKER = 42;
    private ValueCallback<Uri[]> fileChooserCallback;
    private WebView webView;
    private WebViewAssetLoader assetLoader;

    @Override protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        getWindow().setStatusBarColor(android.graphics.Color.rgb(14, 17, 18));
        getWindow().setNavigationBarColor(android.graphics.Color.rgb(14, 17, 18));

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
        WebView.setWebContentsDebuggingEnabled(BuildConfig.DEBUG);
        webView.loadUrl("https://appassets.androidplatform.net/assets/app/index.html");
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
        } else if (requestCode == REQUEST_CAPTURE && resultCode == RESULT_OK && webView != null) {
            webView.post(() -> webView.evaluateJavascript(
                    "window.dispatchEvent(new CustomEvent('splatxr-native-capture'))", null));
        }
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
