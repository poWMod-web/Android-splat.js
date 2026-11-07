package space.arrival.splatxr;

import android.Manifest;
import android.app.Activity;
import android.content.Context;
import android.content.res.ColorStateList;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.ImageFormat;
import android.graphics.Paint;
import android.graphics.Typeface;
import android.graphics.drawable.GradientDrawable;
import android.graphics.Rect;
import android.graphics.YuvImage;
import android.media.Image;
import android.text.InputType;
import android.opengl.GLES11Ext;
import android.opengl.GLES20;
import android.opengl.GLSurfaceView;
import android.opengl.Matrix;
import android.os.Build;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.view.Gravity;
import android.view.Surface;
import android.view.View;
import android.view.WindowInsets;
import android.view.inputmethod.EditorInfo;
import android.widget.CheckBox;
import android.widget.EditText;
import android.widget.FrameLayout;
import android.widget.LinearLayout;
import android.widget.TextView;
import android.widget.Toast;

import com.google.ar.core.ArCoreApk;
import com.google.ar.core.Camera;
import com.google.ar.core.CameraConfig;
import com.google.ar.core.CameraConfigFilter;
import com.google.ar.core.Config;
import com.google.ar.core.Frame;
import com.google.ar.core.CameraIntrinsics;
import com.google.ar.core.PointCloud;
import com.google.ar.core.Pose;
import com.google.ar.core.Session;
import com.google.ar.core.TrackingState;
import com.google.ar.core.exceptions.CameraNotAvailableException;
import com.google.ar.core.exceptions.NotYetAvailableException;

import org.json.JSONArray;
import org.json.JSONObject;

import java.io.ByteArrayOutputStream;
import java.io.DataOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.concurrent.atomic.AtomicBoolean;
import javax.microedition.khronos.egl.EGLConfig;
import javax.microedition.khronos.opengles.GL10;

/** Native ARCore capture surface. Keyframes, intrinsics and poses are sampled from the same ARCore frame. */
public final class NativeCaptureActivity extends Activity implements GLSurfaceView.Renderer {
    private static volatile NativeCaptureActivity active;
    private static final int CAMERA_PERMISSION = 71;
    private Session session;
    private GLSurfaceView glView;
    private FrameLayout captureRoot;
    private FrameLayout topBar;
    private FrameLayout controlBar;
    private TextView guideText;
    private TextView countLabel;
    private TextView trackingLabel;
    private TextView seedsLabel;
    private TextView finishLabel;
    private EditText spacingInput;
    private volatile boolean autoCaptureEnabled;
    private volatile int captureSpacingCm = 10;
    private boolean installRequested;
    private boolean sessionResumed;
    private final Handler main = new Handler(Looper.getMainLooper());
    private final AtomicBoolean captureRequested = new AtomicBoolean(false);
    private final CoverageRenderer cloudRenderer = new CoverageRenderer();
    private File outputDir;
    private volatile int capturedFrameCount;
    private volatile int capturedSeedCount;
    private boolean trackingUiInitialized;
    private boolean lastTrackingUiState;

    static void requestKeyframe() {
        NativeCaptureActivity value = active;
        if (value != null) value.captureRequested.set(true);
    }

    static void resetCaptureSession(Context context) {
        NativeCaptureActivity value = active;
        if (value != null) {
            value.glView.queueEvent(value.cloudRenderer::resetData);
            return;
        }
        deleteTree(new File(context.getFilesDir(), "captures/current"));
    }

    static int frameCount(File dir) {
        File[] files = dir.listFiles((d, name) -> name.matches("frame_[0-9]{4}\\.jpg"));
        return files == null ? 0 : files.length;
    }

    static int pointCount(File dir) {
        try {
            JSONObject data = new JSONObject(readText(new File(dir, "seed_points.json")));
            return data.getJSONArray("points").length();
        } catch (Exception ignored) { return 0; }
    }

    @Override protected void onCreate(Bundle state) {
        super.onCreate(state);
        active = this;
        captureSpacingCm = Math.max(2, Math.min(200, getPreferences(MODE_PRIVATE).getInt("capture_spacing_cm", 10)));
        cloudRenderer.setSpacingCm(captureSpacingCm);
        outputDir = new File(getFilesDir(), "captures/current");
        deleteTree(outputDir);
        outputDir.mkdirs();

        getWindow().setStatusBarColor(Color.rgb(7, 9, 9));
        getWindow().setNavigationBarColor(Color.rgb(7, 9, 9));
        // Keep app content out of the system bars. The inset listener also pads
        // the root on Android versions that enforce edge-to-edge for targetSdk 35.
        getWindow().getDecorView().setSystemUiVisibility(0);
        captureRoot = buildCaptureUi();
        setContentView(captureRoot);
        applySafeAreaInsets(captureRoot);
    }


    private FrameLayout buildCaptureUi() {
        FrameLayout root = new FrameLayout(this);
        root.setBackgroundColor(Color.BLACK);
        glView = new GLSurfaceView(this);
        glView.setEGLContextClientVersion(3);
        glView.setPreserveEGLContextOnPause(true);
        glView.setRenderer(this);
        glView.setRenderMode(GLSurfaceView.RENDERMODE_CONTINUOUSLY);
        root.addView(glView, new FrameLayout.LayoutParams(-1, -1));

        ReticleOverlay reticle = new ReticleOverlay(this);
        root.addView(reticle, new FrameLayout.LayoutParams(-1, -1));

        topBar = new FrameLayout(this);
        LinearLayout telemetry = new LinearLayout(this);
        telemetry.setOrientation(LinearLayout.VERTICAL);
        telemetry.setPadding(dp(14), dp(10), dp(14), dp(10));
        telemetry.setBackground(rounded(Color.argb(205, 7, 9, 9), Color.TRANSPARENT, 0, dp(10)));
        countLabel = telemetryText("0 photos", 14, Color.rgb(47, 212, 193));
        trackingLabel = telemetryText("Tracking: waiting for tracking", 10, Color.argb(200, 230, 236, 235));
        seedsLabel = telemetryText("Seeds: none measured (random fallback)", 10, Color.argb(200, 230, 236, 235));
        telemetry.addView(countLabel);
        telemetry.addView(trackingLabel, topMargin(dp(4)));
        telemetry.addView(seedsLabel, topMargin(dp(4)));
        FrameLayout.LayoutParams telemetryParams = new FrameLayout.LayoutParams(-2, -2, Gravity.TOP | Gravity.LEFT);
        topBar.addView(telemetry, telemetryParams);

        TextView cancel = controlText("Cancel", 12, Color.WHITE);
        cancel.setPadding(dp(14), dp(9), dp(14), dp(9));
        cancel.setBackground(rounded(Color.argb(205, 7, 9, 9), Color.argb(55, 255, 255, 255), dp(1), dp(22)));
        cancel.setOnClickListener(v -> { setResult(RESULT_CANCELED); finish(); });
        FrameLayout.LayoutParams cancelParams = new FrameLayout.LayoutParams(-2, -2, Gravity.TOP | Gravity.RIGHT);
        topBar.addView(cancel, cancelParams);
        root.addView(topBar, new FrameLayout.LayoutParams(-1, -2, Gravity.TOP));

        guideText = controlText(
                "Manual capture is ready. Press Shutter for each photo, or enable auto-capture to take one every " + captureSpacingCm + " cm traveled. Move slowly and aim for 20 or more overlapping views. Rotating in place does not trigger automatic photos.",
                12, Color.rgb(230, 236, 235));
        guideText.setGravity(Gravity.CENTER);
        guideText.setLineSpacing(dp(2), 1f);
        guideText.setPadding(dp(16), dp(12), dp(16), dp(12));
        guideText.setBackground(rounded(Color.argb(210, 7, 9, 9), Color.argb(30, 255, 255, 255), dp(1), dp(12)));
        FrameLayout.LayoutParams guideParams = new FrameLayout.LayoutParams(-1, -2, Gravity.BOTTOM | Gravity.CENTER_HORIZONTAL);
        guideParams.setMargins(dp(16), 0, dp(16), dp(125));
        root.addView(guideText, guideParams);

        controlBar = new FrameLayout(this);
        LinearLayout autoPanel = new LinearLayout(this);
        autoPanel.setOrientation(LinearLayout.VERTICAL);
        autoPanel.setPadding(dp(8), dp(6), dp(8), dp(6));
        autoPanel.setBackground(rounded(Color.argb(210, 7, 9, 9), Color.TRANSPARENT, 0, dp(12)));
        CheckBox autoToggle = new CheckBox(this);
        autoToggle.setText("Auto-capture");
        autoToggle.setTextSize(11);
        autoToggle.setTextColor(Color.argb(220, 230, 236, 235));
        autoToggle.setButtonTintList(new ColorStateList(
                new int[][]{new int[]{android.R.attr.state_checked}, new int[]{}},
                new int[]{Color.rgb(47, 212, 193), Color.WHITE}));
        autoToggle.setChecked(false);
        autoPanel.addView(autoToggle, new LinearLayout.LayoutParams(-2, dp(28)));

        LinearLayout spacingRow = new LinearLayout(this);
        spacingRow.setGravity(Gravity.CENTER_VERTICAL);
        spacingRow.setPadding(dp(22), 0, 0, 0);
        spacingInput = new EditText(this);
        spacingInput.setSingleLine(true);
        spacingInput.setInputType(InputType.TYPE_CLASS_NUMBER);
        spacingInput.setImeOptions(EditorInfo.IME_ACTION_DONE);
        spacingInput.setText(String.valueOf(captureSpacingCm));
        spacingInput.setTextSize(10);
        spacingInput.setTypeface(Typeface.MONOSPACE);
        spacingInput.setTextColor(Color.WHITE);
        spacingInput.setGravity(Gravity.CENTER);
        spacingInput.setPadding(dp(4), 0, dp(4), 0);
        spacingInput.setSelectAllOnFocus(true);
        spacingInput.setContentDescription("Auto-capture travel spacing in centimeters, from 2 to 200");
        spacingInput.setBackground(rounded(Color.argb(24, 255, 255, 255), Color.argb(45, 255, 255, 255), dp(1), dp(6)));
        spacingRow.addView(spacingInput, new LinearLayout.LayoutParams(dp(45), dp(27)));
        TextView cm = controlText("cm", 10, Color.argb(220, 230, 236, 235));
        LinearLayout.LayoutParams cmParams = new LinearLayout.LayoutParams(-2, -2);
        cmParams.setMargins(dp(4), 0, 0, 0);
        spacingRow.addView(cm, cmParams);
        autoPanel.addView(spacingRow);
        FrameLayout.LayoutParams autoParams = new FrameLayout.LayoutParams(-2, -2, Gravity.LEFT | Gravity.CENTER_VERTICAL);
        controlBar.addView(autoPanel, autoParams);

        CaptureShutterView shutter = new CaptureShutterView(this);
        shutter.setOnClickListener(v -> {
            if (session == null || !sessionResumed) { showStatus("AR tracking is not ready yet."); return; }
            captureRequested.set(true);
            showStatus("Shutter armed; waiting for the next tracked camera frame…");
        });
        int shutterSize = dp(76);
        FrameLayout.LayoutParams shutterParams = new FrameLayout.LayoutParams(shutterSize, shutterSize, Gravity.CENTER);
        controlBar.addView(shutter, shutterParams);

        finishLabel = controlText("Finish (0/3)", 12, Color.rgb(47, 212, 193));
        finishLabel.setGravity(Gravity.CENTER);
        finishLabel.setPadding(dp(13), dp(11), dp(13), dp(11));
        finishLabel.setBackground(rounded(Color.argb(205, 7, 9, 9), Color.argb(80, 47, 212, 193), dp(1), dp(24)));
        finishLabel.setEnabled(false);
        finishLabel.setOnClickListener(v -> finishCapture());
        FrameLayout.LayoutParams finishParams = new FrameLayout.LayoutParams(-2, -2, Gravity.RIGHT | Gravity.CENTER_VERTICAL);
        controlBar.addView(finishLabel, finishParams);
        FrameLayout.LayoutParams controlParams = new FrameLayout.LayoutParams(-1, dp(82), Gravity.BOTTOM);
        controlParams.setMargins(dp(18), 0, dp(18), dp(25));
        root.addView(controlBar, controlParams);

        autoToggle.setOnCheckedChangeListener((view, enabled) -> {
            autoCaptureEnabled = enabled;
            glView.queueEvent(cloudRenderer::resetDistanceBaseline);
            showStatus(enabled
                    ? "Auto-capture enabled: one photo every " + captureSpacingCm + " cm traveled. Move slowly."
                    : "Manual capture enabled. Press Shutter to take a photo.");
        });
        spacingInput.setOnFocusChangeListener((view, hasFocus) -> { if (!hasFocus) commitCaptureSpacing(); });
        spacingInput.setOnEditorActionListener((view, actionId, event) -> {
            if (actionId == EditorInfo.IME_ACTION_DONE) {
                commitCaptureSpacing();
                spacingInput.clearFocus();
                return true;
            }
            return false;
        });
        return root;
    }

    private void applySafeAreaInsets(FrameLayout root) {
        root.setOnApplyWindowInsetsListener((view, insets) -> {
            int left, top, right, bottom;
            if (Build.VERSION.SDK_INT >= 30) {
                android.graphics.Insets bars = insets.getInsets(
                        WindowInsets.Type.systemBars() | WindowInsets.Type.displayCutout());
                left = bars.left;
                top = bars.top;
                right = bars.right;
                bottom = bars.bottom;
            } else {
                left = insets.getSystemWindowInsetLeft();
                top = insets.getSystemWindowInsetTop();
                right = insets.getSystemWindowInsetRight();
                bottom = insets.getSystemWindowInsetBottom();
            }
            // Pad the whole content area instead of drawing controls underneath
            // the status/navigation bars. This also keeps the camera surface clear.
            view.setPadding(left, top, right, bottom);
            FrameLayout.LayoutParams topParams = (FrameLayout.LayoutParams) topBar.getLayoutParams();
            topParams.topMargin = dp(16);
            topBar.setLayoutParams(topParams);
            FrameLayout.LayoutParams guideParams = (FrameLayout.LayoutParams) guideText.getLayoutParams();
            guideParams.bottomMargin = dp(125);
            guideText.setLayoutParams(guideParams);
            FrameLayout.LayoutParams controlParams = (FrameLayout.LayoutParams) controlBar.getLayoutParams();
            controlParams.bottomMargin = dp(25);
            controlBar.setLayoutParams(controlParams);
            return insets;
        });
        root.requestApplyInsets();
    }

    private LinearLayout.LayoutParams topMargin(int value) {
        LinearLayout.LayoutParams params = new LinearLayout.LayoutParams(-2, -2);
        params.topMargin = value;
        return params;
    }

    private TextView telemetryText(String text, float size, int color) {
        TextView view = controlText(text, size, color);
        view.setTypeface(Typeface.MONOSPACE);
        return view;
    }

    private TextView controlText(String text, float size, int color) {
        TextView view = new TextView(this);
        view.setText(text);
        view.setTextSize(size);
        view.setTextColor(color);
        return view;
    }

    private GradientDrawable rounded(int fill, int stroke, int strokeWidth, float radius) {
        GradientDrawable drawable = new GradientDrawable();
        drawable.setColor(fill);
        drawable.setCornerRadius(radius);
        if (strokeWidth > 0) drawable.setStroke(strokeWidth, stroke);
        return drawable;
    }

    private final class ReticleOverlay extends View {
        private final Paint paint = new Paint(Paint.ANTI_ALIAS_FLAG);
        ReticleOverlay(Context context) { super(context); setClickable(false); setFocusable(false); }
        @Override protected void onDraw(Canvas canvas) {
            super.onDraw(canvas);
            float cx = getWidth() * 0.5f;
            float cy = getHeight() * 0.43f;
            float w = Math.min(getWidth() * 0.58f, dp(340));
            float h = Math.min(getHeight() * 0.46f, dp(340));
            paint.setStyle(Paint.Style.STROKE);
            paint.setStrokeWidth(dp(1));
            paint.setColor(Color.argb(145, 47, 212, 193));
            canvas.drawRoundRect(cx-w/2, cy-h/2, cx+w/2, cy+h/2, dp(18), dp(18), paint);
            paint.setStyle(Paint.Style.FILL);
            paint.setColor(Color.argb(190, 255, 255, 255));
            paint.setStrokeWidth(dp(2));
            canvas.drawLine(cx-dp(4), cy, cx+dp(4), cy, paint);
            canvas.drawLine(cx, cy-dp(4), cx, cy+dp(4), paint);
        }
    }

    private final class CaptureShutterView extends View {
        private final Paint paint = new Paint(Paint.ANTI_ALIAS_FLAG);
        CaptureShutterView(Context context) {
            super(context);
            setClickable(true);
            setContentDescription("Take photo");
        }
        @Override protected void onDraw(Canvas canvas) {
            super.onDraw(canvas);
            float cx = getWidth()/2f, cy = getHeight()/2f;
            paint.setStyle(Paint.Style.FILL);
            paint.setColor(Color.argb(100, 0, 0, 0));
            canvas.drawCircle(cx, cy, dp(37), paint);
            paint.setStyle(Paint.Style.STROKE);
            paint.setStrokeWidth(dp(4));
            paint.setColor(Color.rgb(230, 236, 235));
            canvas.drawCircle(cx, cy, dp(33), paint);
            paint.setStyle(Paint.Style.FILL);
            paint.setColor(isPressed() ? Color.rgb(25, 183, 165) : Color.rgb(47, 212, 193));
            canvas.drawCircle(cx, cy, dp(28), paint);
            paint.setStyle(Paint.Style.STROKE);
            paint.setStrokeWidth(dp(4));
            paint.setColor(Color.argb(200, 7, 9, 9));
            canvas.drawCircle(cx, cy, dp(21), paint);
            paint.setStyle(Paint.Style.FILL);
            paint.setColor(Color.rgb(3, 36, 32));
            paint.setTextSize(dp(11));
            paint.setTypeface(Typeface.create("sans-serif-medium", Typeface.NORMAL));
            paint.setTextAlign(Paint.Align.CENTER);
            canvas.drawText("Shutter", cx, cy+dp(4), paint);
        }
        @Override public boolean onTouchEvent(android.view.MotionEvent event) {
            if (event.getAction() == android.view.MotionEvent.ACTION_DOWN) { setPressed(true); invalidate(); return true; }
            if (event.getAction() == android.view.MotionEvent.ACTION_UP) {
                setPressed(false); invalidate(); performClick(); return true;
            }
            if (event.getAction() == android.view.MotionEvent.ACTION_CANCEL) { setPressed(false); invalidate(); return true; }
            return true;
        }
        @Override public boolean performClick() { super.performClick(); return true; }
    }

    private int dp(float px) { return Math.round(px * getResources().getDisplayMetrics().density); }

    private void commitCaptureSpacing() {
        if (spacingInput == null) return;
        String raw = spacingInput.getText().toString().trim();
        int value;
        try { value = Integer.parseInt(raw); }
        catch (NumberFormatException e) { value = -1; }
        if (value < 2 || value > 200) {
            spacingInput.setError("Enter a whole number from 2 to 200 cm");
            spacingInput.setText(String.valueOf(captureSpacingCm));
            return;
        }
        spacingInput.setError(null);
        if (value == captureSpacingCm) return;
        final int spacing = value;
        captureSpacingCm = spacing;
        getPreferences(MODE_PRIVATE).edit().putInt("capture_spacing_cm", spacing).apply();
        glView.queueEvent(() -> cloudRenderer.setSpacingCm(spacing));
        showStatus(autoCaptureEnabled
                ? "Auto-capture updated: one photo every " + value + " cm traveled."
                : "Auto-capture distance set to " + value + " cm. Auto-capture is still off.");
    }

    private void showStatus(String message) {
        main.post(() -> { if (guideText != null) guideText.setText(message); });
    }

    private void refreshCaptureTelemetry() {
        main.post(() -> {
            if (countLabel != null) {
                int count = capturedFrameCount;
                countLabel.setText(count + " photo" + (count == 1 ? "" : "s"));
            }
            if (seedsLabel != null) {
                int count = capturedSeedCount;
                seedsLabel.setText(count == 0 ? "Seeds: none measured (random fallback)" : "Seeds: " + count + " measured");
            }
            if (finishLabel != null) {
                int count = capturedFrameCount;
                finishLabel.setText(count >= 3 ? "Finish (" + count + ")" : "Finish (" + count + "/3)");
                finishLabel.setEnabled(count >= 3);
                finishLabel.setAlpha(count >= 3 ? 1f : 0.42f);
            }
        });
    }

    private void setTrackingTelemetry(boolean tracking) {
        if (trackingUiInitialized && tracking == lastTrackingUiState) return;
        trackingUiInitialized = true;
        lastTrackingUiState = tracking;
        main.post(() -> {
            if (trackingLabel != null) trackingLabel.setText(tracking ? "Tracking: active" : "Tracking: waiting for tracking");
        });
    }

    @Override protected void onResume() {
        super.onResume();
        resumeArIfReady();
    }

    private void resumeArIfReady() {
        if (checkSelfPermission(Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
            requestPermissions(new String[]{Manifest.permission.CAMERA}, CAMERA_PERMISSION);
            return;
        }
        try {
            ArCoreApk.InstallStatus install = ArCoreApk.getInstance().requestInstall(this, !installRequested);
            if (install == ArCoreApk.InstallStatus.INSTALL_REQUESTED) {
                installRequested = true;
                return;
            }
            if (session == null) createSession();
            if (!sessionResumed) {
                session.resume();
                sessionResumed = true;
                glView.onResume();
            }
            showStatus("Manual capture is ready. Press Shutter for each photo, or enable auto-capture to take one every " + captureSpacingCm + " cm traveled. Move slowly and aim for 20 or more overlapping views. Rotating in place does not trigger automatic photos.");
        } catch (Exception e) {
            showStatus("ARCore unavailable: " + safeMessage(e));
            Toast.makeText(this, "Install or update Google Play Services for AR, then try again.", Toast.LENGTH_LONG).show();
        }
    }

    private void createSession() throws Exception {
        session = new Session(this);
        try {
            List<CameraConfig> configs = session.getSupportedCameraConfigs(new CameraConfigFilter(session));
            CameraConfig best = configs.stream().max(Comparator.comparingLong(c ->
                    (long) c.getImageSize().getWidth() * c.getImageSize().getHeight())).orElse(null);
            if (best != null) session.setCameraConfig(best);
        } catch (Exception ignored) { /* use ARCore's selected camera config */ }
        Config config = new Config(session);
        config.setUpdateMode(Config.UpdateMode.LATEST_CAMERA_IMAGE);
        config.setFocusMode(Config.FocusMode.AUTO);
        if (session.isDepthModeSupported(Config.DepthMode.AUTOMATIC)) config.setDepthMode(Config.DepthMode.AUTOMATIC);
        session.configure(config);
        cloudRenderer.setSession(session);
    }

    @Override protected void onPause() {
        if (sessionResumed) {
            glView.onPause();
            session.pause();
            sessionResumed = false;
        }
        super.onPause();
    }

    @Override protected void onDestroy() {
        if (active == this) active = null;
        if (session != null) { session.close(); session = null; }
        super.onDestroy();
    }

    @Override public void onRequestPermissionsResult(int requestCode, String[] permissions, int[] results) {
        super.onRequestPermissionsResult(requestCode, permissions, results);
        if (requestCode == CAMERA_PERMISSION) {
            if (results.length > 0 && results[0] == PackageManager.PERMISSION_GRANTED) resumeArIfReady();
            else { showStatus("Camera permission is required for native AR capture."); }
        }
    }

    private void finishCapture() {
        if (capturedFrameCount < 3) {
            showStatus("Capture at least 3 overlapping frames before training (20–200 recommended).");
            return;
        }
        showStatus("Writing camera poses and RGB seed cloud…");
        glView.queueEvent(() -> {
            try {
                cloudRenderer.writeDataset(outputDir);
                main.post(() -> {
                    setResult(RESULT_OK);
                    finish();
                });
            } catch (Exception e) {
                showStatus("Could not export capture: " + safeMessage(e));
            }
        });
    }

    @Override public void onSurfaceCreated(GL10 gl, EGLConfig config) {
        cloudRenderer.onSurfaceCreated();
    }
    @Override public void onSurfaceChanged(GL10 gl, int width, int height) {
        cloudRenderer.onSurfaceChanged(width, height);
    }
    @Override public void onDrawFrame(GL10 gl) {
        cloudRenderer.onDrawFrame();
    }

    private final class CoverageRenderer {
        private static final int MAX_POINTS = 50000;
        private static final float VOXEL = 0.025f;
        private final Map<String, CoveragePoint> coverage = new HashMap<>();
        private final Map<String, SeedPoint> seeds = new HashMap<>();
        private final List<JSONObject> frames = new ArrayList<>();
        private final List<CaptureViewpoint> viewpoints = new ArrayList<>();
        private final TravelCaptureTrigger travelTrigger = new TravelCaptureTrigger(0.10f);
        private Session arSession;
        private int cameraTexture = -1;
        private int bgProgram, pointProgram;
        private int bgPosition, bgUv, bgTexture;
        private int pointPosition, pointColor, pointMvp;
        private int width = 1, height = 1;
        private boolean cameraTextureAssigned;
        private final FloatBuffer quadPositions = floats(new float[]{-1,-1, 1,-1, -1,1, 1,1});
        private final FloatBuffer uvInput = floats(new float[]{0,0, 1,0, 0,1, 1,1});
        private final FloatBuffer uvOutput = floats(new float[8]);
        private final FloatBuffer pointPositionsBuffer = ByteBuffer.allocateDirect(MAX_POINTS * 3 * 4).order(ByteOrder.nativeOrder()).asFloatBuffer();
        private final FloatBuffer pointColorsBuffer = ByteBuffer.allocateDirect(MAX_POINTS * 3 * 4).order(ByteOrder.nativeOrder()).asFloatBuffer();
        private final FloatBuffer viewpointPositionsBuffer = ByteBuffer.allocateDirect(200 * 16 * 3 * 4).order(ByteOrder.nativeOrder()).asFloatBuffer();
        private final FloatBuffer viewpointColorsBuffer = ByteBuffer.allocateDirect(200 * 16 * 3 * 4).order(ByteOrder.nativeOrder()).asFloatBuffer();
        private final float[] projection = new float[16], view = new float[16], mvp = new float[16];

        void setSession(Session value) { arSession = value; }
        void setSpacingCm(int centimeters) { travelTrigger.setSpacingMeters(centimeters / 100f); }
        void resetDistanceBaseline() { travelTrigger.reset(); }
        boolean sampleTravel(float[] position) { return travelTrigger.sample(position); }
        void breakTravelContinuity() { travelTrigger.breakContinuity(); }
        void capturePosition(float[] position) { travelTrigger.captured(position); }

        void onSurfaceCreated() {
            cameraTextureAssigned = false;
            GLES20.glClearColor(0.02f, 0.03f, 0.03f, 1f);
            int[] textures = new int[1];
            GLES20.glGenTextures(1, textures, 0);
            cameraTexture = textures[0];
            GLES20.glBindTexture(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, cameraTexture);
            GLES20.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GLES20.GL_TEXTURE_MIN_FILTER, GLES20.GL_LINEAR);
            GLES20.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GLES20.GL_TEXTURE_MAG_FILTER, GLES20.GL_LINEAR);
            GLES20.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GLES20.GL_TEXTURE_WRAP_S, GLES20.GL_CLAMP_TO_EDGE);
            GLES20.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GLES20.GL_TEXTURE_WRAP_T, GLES20.GL_CLAMP_TO_EDGE);
            bgProgram = program(
                    "attribute vec4 aPosition; attribute vec2 aUv; varying vec2 vUv; void main(){ gl_Position=aPosition; vUv=aUv; }",
                    "#extension GL_OES_EGL_image_external : require\nprecision mediump float; uniform samplerExternalOES sTexture; varying vec2 vUv; void main(){ gl_FragColor=texture2D(sTexture,vUv); }");
            bgPosition = GLES20.glGetAttribLocation(bgProgram, "aPosition");
            bgUv = GLES20.glGetAttribLocation(bgProgram, "aUv");
            bgTexture = GLES20.glGetUniformLocation(bgProgram, "sTexture");
            pointProgram = program(
                    "attribute vec3 aPosition; attribute vec3 aColor; uniform mat4 uMvp; varying vec3 vColor; void main(){ gl_Position=uMvp*vec4(aPosition,1.0); gl_PointSize=5.0; vColor=aColor; }",
                    "precision mediump float; varying vec3 vColor; void main(){ gl_FragColor=vec4(vColor,0.92); }");
            pointPosition = GLES20.glGetAttribLocation(pointProgram, "aPosition");
            pointColor = GLES20.glGetAttribLocation(pointProgram, "aColor");
            pointMvp = GLES20.glGetUniformLocation(pointProgram, "uMvp");
        }

        void onSurfaceChanged(int w, int h) {
            width = Math.max(1, w); height = Math.max(1, h);
            GLES20.glViewport(0, 0, width, height);
        }

        void onDrawFrame() {
            GLES20.glClear(GLES20.GL_COLOR_BUFFER_BIT | GLES20.GL_DEPTH_BUFFER_BIT);
            if (arSession == null) return;
            try {
                if (!cameraTextureAssigned && cameraTexture >= 0) {
                    arSession.setCameraTextureName(cameraTexture);
                    cameraTextureAssigned = true;
                }
                int rotation = getWindowManager().getDefaultDisplay().getRotation();
                int displayRotation = rotation == Surface.ROTATION_90 ? 1 : rotation == Surface.ROTATION_180 ? 2 : rotation == Surface.ROTATION_270 ? 3 : 0;
                arSession.setDisplayGeometry(displayRotation, width, height);
                Frame frame = arSession.update();
                drawBackground(frame);
                boolean tracking = frame.getCamera().getTrackingState() == TrackingState.TRACKING;
                setTrackingTelemetry(tracking);
                if (tracking) {
                    float[] position = frame.getCamera().getPose().getTranslation();
                    boolean autoDue = autoCaptureEnabled && cloudRenderer.sampleTravel(position);
                    if (!autoCaptureEnabled) cloudRenderer.breakTravelContinuity();
                    boolean manualDue = captureRequested.getAndSet(false);
                    if (manualDue || autoDue) capture(frame);
                    // Render accumulated capture samples each preview frame, but only
                    // successful keyframe capture may add coverage observations.
                    drawCoverage(frame);
                } else {
                    cloudRenderer.breakTravelContinuity();
                }
            } catch (CameraNotAvailableException e) {
                setTrackingTelemetry(false);
                showStatus("AR camera is unavailable. Close other camera apps and retry.");
            } catch (Exception e) {
                setTrackingTelemetry(false);
                showStatus("AR frame error: " + safeMessage(e));
            }
        }

        private void drawBackground(Frame frame) {
            // ARCore accounts for display orientation and camera-texture UV layout.
            // Rotate preview by 180° by exchanging opposite UV pairs: (0 <-> 3) and (1 <-> 2).
            uvInput.position(0);
            uvOutput.position(0);
            frame.transformDisplayUvCoords(uvInput, uvOutput);

            // Perform 180° rotation by swapping diagonally opposite quad corners directly
            swapUvCoords(uvOutput, 0, 3);
            swapUvCoords(uvOutput, 1, 2);

            uvOutput.position(0);

            // OpenGL ES setup and draw
            GLES20.glDisable(GLES20.GL_DEPTH_TEST);
            GLES20.glUseProgram(bgProgram);

            GLES20.glActiveTexture(GLES20.GL_TEXTURE0);
            GLES20.glBindTexture(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, cameraTexture);
            GLES20.glUniform1i(bgTexture, 0);

            quadPositions.position(0);
            uvOutput.position(0);

            GLES20.glEnableVertexAttribArray(bgPosition);
            GLES20.glEnableVertexAttribArray(bgUv);

            GLES20.glVertexAttribPointer(bgPosition, 2, GLES20.GL_FLOAT, false, 0, quadPositions);
            GLES20.glVertexAttribPointer(bgUv, 2, GLES20.GL_FLOAT, false, 0, uvOutput);

            GLES20.glDrawArrays(GLES20.GL_TRIANGLE_STRIP, 0, 4);

            GLES20.glDisableVertexAttribArray(bgPosition);
            GLES20.glDisableVertexAttribArray(bgUv);
        }

        private void swapUvCoords(FloatBuffer buffer, int i, int j) {
            int idx1 = i * 2;
            int idx2 = j * 2;

            float u = buffer.get(idx1);
            float v = buffer.get(idx1 + 1);

            buffer.put(idx1, buffer.get(idx2));
            buffer.put(idx1 + 1, buffer.get(idx2 + 1));

            buffer.put(idx2, u);
            buffer.put(idx2 + 1, v);
        }

        private void updateCoverage(Frame frame) {
            Camera camera = frame.getCamera();
            Pose pose = camera.getPose();
            float[] translation = pose.getTranslation();
            if (!finite(translation)) return;
            PointCloud cloud = null;
            try {
                cloud = frame.acquirePointCloud();
                FloatBuffer points = cloud.getPoints();
                points.rewind();
                while (points.remaining() >= 4) {
                    float x = points.get(), y = points.get(), z = points.get();
                    float confidence = points.get();
                    if (!finite(x, y, z, confidence) || confidence < 0.15f) continue;
                    String key = voxelKey(x, y, z, VOXEL);
                    CoveragePoint point = coverage.get(key);
                    if (point == null) {
                        if (coverage.size() >= MAX_POINTS) continue;
                        point = new CoveragePoint(x, y, z); coverage.put(key, point);
                    }
                    point.observe(translation[0] - x, translation[1] - y, translation[2] - z);
                }
            } catch (Exception ignored) { }
            finally { if (cloud != null) cloud.release(); }
        }

        private void drawCoverage(Frame frame) {
            if (coverage.isEmpty() && viewpoints.isEmpty()) return;
            Camera camera = frame.getCamera();
            camera.getProjectionMatrix(projection, 0, 0.05f, 100f);
            camera.getViewMatrix(view, 0);
            Matrix.multiplyMM(mvp, 0, projection, 0, view, 0);
            if (!coverage.isEmpty()) {
                // Keep ARCore's projected screen coordinates unchanged so the sparse
                // cloud shares the camera image's display orientation.
                pointPositionsBuffer.clear(); pointColorsBuffer.clear();
                int count = 0;
                for (CoveragePoint point : coverage.values()) {
                    if (count >= MAX_POINTS) break;
                    pointPositionsBuffer.put(point.xyz);
                    putCoverageColor(pointColorsBuffer, point.maxAngle);
                    count++;
                }
                pointPositionsBuffer.flip(); pointColorsBuffer.flip();
                GLES20.glEnable(GLES20.GL_DEPTH_TEST);
                GLES20.glEnable(GLES20.GL_BLEND);
                GLES20.glBlendFunc(GLES20.GL_SRC_ALPHA, GLES20.GL_ONE_MINUS_SRC_ALPHA);
                GLES20.glUseProgram(pointProgram);
                GLES20.glUniformMatrix4fv(pointMvp, 1, false, mvp, 0);
                GLES20.glEnableVertexAttribArray(pointPosition); GLES20.glEnableVertexAttribArray(pointColor);
                GLES20.glVertexAttribPointer(pointPosition, 3, GLES20.GL_FLOAT, false, 0, pointPositionsBuffer);
                GLES20.glVertexAttribPointer(pointColor, 3, GLES20.GL_FLOAT, false, 0, pointColorsBuffer);
                GLES20.glDrawArrays(GLES20.GL_POINTS, 0, count);
                GLES20.glDisableVertexAttribArray(pointPosition); GLES20.glDisableVertexAttribArray(pointColor);
                GLES20.glDisable(GLES20.GL_BLEND);
            }
            drawViewpoints(mvp);
        }

        /** Draw captured camera poses as world-locked wireframe pyramids. */
        private void drawViewpoints(float[] viewProjection) {
            if (viewpoints.isEmpty()) return;
            viewpointPositionsBuffer.clear(); viewpointColorsBuffer.clear();
            int vertexCount = 0;
            for (int i = 0; i < viewpoints.size() && i < 200; i++) {
                float[] vertices = viewpoints.get(i).lineVertices;
                boolean latest = i == viewpoints.size() - 1;
                float r = latest ? 0.18f : 0.78f;
                float g = latest ? 0.90f : 0.87f;
                float b = latest ? 0.82f : 0.85f;
                for (int vertex = 0; vertex < 16; vertex++) {
                    int offset = vertex * 3;
                    viewpointPositionsBuffer.put(vertices[offset]).put(vertices[offset + 1]).put(vertices[offset + 2]);
                    viewpointColorsBuffer.put(r).put(g).put(b);
                    vertexCount++;
                }
            }
            if (vertexCount == 0) return;
            viewpointPositionsBuffer.flip(); viewpointColorsBuffer.flip();
            // Like the camera overlay in Details, keep the wireframes visible
            // above the sparse point cloud rather than depth-occluding them.
            GLES20.glDisable(GLES20.GL_DEPTH_TEST);
            GLES20.glEnable(GLES20.GL_BLEND);
            GLES20.glBlendFunc(GLES20.GL_SRC_ALPHA, GLES20.GL_ONE_MINUS_SRC_ALPHA);
            GLES20.glUseProgram(pointProgram);
            GLES20.glUniformMatrix4fv(pointMvp, 1, false, viewProjection, 0);
            GLES20.glEnableVertexAttribArray(pointPosition); GLES20.glEnableVertexAttribArray(pointColor);
            GLES20.glVertexAttribPointer(pointPosition, 3, GLES20.GL_FLOAT, false, 0, viewpointPositionsBuffer);
            GLES20.glVertexAttribPointer(pointColor, 3, GLES20.GL_FLOAT, false, 0, viewpointColorsBuffer);
            GLES20.glDrawArrays(GLES20.GL_LINES, 0, vertexCount);
            GLES20.glDisableVertexAttribArray(pointPosition); GLES20.glDisableVertexAttribArray(pointColor);
            GLES20.glDisable(GLES20.GL_BLEND);
        }

        private void putCoverageColor(FloatBuffer colors, float maxAngle) {
            // Four evenly spaced angular stops: red at 0°, orange at 40°,
            // yellow at 80°, and green at 120°, with linear RGB interpolation.
            float degrees = Float.isFinite(maxAngle)
                    ? Math.max(0f, Math.min(120f, (float)Math.toDegrees(maxAngle))) : 0f;
            float t, red, green;
            if (degrees < 40f) {
                t = degrees / 40f;
                red = 1f;
                green = 0.5f * t;
            } else if (degrees < 80f) {
                t = (degrees - 40f) / 40f;
                red = 1f;
                green = 0.5f + 0.5f * t;
            } else {
                t = (degrees - 80f) / 40f;
                red = 1f - t;
                green = 1f;
            }
            colors.put(red).put(green).put(0f);
        }

        private void capture(Frame frame) {
            Image image = null;
            try {
                if (frames.size() >= 200) { showStatus("Maximum of 200 keyframes reached."); return; }
                image = frame.acquireCameraImage();
                byte[] jpeg = jpegFromYuv(image);
                Bitmap bitmap = BitmapFactory.decodeByteArray(jpeg, 0, jpeg.length);
                String name = String.format(Locale.US, "frame_%04d.jpg", frames.size() + 1);
                try (FileOutputStream out = new FileOutputStream(new File(outputDir, name))) { out.write(jpeg); }

                Camera camera = frame.getCamera();
                Pose pose = camera.getPose();
                float[] matrix = new float[16]; pose.toMatrix(matrix, 0);
                CameraIntrinsics intrinsics = camera.getImageIntrinsics();
                float[] focal = new float[2], principal = new float[2];
                int[] imageSize = new int[2];
                intrinsics.getFocalLength(focal, 0);
                intrinsics.getPrincipalPoint(principal, 0);
                intrinsics.getImageDimensions(imageSize, 0);
                if (!finite(matrix) || !finite(focal[0], focal[1], principal[0], principal[1]) ||
                        focal[0] <= 0 || focal[1] <= 0 || imageSize[0] <= 0 || imageSize[1] <= 0) {
                    throw new IllegalStateException("ARCore returned invalid camera pose or intrinsics for this frame.");
                }
                JSONObject item = new JSONObject();
                item.put("file_path", name);
                item.put("transform_matrix", matrixRows(matrix));
                item.put("fl_x", focal[0]); item.put("fl_y", focal[1]);
                item.put("cx", principal[0]); item.put("cy", principal[1]);
                item.put("w", imageSize[0]); item.put("h", imageSize[1]);
                frames.add(item);
                viewpoints.add(new CaptureViewpoint(matrix, focal[0], focal[1], principal[0], principal[1], imageSize[0], imageSize[1]));
                capturedFrameCount = frames.size();
                addColoredSeeds(frame, pose, bitmap, focal, principal, imageSize, name);
                bitmap.recycle();
                capturedSeedCount = seeds.size();
                refreshCaptureTelemetry();
                updateCoverage(frame);
                cloudRenderer.capturePosition(pose.getTranslation());
                showStatus("Captured " + frames.size() + " / 200 · " + coverage.size() + " coverage points · " +
                        (autoCaptureEnabled ? "auto every " + captureSpacingCm + " cm" : "manual capture"));
            } catch (NotYetAvailableException e) {
                showStatus("Camera image is not ready yet. Hold steady and capture again.");
            } catch (Exception e) {
                showStatus("Capture failed: " + safeMessage(e));
            } finally {
                if (image != null) image.close();
            }
        }

        private void addColoredSeeds(Frame frame, Pose cameraPose, Bitmap image, float[] focal, float[] principal, int[] dimensions, String frameName) {
            PointCloud cloud = null;
            try {
                cloud = frame.acquirePointCloud();
                Pose worldToCamera = cameraPose.inverse();
                float[] world = new float[3], local = new float[3];
                FloatBuffer points = cloud.getPoints(); points.rewind();
                while (points.remaining() >= 4 && seeds.size() < 30000) {
                    world[0] = points.get(); world[1] = points.get(); world[2] = points.get();
                    float confidence = points.get();
                    if (!finite(world[0], world[1], world[2], confidence) || confidence < 0.25f) continue;
                    worldToCamera.transformPoint(world, 0, local, 0);
                    if (!finite(local[0], local[1], local[2])) continue;
                    float z = -local[2];
                    if (z <= 0.05f) continue;
                    int u = Math.round(focal[0] * local[0] / z + principal[0]);
                    int v = Math.round(principal[1] - focal[1] * local[1] / z);
                    if (u < 0 || v < 0 || u >= dimensions[0] || v >= dimensions[1] || u >= image.getWidth() || v >= image.getHeight()) continue;
                    String key = voxelKey(world[0], world[1], world[2], VOXEL);
                    if (seeds.containsKey(key)) continue;
                    int color = image.getPixel(u, v);
                    SeedPoint seed = new SeedPoint(world[0], world[1], world[2],
                            ((color >> 16) & 255) / 255f, ((color >> 8) & 255) / 255f, (color & 255) / 255f, frameName);
                    seeds.put(key, seed);
                }
            } catch (Exception ignored) { }
            finally { if (cloud != null) cloud.release(); }
        }

        void resetData() {
            coverage.clear(); seeds.clear(); frames.clear(); viewpoints.clear(); capturedFrameCount = 0; capturedSeedCount = 0;
            refreshCaptureTelemetry();
            travelTrigger.reset();
            deleteTree(outputDir); outputDir.mkdirs();
            showStatus("Capture reset. Move around the subject and capture at least 3 views.");
        }

        void writeDataset(File dir) throws Exception {
            if (frames.size() < 3) throw new IllegalStateException("Capture at least 3 frames first.");
            JSONObject transforms = new JSONObject();
            JSONObject first = frames.get(0);
            int w = first.getInt("w"), h = first.getInt("h");
            float fx = (float) first.getDouble("fl_x");
            if (w <= 0 || h <= 0 || !Float.isFinite(fx) || fx <= 0) {
                throw new IllegalStateException("The captured camera intrinsics are incomplete.");
            }
            transforms.put("camera_angle_x", 2.0 * Math.atan(w / (2.0 * fx)));
            for (String key : new String[]{"fl_x", "fl_y", "cx", "cy", "w", "h"}) transforms.put(key, first.get(key));
            JSONArray frameArray = new JSONArray();
            for (JSONObject frame : frames) frameArray.put(frame);
            transforms.put("frames", frameArray);
            writeText(new File(dir, "transforms.json"), jsonText(transforms, 2, "transforms.json"));

            JSONArray points = new JSONArray();
            for (SeedPoint seed : seeds.values()) {
                JSONObject p = new JSONObject();
                p.put("X", new JSONArray(new float[]{seed.x, seed.y, seed.z}));
                p.put("rgb", new JSONArray(new float[]{seed.r, seed.g, seed.b}));
                p.put("frameName", seed.frameName);
                points.put(p);
            }
            JSONObject pointDocument = new JSONObject();
            pointDocument.put("points", points);
            pointDocument.put("metadata", new JSONObject().put("seedSource", seeds.isEmpty() ? "none" : "arcore-point-cloud-rgb"));
            writeText(new File(dir, "seed_points.json"), jsonText(pointDocument, 0, "seed_points.json"));
            writePly(new File(dir, "seed.ply"));
            JSONObject metadata = new JSONObject()
                    .put("frames", frames.size()).put("pointCount", seeds.size())
                    .put("seedSource", seeds.isEmpty() ? "none" : "arcore-point-cloud-rgb")
                    .put("coordinateConvention", "ARCore world meters; OpenGL camera-to-world transforms; RGB measured from matching camera image");
            writeText(new File(dir, "capture.json"), jsonText(metadata, 2, "capture.json"));
        }

        private void writePly(File file) throws Exception {
            try (DataOutputStream out = new DataOutputStream(new FileOutputStream(file))) {
                String header = "ply\nformat binary_little_endian 1.0\nelement vertex " + seeds.size() +
                        "\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n";
                out.write(header.getBytes(java.nio.charset.StandardCharsets.US_ASCII));
                for (SeedPoint p : seeds.values()) {
                    writeFloatLE(out, p.x); writeFloatLE(out, p.y); writeFloatLE(out, p.z);
                    out.writeByte(Math.round(p.r * 255)); out.writeByte(Math.round(p.g * 255)); out.writeByte(Math.round(p.b * 255));
                }
            }
        }

        private byte[] jpegFromYuv(Image image) throws Exception {
            int w = image.getWidth(), h = image.getHeight();
            Image.Plane[] p = image.getPlanes();
            byte[] nv21 = new byte[w * h * 3 / 2];
            ByteBuffer y = p[0].getBuffer(), u = p[1].getBuffer(), v = p[2].getBuffer();
            int yp = p[0].getPixelStride(), yr = p[0].getRowStride();
            for (int row = 0; row < h; row++) for (int col = 0; col < w; col++)
                nv21[row * w + col] = y.get(row * yr + col * yp);
            int offset = w * h, uvRows = h / 2, uvCols = w / 2;
            int up = p[1].getPixelStride(), ur = p[1].getRowStride();
            int vp = p[2].getPixelStride(), vr = p[2].getRowStride();
            for (int row = 0; row < uvRows; row++) for (int col = 0; col < uvCols; col++) {
                nv21[offset++] = v.get(row * vr + col * vp);
                nv21[offset++] = u.get(row * ur + col * up);
            }
            ByteArrayOutputStream out = new ByteArrayOutputStream();
            new YuvImage(nv21, ImageFormat.NV21, w, h, null).compressToJpeg(new Rect(0, 0, w, h), 95, out);
            return out.toByteArray();
        }

        private JSONArray matrixRows(float[] m) throws Exception {
            JSONArray rows = new JSONArray();
            for (int row = 0; row < 4; row++) {
                JSONArray values = new JSONArray();
                for (int col = 0; col < 4; col++) values.put(m[col * 4 + row]);
                rows.put(values);
            }
            return rows;
        }

        private FloatBuffer floats(float[] values) {
            FloatBuffer buffer = ByteBuffer.allocateDirect(values.length * 4).order(ByteOrder.nativeOrder()).asFloatBuffer();
            buffer.put(values).position(0);
            return buffer;
        }

        private int program(String vertex, String fragment) {
            int vs = shader(GLES20.GL_VERTEX_SHADER, vertex), fs = shader(GLES20.GL_FRAGMENT_SHADER, fragment);
            int p = GLES20.glCreateProgram(); GLES20.glAttachShader(p, vs); GLES20.glAttachShader(p, fs); GLES20.glLinkProgram(p);
            int[] ok = new int[1]; GLES20.glGetProgramiv(p, GLES20.GL_LINK_STATUS, ok, 0);
            if (ok[0] == 0) throw new IllegalStateException(GLES20.glGetProgramInfoLog(p));
            GLES20.glDeleteShader(vs); GLES20.glDeleteShader(fs); return p;
        }
        private int shader(int kind, String source) {
            int s = GLES20.glCreateShader(kind); GLES20.glShaderSource(s, source); GLES20.glCompileShader(s);
            int[] ok = new int[1]; GLES20.glGetShaderiv(s, GLES20.GL_COMPILE_STATUS, ok, 0);
            if (ok[0] == 0) throw new IllegalStateException(GLES20.glGetShaderInfoLog(s));
            return s;
        }

        private String voxelKey(float x, float y, float z, float size) {
            return Math.round(x / size) + ":" + Math.round(y / size) + ":" + Math.round(z / size);
        }

        private String jsonText(JSONObject value, int indent, String label) throws Exception {
            String text = indent > 0 ? value.toString(indent) : value.toString();
            if (text == null) throw new IllegalStateException("Could not serialize " + label + "; discard non-finite ARCore values and retry.");
            return text;
        }

        private void writeText(File file, String text) throws Exception {
            if (text == null) throw new IllegalStateException("Refusing to write null capture data to " + file.getName() + ".");
            try (FileOutputStream out = new FileOutputStream(file)) {
                out.write(text.getBytes(java.nio.charset.StandardCharsets.UTF_8));
            }
        }

        private void writeFloatLE(DataOutputStream out, float value) throws Exception {
            int bits = Float.floatToIntBits(value);
            out.writeByte(bits); out.writeByte(bits >>> 8); out.writeByte(bits >>> 16); out.writeByte(bits >>> 24);
        }
    }

    /** Distance-only auto shutter trigger. Tracking gaps establish a new pose baseline. */
    private static final class TravelCaptureTrigger {
        private float spacingMeters;
        private float distanceSinceCapture;
        private float[] lastPosition;
        private static final float NOISE_FLOOR_METERS = 0.002f;

        TravelCaptureTrigger(float spacingMeters) { setSpacingMeters(spacingMeters); }
        void setSpacingMeters(float value) {
            if (!Float.isFinite(value) || value <= 0f) throw new IllegalArgumentException("Spacing must be positive.");
            spacingMeters = value;
            reset();
        }
        void reset() { lastPosition = null; distanceSinceCapture = 0f; }
        void breakContinuity() { lastPosition = null; }
        boolean sample(float[] position) {
            if (position == null || position.length < 3 || !Float.isFinite(position[0]) ||
                    !Float.isFinite(position[1]) || !Float.isFinite(position[2])) {
                breakContinuity();
                return false;
            }
            if (lastPosition == null) { lastPosition = position.clone(); return false; }
            float dx = position[0] - lastPosition[0];
            float dy = position[1] - lastPosition[1];
            float dz = position[2] - lastPosition[2];
            float distance = (float)Math.sqrt(dx * dx + dy * dy + dz * dz);
            if (distance > NOISE_FLOOR_METERS) {
                distanceSinceCapture += distance;
                lastPosition = position.clone();
            }
            return distanceSinceCapture + 1e-6f >= spacingMeters;
        }
        void captured(float[] position) {
            lastPosition = position.clone();
            distanceSinceCapture = 0f;
        }
    }

    private static final class CaptureViewpoint {
        final float[] lineVertices = new float[16 * 3];

        CaptureViewpoint(float[] pose, float fx, float fy, float cx, float cy, int width, int height) {
            final float depth = 0.012f;
            final float[] origin = {pose[12], pose[13], pose[14]};
            final float[][] corners = new float[4][3];
            final float[] px = {0f, width, width, 0f};
            final float[] py = {0f, 0f, height, height};
            for (int i = 0; i < 4; i++) {
                float x = (px[i] - cx) / fx * depth;
                float y = (cy - py[i]) / fy * depth;
                float z = -depth; // ARCore camera looks down its local -Z axis.
                corners[i][0] = pose[0] * x + pose[4] * y + pose[8] * z + pose[12];
                corners[i][1] = pose[1] * x + pose[5] * y + pose[9] * z + pose[13];
                corners[i][2] = pose[2] * x + pose[6] * y + pose[10] * z + pose[14];
            }
            int out = 0;
            for (int i = 0; i < 4; i++) {
                out = putLineVertex(lineVertices, out, origin);
                out = putLineVertex(lineVertices, out, corners[i]);
            }
            for (int i = 0; i < 4; i++) {
                out = putLineVertex(lineVertices, out, corners[i]);
                out = putLineVertex(lineVertices, out, corners[(i + 1) % 4]);
            }
        }

        private static int putLineVertex(float[] target, int offset, float[] point) {
            target[offset++] = point[0];
            target[offset++] = point[1];
            target[offset++] = point[2];
            return offset;
        }
    }

    private static final class CoveragePoint {
        private static final int MAX_DIRECTIONS = 16;
        private static final float MIN_SAMPLE_DOT = (float)Math.cos(Math.toRadians(4.0));
        final float[] xyz;
        // Fixed-size direction samples keep coverage memory bounded. Farthest-
        // direction replacement prevents the first handful of similar views
        // from permanently hiding later viewpoint changes.
        final float[] directions = new float[MAX_DIRECTIONS * 3];
        final float[] nearestDots = new float[MAX_DIRECTIONS];
        // Preserve the pair that established the widest observed angle even if
        // one endpoint is later replaced in the bounded representative set.
        final float[] maxPair = new float[6];
        int directionCount;
        boolean hasMaxPair;
        float maxAngle;

        CoveragePoint(float x, float y, float z) { xyz = new float[]{x, y, z}; }

        void observe(float x, float y, float z) {
            if (!finite(x, y, z)) return;
            float n = (float)Math.sqrt(x*x + y*y + z*z);
            if (!Float.isFinite(n) || n < 1e-6f) return;
            float dx = x / n, dy = y / n, dz = z / n;

            float closestDot = -1f;
            float widestDot = 1f;
            float ax = 0f, ay = 0f, az = 0f;
            boolean foundPair = false;
            for (int i = 0; i < directionCount; i++) {
                int offset = i * 3;
                float dot = clampDot(directions[offset] * dx + directions[offset + 1] * dy + directions[offset + 2] * dz);
                closestDot = Math.max(closestDot, dot);
                if (dot < widestDot) {
                    widestDot = dot;
                    ax = directions[offset]; ay = directions[offset + 1]; az = directions[offset + 2];
                    foundPair = true;
                }
            }
            if (hasMaxPair) {
                for (int endpoint = 0; endpoint <= 3; endpoint += 3) {
                    float dot = clampDot(maxPair[endpoint] * dx + maxPair[endpoint + 1] * dy + maxPair[endpoint + 2] * dz);
                    if (dot < widestDot) {
                        widestDot = dot;
                        ax = maxPair[endpoint]; ay = maxPair[endpoint + 1]; az = maxPair[endpoint + 2];
                        foundPair = true;
                    }
                }
            }
            if (foundPair && widestDot < Math.cos(maxAngle)) {
                maxAngle = (float)Math.acos(widestDot);
                maxPair[0] = ax; maxPair[1] = ay; maxPair[2] = az;
                maxPair[3] = dx; maxPair[4] = dy; maxPair[5] = dz;
                hasMaxPair = true;
            }

            if (directionCount < MAX_DIRECTIONS) {
                if (directionCount == 0 || closestDot < MIN_SAMPLE_DOT) addDirection(dx, dy, dz);
            } else {
                int redundant = 1; // keep the first captured viewing direction as a stable anchor
                for (int i = 2; i < directionCount; i++) {
                    if (nearestDots[i] > nearestDots[redundant]) redundant = i;
                }
                if (closestDot < nearestDots[redundant]) {
                    int offset = redundant * 3;
                    directions[offset] = dx; directions[offset + 1] = dy; directions[offset + 2] = dz;
                    recomputeNearestDots();
                }
            }
        }

        private void addDirection(float x, float y, float z) {
            int offset = directionCount * 3;
            float nearest = -1f;
            for (int i = 0; i < directionCount; i++) {
                int other = i * 3;
                float dot = clampDot(directions[other] * x + directions[other + 1] * y + directions[other + 2] * z);
                nearest = Math.max(nearest, dot);
                nearestDots[i] = Math.max(nearestDots[i], dot);
            }
            directions[offset] = x; directions[offset + 1] = y; directions[offset + 2] = z;
            nearestDots[directionCount] = nearest;
            directionCount++;
        }

        private void recomputeNearestDots() {
            for (int i = 0; i < directionCount; i++) {
                float nearest = -1f;
                int a = i * 3;
                for (int j = 0; j < directionCount; j++) {
                    if (i == j) continue;
                    int b = j * 3;
                    float dot = clampDot(directions[a] * directions[b] + directions[a + 1] * directions[b + 1] + directions[a + 2] * directions[b + 2]);
                    nearest = Math.max(nearest, dot);
                }
                nearestDots[i] = nearest;
            }
        }

        private static float clampDot(float dot) { return Math.max(-1f, Math.min(1f, dot)); }
    }

    private static final class SeedPoint {
        final float x,y,z,r,g,b; final String frameName;
        SeedPoint(float x, float y, float z, float r, float g, float b, String frameName) {
            this.x=x; this.y=y; this.z=z; this.r=r; this.g=g; this.b=b; this.frameName=frameName;
        }
    }

    private static boolean finite(float... values) {
        for (float value : values) if (!Float.isFinite(value)) return false;
        return true;
    }
    private static String safeMessage(Exception e) { return e.getMessage() == null ? e.getClass().getSimpleName() : e.getMessage(); }
    private static String readText(File f) throws Exception {
        ByteArrayOutputStream bytes = new ByteArrayOutputStream();
        try (FileInputStream in = new FileInputStream(f)) {
            byte[] buffer = new byte[8192];
            int count;
            while ((count = in.read(buffer)) != -1) bytes.write(buffer, 0, count);
        }
        return new String(bytes.toByteArray(), java.nio.charset.StandardCharsets.UTF_8);
    }
    private static void deleteTree(File file) {
        if (file == null || !file.exists()) return;
        File[] children = file.listFiles();
        if (children != null) for (File child : children) deleteTree(child);
        file.delete();
    }
}
