package space.arrival.splatxr;

import android.Manifest;
import android.app.Activity;
import android.content.Context;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.ImageFormat;
import android.graphics.Rect;
import android.graphics.YuvImage;
import android.media.Image;
import android.text.InputType;
import android.opengl.GLES11Ext;
import android.opengl.GLES20;
import android.opengl.GLSurfaceView;
import android.opengl.Matrix;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.view.Gravity;
import android.view.Surface;
import android.view.inputmethod.EditorInfo;
import android.widget.Button;
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
import java.io.FileOutputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.ShortBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.atomic.AtomicBoolean;
import javax.microedition.khronos.egl.EGLConfig;
import javax.microedition.khronos.opengles.GL10;

/** Native ARCore capture surface. Keyframes, intrinsics and poses are sampled from the same ARCore frame. */
public final class NativeCaptureActivity extends Activity implements GLSurfaceView.Renderer {
    private static volatile NativeCaptureActivity active;
    private static final int CAMERA_PERMISSION = 71;
    private Session session;
    private GLSurfaceView glView;
    private TextView status;
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

        FrameLayout root = new FrameLayout(this);
        glView = new GLSurfaceView(this);
        glView.setEGLContextClientVersion(3);
        glView.setPreserveEGLContextOnPause(true);
        glView.setRenderer(this);
        glView.setRenderMode(GLSurfaceView.RENDERMODE_CONTINUOUSLY);
        root.addView(glView, new FrameLayout.LayoutParams(-1, -1));

        LinearLayout controls = new LinearLayout(this);
        controls.setOrientation(LinearLayout.VERTICAL);
        controls.setPadding(dp(16), dp(12), dp(16), dp(16));
        controls.setBackgroundColor(0xAA0E1112);
        status = new TextView(this);
        status.setTextColor(0xFFFFFFFF);
        status.setTextSize(14);
        status.setText("Starting ARCore… Move slowly around the subject.");
        controls.addView(status, new LinearLayout.LayoutParams(-1, -2));

        LinearLayout captureOptions = new LinearLayout(this);
        captureOptions.setGravity(Gravity.CENTER_VERTICAL);
        CheckBox autoToggle = new CheckBox(this);
        autoToggle.setText("Auto-capture");
        autoToggle.setTextColor(0xFFFFFFFF);
        autoToggle.setChecked(false); // Match WebXR: manual shutter is the default.
        captureOptions.addView(autoToggle, new LinearLayout.LayoutParams(-2, dp(44)));
        TextView everyLabel = new TextView(this);
        everyLabel.setText("Every");
        everyLabel.setTextColor(0xFFFFFFFF);
        LinearLayout.LayoutParams everyParams = new LinearLayout.LayoutParams(-2, -2);
        everyParams.setMargins(dp(12), 0, dp(6), 0);
        captureOptions.addView(everyLabel, everyParams);
        spacingInput = new EditText(this);
        spacingInput.setSingleLine(true);
        spacingInput.setInputType(InputType.TYPE_CLASS_NUMBER);
        spacingInput.setImeOptions(EditorInfo.IME_ACTION_DONE);
        spacingInput.setText(String.valueOf(captureSpacingCm));
        spacingInput.setTextColor(0xFFFFFFFF);
        spacingInput.setSelectAllOnFocus(true);
        spacingInput.setContentDescription("Auto-capture travel spacing in centimeters, from 2 to 200");
        captureOptions.addView(spacingInput, new LinearLayout.LayoutParams(dp(64), dp(44)));
        TextView cmLabel = new TextView(this);
        cmLabel.setText("cm traveled");
        cmLabel.setTextColor(0xFFFFFFFF);
        LinearLayout.LayoutParams cmParams = new LinearLayout.LayoutParams(-2, -2);
        cmParams.setMargins(dp(6), 0, 0, 0);
        captureOptions.addView(cmLabel, cmParams);
        controls.addView(captureOptions, new LinearLayout.LayoutParams(-1, -2));

        autoToggle.setOnCheckedChangeListener((view, enabled) -> {
            autoCaptureEnabled = enabled;
            glView.queueEvent(cloudRenderer::resetDistanceBaseline);
            showStatus(enabled
                    ? "Auto-capture enabled: one photo every " + captureSpacingCm + " cm traveled. Move slowly."
                    : "Manual capture enabled. Press Capture frame to take a photo.");
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

        LinearLayout buttons = new LinearLayout(this);
        buttons.setGravity(Gravity.CENTER);
        Button shutter = button("Capture frame");
        shutter.setOnClickListener(v -> {
            if (session == null || !sessionResumed) { showStatus("AR tracking is not ready yet."); return; }
            captureRequested.set(true);
        });
        Button reset = button("Reset");
        reset.setOnClickListener(v -> {
            captureRequested.set(false);
            glView.queueEvent(() -> {
                cloudRenderer.resetData();
                deleteTree(outputDir);
                outputDir.mkdirs();
                showStatus("Capture reset. Move around the subject and capture at least 3 views.");
            });
        });
        Button finish = button("Finish & train");
        finish.setOnClickListener(v -> finishCapture());
        buttons.addView(shutter, weightedButton());
        buttons.addView(reset, weightedButton());
        buttons.addView(finish, weightedButton());
        controls.addView(buttons, new LinearLayout.LayoutParams(-1, -2));
        FrameLayout.LayoutParams controlParams = new FrameLayout.LayoutParams(-1, -2, Gravity.BOTTOM);
        root.addView(controls, controlParams);
        setContentView(root);
    }

    private LinearLayout.LayoutParams weightedButton() {
        LinearLayout.LayoutParams p = new LinearLayout.LayoutParams(0, dp(48), 1f);
        p.setMargins(dp(4), dp(10), dp(4), 0);
        return p;
    }
    private Button button(String text) {
        Button b = new Button(this);
        b.setText(text);
        b.setTextSize(12);
        b.setAllCaps(false);
        return b;
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
        main.post(() -> { if (status != null) status.setText(message); });
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
            showStatus("AR tracking ready · capture overlapping views around the subject");
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
        private final float[] projection = new float[16], view = new float[16], mvp = new float[16];
        private final float[] cameraCenter = new float[3];

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
                if (frame.getCamera().getTrackingState() == TrackingState.TRACKING) {
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
                showStatus("AR camera is unavailable. Close other camera apps and retry.");
            } catch (Exception e) {
                showStatus("AR frame error: " + safeMessage(e));
            }
        }

        private void drawBackground(Frame frame) {
            frame.transformDisplayUvCoords(uvInput, uvOutput);
            GLES20.glDisable(GLES20.GL_DEPTH_TEST);
            GLES20.glUseProgram(bgProgram);
            GLES20.glActiveTexture(GLES20.GL_TEXTURE0);
            GLES20.glBindTexture(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, cameraTexture);
            GLES20.glUniform1i(bgTexture, 0);
            quadPositions.position(0); uvOutput.position(0);
            GLES20.glEnableVertexAttribArray(bgPosition); GLES20.glEnableVertexAttribArray(bgUv);
            GLES20.glVertexAttribPointer(bgPosition, 2, GLES20.GL_FLOAT, false, 0, quadPositions);
            GLES20.glVertexAttribPointer(bgUv, 2, GLES20.GL_FLOAT, false, 0, uvOutput);
            GLES20.glDrawArrays(GLES20.GL_TRIANGLE_STRIP, 0, 4);
            GLES20.glDisableVertexAttribArray(bgPosition); GLES20.glDisableVertexAttribArray(bgUv);
        }

        private void updateCoverage(Frame frame) {
            Camera camera = frame.getCamera();
            Pose pose = camera.getPose();
            float[] translation = pose.getTranslation();
            PointCloud cloud = null;
            try {
                cloud = frame.acquirePointCloud();
                FloatBuffer points = cloud.getPoints();
                points.rewind();
                while (points.remaining() >= 4) {
                    float x = points.get(), y = points.get(), z = points.get();
                    float confidence = points.get();
                    if (confidence < 0.15f) continue;
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
            if (coverage.isEmpty()) return;
            Camera camera = frame.getCamera();
            camera.getProjectionMatrix(projection, 0, 0.05f, 100f);
            camera.getViewMatrix(view, 0);
            Matrix.multiplyMM(mvp, 0, projection, 0, view, 0);
            pointPositionsBuffer.clear(); pointColorsBuffer.clear();
            int count = 0;
            for (CoveragePoint point : coverage.values()) {
                if (count >= MAX_POINTS) break;
                pointPositionsBuffer.put(point.xyz);
                float s = Math.min(1f, point.maxAngle / (float)(Math.PI / 2.0));
                // Red at one viewing direction, yellow around 45°, green at 90° diversity.
                pointColorsBuffer.put(1f - s).put(s).put(0.02f);
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
                JSONObject item = new JSONObject();
                item.put("file_path", name);
                item.put("transform_matrix", matrixRows(matrix));
                item.put("fl_x", focal[0]); item.put("fl_y", focal[1]);
                item.put("cx", principal[0]); item.put("cy", principal[1]);
                item.put("w", imageSize[0]); item.put("h", imageSize[1]);
                frames.add(item);
                capturedFrameCount = frames.size();
                addColoredSeeds(frame, pose, bitmap, focal, principal, imageSize, name);
                bitmap.recycle();
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
                    if (confidence < 0.25f) continue;
                    worldToCamera.transformPoint(world, 0, local, 0);
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
            coverage.clear(); seeds.clear(); frames.clear(); capturedFrameCount = 0;
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
            transforms.put("camera_angle_x", 2.0 * Math.atan(w / (2.0 * fx)));
            for (String key : new String[]{"fl_x", "fl_y", "cx", "cy", "w", "h"}) transforms.put(key, first.get(key));
            JSONArray frameArray = new JSONArray();
            for (JSONObject frame : frames) frameArray.put(frame);
            transforms.put("frames", frameArray);
            writeText(new File(dir, "transforms.json"), transforms.toString(2));

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
            writeText(new File(dir, "seed_points.json"), pointDocument.toString());
            writePly(new File(dir, "seed.ply"));
            writeText(new File(dir, "capture.json"), new JSONObject()
                    .put("frames", frames.size()).put("pointCount", seeds.size())
                    .put("seedSource", seeds.isEmpty() ? "none" : "arcore-point-cloud-rgb")
                    .put("coordinateConvention", "ARCore world meters; OpenGL camera-to-world transforms; RGB measured from matching camera image")
                    .toString(2));
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

        private void writeText(File file, String text) throws Exception {
            try (FileOutputStream out = new FileOutputStream(file)) { out.write(text.getBytes(java.nio.charset.StandardCharsets.UTF_8)); }
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

    private static final class CoveragePoint {
        final float[] xyz;
        final List<float[]> directions = new ArrayList<>();
        float maxAngle;
        CoveragePoint(float x, float y, float z) { xyz = new float[]{x, y, z}; }
        void observe(float x, float y, float z) {
            float n = (float)Math.sqrt(x*x + y*y + z*z);
            if (n < 1e-6) return;
            float[] d = {x/n, y/n, z/n};
            for (float[] old : directions) maxAngle = Math.max(maxAngle,
                    (float)Math.acos(Math.max(-1f, Math.min(1f, old[0]*d[0] + old[1]*d[1] + old[2]*d[2]))));
            if (directions.size() < 8) directions.add(d);
        }
    }

    private static final class SeedPoint {
        final float x,y,z,r,g,b; final String frameName;
        SeedPoint(float x, float y, float z, float r, float g, float b, String frameName) {
            this.x=x; this.y=y; this.z=z; this.r=r; this.g=g; this.b=b; this.frameName=frameName;
        }
    }

    private static String safeMessage(Exception e) { return e.getMessage() == null ? e.getClass().getSimpleName() : e.getMessage(); }
    private static String readText(File f) throws Exception { return new String(java.nio.file.Files.readAllBytes(f.toPath()), java.nio.charset.StandardCharsets.UTF_8); }
    private static void deleteTree(File file) {
        if (file == null || !file.exists()) return;
        File[] children = file.listFiles();
        if (children != null) for (File child : children) deleteTree(child);
        file.delete();
    }
}
