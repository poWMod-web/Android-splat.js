package space.arrival.splatxr;

import android.Manifest;
import android.app.Notification;
import android.app.NotificationChannel;
import android.app.NotificationManager;
import android.app.PendingIntent;
import android.app.Service;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.content.pm.ServiceInfo;
import android.os.Build;
import android.os.IBinder;
import android.os.PowerManager;

/** Foreground service behind the training notification: a live score + splat
 *  count in the shade, process priority while the app runs in the background,
 *  and a partial wake lock so the GPU work is not suspended with the screen.
 *  Driven best-effort from the WebView through MainActivity's bridge. */
public final class TrainingService extends Service {
    private static final String CHANNEL_ID = "splatxr-training";
    private static final int NOTIF_ID = 101;
    private static final int DONE_ID = 102;

    private boolean foreground;
    private PowerManager.WakeLock wakeLock;
    // the last-shown progress, so the paused flag can rebuild the same
    // notification with its button flipped
    private String lastTitle = "";
    private String lastText = "";
    private int lastDone, lastTotal;
    private boolean lastPaused;

    @Override public IBinder onBind(Intent intent) { return null; }

    @Override public int onStartCommand(Intent intent, int flags, int startId) {
        if (intent == null) { stopSelf(); return START_NOT_STICKY; }
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
            default: // "cancel" and anything unknown: leave nothing behind
                teardown();
                break;
        }
        return START_NOT_STICKY;
    }

    @Override public void onTaskRemoved(Intent rootIntent) {
        // the app was swiped away: training died with the WebView, so the
        // notification must not linger as a stale ongoing entry
        teardown();
        super.onTaskRemoved(rootIntent);
    }

    @Override public void onDestroy() {
        notifications().cancel(NOTIF_ID);
        releaseWakeLock();
        foreground = false;
        super.onDestroy();
    }

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
            // activity turns into a WebView toggle call, so the shade
            // button works while the app sits in the background
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
