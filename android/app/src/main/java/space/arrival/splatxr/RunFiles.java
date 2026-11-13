package space.arrival.splatxr;

import android.content.Context;
import android.graphics.Bitmap;
import android.util.Base64;

import org.json.JSONObject;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.regex.Pattern;

/** The migrate-on-swipe shared bus: files under getFilesDir()/runs/.
 *
 *  Used from BOTH processes — the Activity's staging/checkpoint writers and
 *  the :training service's headless writers — so every method is a static over
 *  plain files: no binder, no parcel limits, no cross-process locking. Each
 *  side only writes while it owns the run, and every publish is tmp-file +
 *  rename, so readers never see a torn file.
 *
 *  Layout (see app/js/headless.js, the web twin of this contract):
 *    runs/active.json              { runId }
 *    runs/<runId>/manifest.json    run parameters
 *    runs/<runId>/photos.json      { names, complete } | { error } | { pending }
 *    runs/<runId>/photos/<name>    the training photos, byte-identical
 *    runs/<runId>/state.bin        the raw trainer state
 *    runs/<runId>/recon.json       cameras/frames/recipe/cloud
 *    runs/<runId>/progress.json    the service notification's numbers
 *    runs/<runId>/status.json      { status, owner, iter, at, error? }
 *    runs/<runId>/service.json     { alive, at } — :training heartbeat
 *    runs/<runId>/activity.json    { alive, at } — Activity liveness beat
 *    runs/<runId>/snapshot.png + snapshot.json — the monitor view
 */
public final class RunFiles {
    private RunFiles() {}

    // same character classes the web side (headless.js) enforces — anything
    // outside these never touches the filesystem from either side
    private static final Pattern RUN_ID = Pattern.compile("[A-Za-z0-9_-]{1,80}");
    private static final Pattern NAME = Pattern.compile("[A-Za-z0-9_.-]{1,120}");
    private static final long MAX_SNAPSHOT_PX = 1024 * 1024;
    private static final int MAX_JSON_BYTES = 2 * 1024 * 1024;

    public static boolean validRunId(String id) {
        return id != null && RUN_ID.matcher(id).matches();
    }

    public static boolean validName(String name) {
        return name != null && NAME.matcher(name).matches() &&
                !name.contains("..") && !name.equals(".");
    }

    public static File runsDir(Context ctx) {
        File d = new File(ctx.getFilesDir(), "runs");
        if (!d.exists()) d.mkdirs();
        return d;
    }

    public static File runDir(Context ctx, String runId) {
        File d = new File(runsDir(ctx), runId);
        if (!d.exists()) d.mkdirs();
        return d;
    }

    /** Resolve a bridge file reference: 'state.bin' or 'photos/<photo>'.
     *  Anything else (escapes, unknown dirs) resolves to null — never written. */
    public static File runFile(Context ctx, String runId, String ref) {
        if (!validRunId(runId) || ref == null) return null;
        File base = runDir(ctx, runId);
        int slash = ref.indexOf('/');
        if (slash < 0) {
            if (!validName(ref)) return null;
            return new File(base, ref);
        }
        String dir = ref.substring(0, slash);
        String name = ref.substring(slash + 1);
        if (!dir.equals("photos") || !validName(name) || name.contains("/")) return null;
        File photos = new File(base, "photos");
        if (!photos.exists()) photos.mkdirs();
        return new File(photos, name);
    }

    // ── atomic small writes ─────────────────────────────────────────────────
    public static boolean writeJson(Context ctx, String runId, String name, String json) {
        File f = runFile(ctx, runId, name);
        if (f == null || json == null || json.length() > MAX_JSON_BYTES) return false;
        return writeAtomic(f, json.getBytes(StandardCharsets.UTF_8));
    }

    public static String readJson(Context ctx, String runId, String name) {
        File f = runFile(ctx, runId, name);
        if (f == null || !f.isFile() || f.length() > MAX_JSON_BYTES) return null;
        try {
            return new String(readAll(f), StandardCharsets.UTF_8);
        } catch (Exception e) {
            return null;
        }
    }

    private static boolean writeAtomic(File f, byte[] data) {
        File tmp = new File(f.getParent(), f.getName() + ".tmp");
        try {
            OutputStream out = new FileOutputStream(tmp);
            try {
                out.write(data);
            } finally {
                out.close();
            }
            if (f.exists() && !f.delete()) return false;
            return tmp.renameTo(f);
        } catch (Exception e) {
            try { tmp.delete(); } catch (Exception ignored) { }
            return false;
        }
    }

    // ── chunked large writes (base64 over the JS bridge) ────────────────────
    private static final Map<String, OutputStream> chunks = new HashMap<>();
    private static final Map<String, ByteArrayOutputStream> snaps = new HashMap<>();

    /** Append a base64 chunk; on last, the tmp file renames over the target. */
    public static synchronized boolean writeChunk(Context ctx, String runId, String ref,
            String base64, boolean last) {
        File f = runFile(ctx, runId, ref);
        if (f == null || base64 == null) return false;
        String key = f.getAbsolutePath();
        File tmp = new File(f.getParent(), f.getName() + ".tmp");
        try {
            OutputStream out = chunks.get(key);
            if (out == null) {
                try { tmp.delete(); } catch (Exception ignored) { }
                out = new FileOutputStream(tmp, false);
                chunks.put(key, out);
            }
            out.write(Base64.decode(base64, Base64.DEFAULT));
            if (!last) return true;
            try { out.close(); } catch (Exception ignored) { }
            chunks.remove(key);
            if (f.exists() && !f.delete()) {
                try { tmp.delete(); } catch (Exception ignored) { }
                return false;
            }
            return tmp.renameTo(f);
        } catch (Exception e) {
            OutputStream out = chunks.remove(key);
            if (out != null) try { out.close(); } catch (Exception ignored) { }
            try { tmp.delete(); } catch (Exception ignored) { }
            return false;
        }
    }

    /** Assemble raw-RGBA snapshot chunks, then publish snapshot.png (+ meta).
     *  The byte count must equal w*h*4 exactly — anything else is a torn
     *  upload and publishes nothing. */
    public static synchronized boolean writeSnapshotChunk(Context ctx, String runId, int w, int h,
            String base64, boolean last) {
        if (!validRunId(runId) || base64 == null) return false;
        if (w < 2 || h < 2 || (long) w * h > MAX_SNAPSHOT_PX) return false;
        String key = runId + "/" + w + "x" + h;
        try {
            ByteArrayOutputStream buf = snaps.get(key);
            if (buf == null) {
                buf = new ByteArrayOutputStream();
                snaps.put(key, buf);
            }
            byte[] raw = Base64.decode(base64, Base64.DEFAULT);
            if (buf.size() + raw.length > (long) w * h * 4) {
                throw new IllegalArgumentException("snapshot overflow");
            }
            buf.write(raw);
            if (!last) return true;
            snaps.remove(key);
            byte[] rgba = buf.toByteArray();
            if (rgba.length != w * h * 4) return false;
            // RGBA bytes in memory are ARGB_8888 pixels on little-endian
            Bitmap bmp = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888);
            bmp.copyPixelsFromBuffer(ByteBuffer.wrap(rgba));
            File dir = runDir(ctx, runId);
            File pngTmp = new File(dir, "snapshot.png.tmp");
            OutputStream out = new FileOutputStream(pngTmp);
            try {
                if (!bmp.compress(Bitmap.CompressFormat.PNG, 100, out)) return false;
            } finally {
                out.close();
            }
            File png = new File(dir, "snapshot.png");
            if (png.exists() && !png.delete()) return false;
            if (!pngTmp.renameTo(png)) return false;
            JSONObject meta = new JSONObject();
            meta.put("w", w);
            meta.put("h", h);
            meta.put("at", System.currentTimeMillis());
            try {
                String prog = readJson(ctx, runId, "progress.json");
                if (prog != null) meta.put("iter", new JSONObject(prog).optInt("iter", 0));
            } catch (Exception ignored) { }
            return writeJson(ctx, runId, "snapshot.json", meta.toString());
        } catch (Exception e) {
            snaps.remove(key);
            return false;
        }
    }

    // ── the active marker (runs/active.json) ────────────────────────────────
    public static String readActive(Context ctx) {
        File f = new File(runsDir(ctx), "active.json");
        if (!f.isFile() || f.length() > 4096) return null;
        try {
            String runId = new JSONObject(new String(readAll(f), StandardCharsets.UTF_8))
                    .optString("runId", null);
            return validRunId(runId) ? runId : null;
        } catch (Exception e) {
            return null;
        }
    }

    public static boolean writeActive(Context ctx, String runId) {
        if (!validRunId(runId)) return false;
        try {
            JSONObject o = new JSONObject();
            o.put("runId", runId);
            return writeAtomic(new File(runsDir(ctx), "active.json"),
                    o.toString().getBytes(StandardCharsets.UTF_8));
        } catch (Exception e) {
            return false;
        }
    }

    /** Recover a marker cleared by an interrupted Activity handback. The run
     *  directory is the durable transaction: require its exact state, recon,
     *  manifest and retained photos. Deliberately dismissed/finished runs are
     *  excluded, so normal Back and Discard never resurrect. */
    public static String recoverActive(Context ctx) {
        String active = readActive(ctx);
        if (active != null) return active;
        File[] dirs = runsDir(ctx).listFiles(File::isDirectory);
        if (dirs == null) return null;
        String best = null;
        long bestAt = 0;
        for (File dir : dirs) {
            String id = dir.getName();
            if (!validRunId(id)) continue;
            File state = new File(dir, "state.bin");
            File recon = new File(dir, "recon.json");
            File manifest = new File(dir, "manifest.json");
            if (!state.isFile() || state.length() < 16 || !recon.isFile() || !manifest.isFile()) continue;
            try {
                String photosRaw = readJson(ctx, id, "photos.json");
                String statusRaw = readJson(ctx, id, "status.json");
                if (photosRaw == null || statusRaw == null) continue;
                JSONObject photos = new JSONObject(photosRaw);
                if (!photos.optBoolean("complete", photos.optBoolean("done", false))) continue;
                JSONObject status = new JSONObject(statusRaw);
                String value = status.optString("status", "");
                if ("finished".equals(value) || "discarded".equals(value) || "dismissed".equals(value)) continue;
                if (!("staging".equals(value) || "booting".equals(value) || "active".equals(value) ||
                        "paused".equals(value) || "interrupted".equals(value))) continue;
                long at = Math.max(status.optLong("at", 0), state.lastModified());
                if (best == null || at > bestAt) { best = id; bestAt = at; }
            } catch (Exception ignored) { /* malformed dirs are not recovery candidates */ }
        }
        return best != null && writeActive(ctx, best) ? best : null;
    }

    /** Clear only when the marker still names runId — a newer run's claim is
     *  never touched. */
    public static boolean clearActive(Context ctx, String runId) {
        String active = readActive(ctx);
        if (active == null) return true;
        if (runId != null && !runId.equals(active)) return false;
        try {
            return new File(runsDir(ctx), "active.json").delete();
        } catch (Exception e) {
            return false;
        }
    }

    // ── retain + cleanup (Activity side) ────────────────────────────────────
    /** Copy the native capture's frames into the run dir (zero bridge bytes).
     *  Returns the copied names; the web side matches them against the run's
     *  own photo list and falls back to uploading anything missing. */
    public static List<String> retainCaptureFrames(Context ctx, String runId) {
        List<String> kept = new ArrayList<>();
        File current = new File(new File(ctx.getFilesDir(), "captures"), "current");
        File[] frames = current.listFiles((dir, name) -> name.matches("frame_[0-9]{4}\\.jpg"));
        if (frames == null) return kept;
        java.util.Arrays.sort(frames, java.util.Comparator.comparing(File::getName));
        File photos = new File(runDir(ctx, runId), "photos");
        if (!photos.exists()) photos.mkdirs();
        byte[] buf = new byte[65536];
        for (File src : frames) {
            try {
                InputStream in = new FileInputStream(src);
                try {
                    OutputStream out = new FileOutputStream(new File(photos, src.getName()));
                    try {
                        int n;
                        while ((n = in.read(buf)) > 0) out.write(buf, 0, n);
                    } finally {
                        out.close();
                    }
                } finally {
                    in.close();
                }
                kept.add(src.getName());
            } catch (Exception ignored) {
                // per-frame best-effort; the manifest check names the missing
            }
        }
        return kept;
    }

    /** Delete every run dir except keepId (staged runs clean predecessors). */
    public static void cleanRunDirs(Context ctx, String keepId) {
        File runs = runsDir(ctx);
        File[] dirs = runs.listFiles();
        if (dirs == null) return;
        for (File d : dirs) {
            if (!d.isDirectory()) continue;
            if (keepId != null && d.getName().equals(keepId)) continue;
            deleteRecursive(d);
        }
    }

    public static void deleteRecursive(File f) {
        if (f.isDirectory()) {
            File[] kids = f.listFiles();
            if (kids != null) for (File k : kids) deleteRecursive(k);
        }
        try { f.delete(); } catch (Exception ignored) { }
    }

    private static byte[] readAll(File f) throws Exception {
        long len = f.length();
        if (len > MAX_JSON_BYTES) throw new IllegalArgumentException("file too large");
        byte[] data = new byte[(int) len];
        InputStream in = new FileInputStream(f);
        try {
            int o = 0, n;
            while (o < data.length && (n = in.read(data, o, data.length - o)) > 0) o += n;
            if (o != data.length) throw new IllegalArgumentException("short read");
            return data;
        } finally {
            in.close();
        }
    }
}
