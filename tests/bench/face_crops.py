# Native-resolution face crops as extra training cameras for a person splat.
#
# The body is trained from downscaled frames (900x1600 out of 2160x3840): a
# face that is 133 px ear-to-ear in the 4K frame is 55 px in the training
# target, and on top of that the head moves relative to the body-registered
# camera (median 37 px at 4K on the Lisa orbit), so even a native-resolution
# target would smear. This stage fixes both:
#   1. MediaPipe FaceLandmarker (478 points) on a head window of every 4K frame
#   2. robust DLT triangulation of every landmark across the SOLVED body
#      cameras -> a canonical 3D face (the head's dominant position)
#   3. per-frame PnP of that canonical face -> a head-stabilised camera
#      pose for the frame (the head's motion goes into the camera)
#   4. a fixed SIDE x SIDE window around the projected face at native
#      resolution -> crop image + RVM matte (gated by the body matte); the
#      crop camera keeps the native focal and an off-centre principal point
# Writes data/lisaface (body frames + face crops + masks + files.json) and
# two recon JSONs for ?gtrecon=: the training one (held-out crops left out)
# and the full one (for render_views.js evaluation; carries `heldOut`).
#
#   python tests/bench/face_crops.py [--side 768] [--hold 8]
import os, sys, json, time, shutil
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mpp
from mediapipe.tasks.python import vision

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SP = r'C:\Users\trt\AppData\Local\Temp\claude\C--Dev-arrival-space-Browser-3DGS\7b704fd1-8971-4fe5-946e-5241da58e190\scratchpad'
RECON = os.path.join(ROOT, 'scratch', 'lisa_v12_recon.json')
DIR4K = os.path.join(ROOT, 'scratch', 'lisa_4k')
BODY = os.path.join(ROOT, 'data', 'lisa')
OUT = os.path.join(ROOT, 'data', 'lisaface')
arg = lambda k, d: type(d)(sys.argv[sys.argv.index(k) + 1]) if k in sys.argv else d
SIDE = arg('--side', 768)
HOLD = arg('--hold', 8)
MIN_VIEWS = 6
MAX_LM_ERR = 80.0        # px at 4K: landmark kept in the canonical face if its median reprojection is under this

rc = json.load(open(RECON))
frames, cams = rc['frames'], rc['cams']
print(f'{len(cams)} cams, {len(frames)} frames, fFeat {rc["fFeat"]:.1f}')

def K4(c):
    fr = frames[c['imgIdx']]
    S = 2160.0 / fr['fw']
    fy = c.get('fy', c['f'])
    return np.array([[c['f'] * S, 0, c['cx'] * S], [0, fy * S, c['cy'] * S], [0, 0, 1]], np.float64)
def Rt(c):
    return np.array(c['R'], np.float64).reshape(3, 3), np.array(c['t'], np.float64)
P4 = [K4(c) @ np.hstack([Rt(c)[0], Rt(c)[1][:, None]]) for c in cams]

# --- 1. face landmarks at 4K ------------------------------------------------
pose = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
    base_options=mpp.BaseOptions(model_asset_path=os.path.join(SP, 'pose_landmarker_heavy.task')),
    running_mode=vision.RunningMode.IMAGE, num_poses=1))
face = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
    base_options=mpp.BaseOptions(model_asset_path=os.path.join(SP, 'face_landmarker.task')),
    running_mode=vision.RunningMode.IMAGE, num_faces=1, min_face_detection_confidence=0.4))

lm2d = {}       # cam index -> (478, 2) at 4K
t0 = time.time()
for ci, c in enumerate(cams):
    name = frames[c['imgIdx']]['name']
    body = cv2.imread(os.path.join(BODY, name))
    res = pose.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(body, cv2.COLOR_BGR2RGB)))
    if not res.pose_landmarks:
        continue
    P = res.pose_landmarks[0]
    head = [P[i] for i in range(0, 11) if P[i].visibility > 0.3]
    if len(head) < 4:
        continue
    S = 2160.0 / body.shape[1]
    hx = np.array([p.x * body.shape[1] for p in head]) * S
    hy = np.array([p.y * body.shape[0] for p in head]) * S
    ext = max(hx.max() - hx.min(), hy.max() - hy.min())
    win = int(np.clip(3.5 * ext, 384, 1400))
    img = cv2.imread(os.path.join(DIR4K, name))
    x0 = int(np.clip(hx.mean() - win / 2, 0, img.shape[1] - win))
    y0 = int(np.clip(hy.mean() - win / 2, 0, img.shape[0] - win))
    crop = cv2.cvtColor(img[y0:y0 + win, x0:x0 + win], cv2.COLOR_BGR2RGB)
    fres = face.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(crop)))
    if not fres.face_landmarks:
        continue
    L = fres.face_landmarks[0]
    lm2d[ci] = np.array([[p.x * win + x0, p.y * win + y0] for p in L], np.float64)
print(f'face landmarks in {len(lm2d)}/{len(cams)} cams ({time.time() - t0:.0f}s)')
json.dump({str(ci): p.round(2).tolist() for ci, p in lm2d.items()}, open(os.path.join(ROOT, 'scratch', 'face_lm2d.json'), 'w'))   # 2D obs at 4K, for the JS triangulation test
NL = 478

# --- 2. canonical 3D face: robust DLT per landmark across the body cams -------
def dlt(rows):
    A = []
    for ci, u, v in rows:
        P = P4[ci]
        A.append(u * P[2] - P[0]); A.append(v * P[2] - P[1])
    _, _, Vt = np.linalg.svd(np.array(A))
    X = Vt[-1]
    return X[:3] / X[3]
def reproj(X, rows):
    Xh = np.append(X, 1.0); e = []
    for ci, u, v in rows:
        x = P4[ci] @ Xh
        e.append(1e9 if x[2] <= 1e-9 else np.hypot(x[0] / x[2] - u, x[1] / x[2] - v))
    return np.array(e)
def triangulate(rows):
    rows = list(rows)
    if len(rows) < MIN_VIEWS: return None, None, 0
    X = dlt(rows)
    for _ in range(6):
        e = reproj(X, rows)
        cut = max(6.0, 1.5 * np.median(e))
        keep = [r for r, ei in zip(rows, e) if ei <= cut]
        if len(keep) < MIN_VIEWS or len(keep) == len(rows): break
        rows = keep; X = dlt(rows)
    return X, float(np.median(reproj(X, rows))), len(rows)

canon = np.full((NL, 3), np.nan); lmerr = np.full(NL, np.nan); lmviews = np.zeros(NL, int)
for k in range(NL):
    rows = [(ci, p[k, 0], p[k, 1]) for ci, p in lm2d.items()]
    X, med, n = triangulate(rows)
    if X is None: continue
    canon[k] = X; lmerr[k] = med; lmviews[k] = n
good = np.where(np.isfinite(lmerr) & (lmerr <= MAX_LM_ERR))[0]
# the triangulated face (PLY frame) — the head registration target for the body-model fit
json.dump({'frame': 'ply (y down)', 'landmarks': {int(k): canon[k].round(5).tolist() for k in good},
           'medianReprojPx4k': {int(k): round(float(lmerr[k]), 2) for k in good}},
          open(os.path.join(ROOT, 'scratch', 'face_canon3d.json'), 'w'))
print(f'canonical face: {len(good)}/{NL} landmarks, median reproj {np.nanmedian(lmerr[good]):.1f} px at 4K '
      f'(views per landmark median {np.median(lmviews[good]):.0f})')

# --- 3. per-frame PnP: the head's motion goes into the camera ----------------
obj = np.ascontiguousarray(canon[good])
rows = []           # per cam: diagnostics + crop camera
for ci, p in lm2d.items():
    img_pts = np.ascontiguousarray(p[good])
    K = K4(cams[ci])
    R0, t0_ = Rt(cams[ci])
    e_body = reproj(np.zeros(3), [])  # placeholder
    Xh = np.hstack([obj, np.ones((len(obj), 1))])
    xb = (P4[ci] @ Xh.T).T
    e_body = np.hypot(xb[:, 0] / xb[:, 2] - img_pts[:, 0], xb[:, 1] / xb[:, 2] - img_pts[:, 1])
    rvec0, _ = cv2.Rodrigues(R0)
    ok, rvec, tvec, inl = cv2.solvePnPRansac(obj, img_pts, K, None, rvec=rvec0.copy(), tvec=t0_.reshape(3, 1).copy(),
                                             useExtrinsicGuess=True, iterationsCount=2000, reprojectionError=6.0,
                                             confidence=0.999, flags=cv2.SOLVEPNP_ITERATIVE)
    row = {'cam': ci, 'name': frames[cams[ci]['imgIdx']]['name'], 'errBody': float(np.median(e_body)), 'ok': False}
    if ok and inl is not None and len(inl) >= 40:
        inl = inl.ravel()
        rvec, tvec = cv2.solvePnPRefineLM(obj[inl], img_pts[inl], K, None, rvec, tvec)
        R, _ = cv2.Rodrigues(rvec); t = tvec.ravel()
        xp = (K @ (R @ obj.T + t[:, None])).T
        e_pnp = np.hypot(xp[:, 0] / xp[:, 2] - img_pts[:, 0], xp[:, 1] / xp[:, 2] - img_pts[:, 1])
        # a valid head camera must still look from roughly where the body cam is
        C0 = -R0.T @ t0_; C1 = -R.T @ t
        dang = np.degrees(np.arccos(np.clip((np.trace(R0.T @ R) - 1) / 2, -1, 1)))
        row.update({'inliers': int(len(inl)), 'errPnp': float(np.median(e_pnp[inl])), 'errPnpAll': float(np.median(e_pnp)),
                    'rotDeg': float(dang), 'shift': float(np.linalg.norm(C1 - C0)), 'R': R, 't': t})
        row['ok'] = row['errPnp'] <= 4.0 and row['inliers'] >= 60 and dang < 25
    rows.append(row)
okrows = [r for r in rows if r['ok']]
eb = np.array([r['errBody'] for r in rows]); ep = np.array([r['errPnp'] for r in okrows])
print(f'PnP: {len(okrows)}/{len(rows)} frames stabilised; landmark reprojection median '
      f'{np.median(eb):.1f} px (body cam) -> {np.median(ep):.2f} px (head cam); '
      f'head-vs-body pose delta median {np.median([r["rotDeg"] for r in okrows]):.2f} deg, '
      f'p90 {np.percentile([r["rotDeg"] for r in okrows], 90):.2f} deg')

# --- 4./5. crops + mattes ----------------------------------------------------
import onnxruntime as ort
rvm = ort.InferenceSession(os.path.join(SP, 'rvm.onnx'), providers=['CPUExecutionProvider'])
os.makedirs(os.path.join(OUT, 'masks'), exist_ok=True)
for n in os.listdir(BODY):
    if n.endswith('.jpg'): shutil.copy2(os.path.join(BODY, n), os.path.join(OUT, n))
for n in os.listdir(os.path.join(BODY, 'masks')):
    shutil.copy2(os.path.join(BODY, 'masks', n), os.path.join(OUT, 'masks', n))

face_frames, face_cams, sheet, report = [], [], [], []
zero = np.zeros((1, 1, 1, 1), np.float32)
for k, r in enumerate(okrows):
    c = cams[r['cam']]; name = r['name']
    K = K4(c); R, t = r['R'], r['t']
    xp = (K @ (R @ obj.T + t[:, None])).T; xp = xp[:, :2] / xp[:, 2:3]
    bx0, by0, bx1, by1 = xp[:, 0].min(), xp[:, 1].min(), xp[:, 0].max(), xp[:, 1].max()
    cxf, cyf = (bx0 + bx1) / 2, (by0 + by1) / 2 - 0.25 * (by1 - by0)      # hair sits above the face mesh
    img = cv2.imread(os.path.join(DIR4K, name))
    H4, W4 = img.shape[:2]
    x0 = int(np.clip(round(cxf - SIDE / 2), 0, W4 - SIDE)); y0 = int(np.clip(round(cyf - SIDE / 2), 0, H4 - SIDE))
    crop = img[y0:y0 + SIDE, x0:x0 + SIDE]
    # matte: RVM on the native crop, gated by the (dilated) body matte so a
    # room detection far from the person cannot survive
    src = (cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32) / 255).transpose(2, 0, 1)[None]
    _, pha, *_ = rvm.run([], {'src': src, 'r1i': zero, 'r2i': zero, 'r3i': zero, 'r4i': zero,
                              'downsample_ratio': np.array([512.0 / SIDE], np.float32)})
    a = pha[0, 0]
    bm = cv2.imread(os.path.join(BODY, 'masks', name.replace('.jpg', '.png')), cv2.IMREAD_GRAYSCALE)
    bm = cv2.resize(bm, (W4, H4), interpolation=cv2.INTER_LINEAR)[y0:y0 + SIDE, x0:x0 + SIDE]
    gate = cv2.dilate((bm > 8).astype(np.uint8), np.ones((25, 25), np.uint8))
    a = a * gate
    fname = 'face_' + name
    cv2.imwrite(os.path.join(OUT, fname), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
    cv2.imwrite(os.path.join(OUT, 'masks', fname.replace('.jpg', '.png')), (a * 255).round().astype(np.uint8))
    held = (k % HOLD) == HOLD // 2
    face_frames.append({'name': fname, 'fw': SIDE, 'fh': SIDE, 'tw': SIDE, 'th': SIDE})
    face_cams.append({'imgIdx': len(frames) + k, 'name': fname, 'R': R.reshape(-1).tolist(), 't': t.tolist(),
                      'f': float(K[0, 0]), 'fy': float(K[1, 1]), 'cx': float(K[0, 2] - x0), 'cy': float(K[1, 2] - y0),
                      'held': held, 'body': r['cam']})
    report.append({'name': name, 'held': held, 'facePx': float(bx1 - bx0), 'errBody': r['errBody'], 'errPnp': r['errPnp'],
                   'rotDeg': r['rotDeg'], 'shift': r['shift'], 'subject': float((a > 0.5).mean())})
    if k % 12 == 0 and len(sheet) < 12:
        m = np.dstack([a] * 3)
        sheet.append(np.hstack([crop, (crop * m + 255 * (1 - m)).astype(np.uint8)]))
print(f'{len(face_cams)} face crops ({sum(c["held"] for c in face_cams)} held out) -> {OUT}')

# --- 6. datasets + recons ----------------------------------------------------
train_names = sorted(n for n in os.listdir(BODY) if n.endswith('.jpg')) + [c['name'] for c in face_cams if not c['held']]
json.dump(train_names, open(os.path.join(OUT, 'files.json'), 'w'))
base = {k: rc[k] for k in ('version', 'app', 'fFeat', 'k1', 'k2', 'sceneRadius', 'center', 'cloud') if k in rc}
def strip(c): return {k: v for k, v in c.items() if k not in ('held', 'body')}
all_cams = [dict(c) for c in cams] + [strip(c) for c in face_cams]
train_cams = [dict(c) for c in cams] + [strip(c) for c in face_cams if not c['held']]
json.dump({**base, 'name': 'lisa_face', 'frames': frames + face_frames, 'cams': train_cams},
          open(os.path.join(ROOT, 'scratch', 'lisa_face_recon.json'), 'w'))
json.dump({**base, 'name': 'lisa_face_all', 'frames': frames + face_frames, 'cams': all_cams,
           'faceCamStart': len(cams), 'heldOut': [len(cams) + i for i, c in enumerate(face_cams) if c['held']],
           'faceBody': [c['body'] for c in face_cams]},
          open(os.path.join(ROOT, 'scratch', 'lisa_face_recon_all.json'), 'w'))
json.dump({'side': SIDE, 'landmarks': int(len(good)), 'canonErrPx': float(np.nanmedian(lmerr[good])),
           'rows': report}, open(os.path.join(ROOT, 'scratch', 'face_crops_report.json'), 'w'), indent=1)
if sheet:
    cv2.imwrite(os.path.join(ROOT, 'scratch', 'face_crops_sheet.jpg'),
                np.vstack([np.hstack(sheet[i:i + 3]) for i in range(0, len(sheet) - len(sheet) % 3, 3)]), [cv2.IMWRITE_JPEG_QUALITY, 80])
fp = np.array([r['facePx'] for r in report])
print(f'face width in crops: median {np.median(fp):.0f} px, max {fp.max():.0f} px; subject share median {np.median([r["subject"] for r in report]) * 100:.0f}%')
print('done')
