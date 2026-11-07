# Dense landmark <-> body-model head correspondences, found by running the same
# face landmarker on a shaded RENDER of the fitted head (frontal crop camera):
# each 2D landmark on the render is traced back through the triangle-id buffer
# to a barycentric point on the mesh. Output: scratch/anny_face_corr.json.
#   python tests/bench/head_correspond.py --surface scratch/lisa_anny_surface3_head.json --recon scratch/lisa_face_recon_all.json --cam 193
import sys, os, json, numpy as np, cv2
import mediapipe as mp
from mediapipe.tasks import python as mpp
from mediapipe.tasks.python import vision
SP = r'C:\Users\trt\AppData\Local\Temp\claude\C--Dev-arrival-space-Browser-3DGS\7b704fd1-8971-4fe5-946e-5241da58e190\scratchpad'
arg = lambda k, d=None: sys.argv[sys.argv.index(k) + 1] if k in sys.argv else d
S = json.load(open(arg('--surface'))); V = np.array(S['vertices']); T = np.array(S['faces'], np.int64)
labels = S['boneLabels']; bi = np.array(S['boneIndices']); bw = np.array(S['boneWeights'])
wh = (bw * (bi == labels.index('head'))).sum(1); headT = T[(wh[T] > 0.3).all(1)]
rc = json.load(open(arg('--recon'))); cam = {c['imgIdx']: c for c in rc['cams']}[int(arg('--cam', '193'))]
R = np.array(cam['R']).reshape(3, 3); t = np.array(cam['t']); f, fy, cx, cy = cam['f'], cam.get('fy', cam['f']), cam['cx'], cam['cy']
W = 768; SS = 2                      # render at 2x and downsample: cleaner shading for the detector
X = (R @ V.T).T + t; z = X[:, 2]; u = (f * X[:, 0] / z + cx) * SS; v = (fy * X[:, 1] / z + cy) * SS
img = np.full((W * SS, W * SS, 3), 235, np.uint8); zbuf = np.full((W * SS, W * SS), np.inf); tid = np.full((W * SS, W * SS), -1, np.int64)
bary = np.zeros((W * SS, W * SS, 3))
n = np.zeros((len(T), 3))
for k, (a, b, c) in enumerate(headT):
    pa, pb, pc = X[a], X[b], X[c]; nrm = np.cross(pb - pa, pc - pa); nn = np.linalg.norm(nrm)
    if nn == 0: continue
    nrm /= nn
    if nrm[2] > 0: continue                                       # back-facing (camera looks down +z)
    light = np.array([0.3, -0.5, -0.8]); light /= np.linalg.norm(light)
    shade = 0.35 + 0.65 * max(0.0, float(nrm @ light))
    col = np.array([205, 175, 155]) * shade                       # skin-ish, lit from the camera side
    tri = np.array([[u[a], v[a]], [u[b], v[b]], [u[c], v[c]]])
    x0, y0 = np.floor(tri.min(0)).astype(int); x1, y1 = np.ceil(tri.max(0)).astype(int)
    x0, y0 = max(x0, 0), max(y0, 0); x1, y1 = min(x1, W * SS - 1), min(y1, W * SS - 1)
    if x1 < x0 or y1 < y0: continue
    ys, xs = np.mgrid[y0:y1 + 1, x0:x1 + 1]; px = np.stack([xs.ravel(), ys.ravel()], 1) + 0.5
    d = (tri[1, 0] - tri[0, 0]) * (tri[2, 1] - tri[0, 1]) - (tri[2, 0] - tri[0, 0]) * (tri[1, 1] - tri[0, 1])
    if abs(d) < 1e-9: continue
    l1 = ((tri[1, 0] - px[:, 0]) * (tri[2, 1] - px[:, 1]) - (tri[2, 0] - px[:, 0]) * (tri[1, 1] - px[:, 1])) / d
    l2 = ((tri[2, 0] - px[:, 0]) * (tri[0, 1] - px[:, 1]) - (tri[0, 0] - px[:, 0]) * (tri[2, 1] - px[:, 1])) / d
    l3 = 1 - l1 - l2; ins = (l1 >= 0) & (l2 >= 0) & (l3 >= 0)
    if not ins.any(): continue
    zz = l1[ins] * z[a] + l2[ins] * z[b] + l3[ins] * z[c]; yy, xx = ys.ravel()[ins], xs.ravel()[ins]
    better = zz < zbuf[yy, xx]
    zbuf[yy[better], xx[better]] = zz[better]; img[yy[better], xx[better]] = col; tid[yy[better], xx[better]] = k
    bary[yy[better], xx[better]] = np.stack([l1[ins][better], l2[ins][better], l3[ins][better]], 1)
small = cv2.resize(img, (W, W), interpolation=cv2.INTER_AREA)
cv2.imwrite('scratch/anny_head_render.png', small)
det = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
    base_options=mpp.BaseOptions(model_asset_path=os.path.join(SP, 'face_landmarker.task')),
    running_mode=vision.RunningMode.IMAGE, num_faces=1, min_face_detection_confidence=0.2, min_face_presence_confidence=0.2))
res = det.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))))
if not res.face_landmarks: print('NO FACE DETECTED on the render'); sys.exit(1)
L = res.face_landmarks[0]; corr = {}; miss = 0
for i, p in enumerate(L):
    if i >= 468: continue
    px, py = int(p.x * W * SS), int(p.y * W * SS)
    if not (0 <= px < W * SS and 0 <= py < W * SS) or tid[py, px] < 0:
        # search a small neighbourhood for the nearest rendered pixel
        found = False
        for r_ in range(1, 12):
            ys, xs = np.mgrid[max(py - r_, 0):min(py + r_ + 1, W * SS), max(px - r_, 0):min(px + r_ + 1, W * SS)]
            ok = tid[ys, xs] >= 0
            if ok.any():
                yy, xx = ys[ok][0], xs[ok][0]; py, px = yy, xx; found = True; break
        if not found: miss += 1; continue
    k = tid[py, px]; corr[i] = {'tri': [int(x) for x in headT[k]], 'bary': [float(x) for x in bary[py, px]]}
json.dump({'surface': arg('--surface'), 'cam': int(arg('--cam', '193')), 'corr': corr}, open('scratch/anny_face_corr.json', 'w'))
# draw for inspection
vis = small.copy()
for i, p in enumerate(L):
    if i < 468: cv2.circle(vis, (int(p.x * W), int(p.y * W)), 1, (0, 0, 255), -1)
cv2.imwrite('scratch/anny_head_landmarks.png', vis)
print(f'correspondences: {len(corr)}/468 (missed {miss}); render scratch/anny_head_render.png, landmarks scratch/anny_head_landmarks.png')
