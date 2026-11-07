// pcview.js — the SOG-native result viewer: the PlayCanvas engine renders
// the compressed splat directly (sorted alpha compositing, WebGL2), so a
// shared creation never gets decoded into hundreds of MB of floats and
// phones without WebGPU can still view it. The object returned here wears
// the same face the app's render loop expects from a training session:
// view.attach / setCamera / renderNow / lookThrough, trainer.camMeta,
// model.radius, frames, dispose — a stand-in, not a trainer.

let pcPromise = null;
const loadPc = () => (pcPromise ??= import('../vendor/playcanvas.min.mjs'));

export async function createSogView(sogUrl, { radius = 10 } = {}) {
  const pc = await loadPc();

  const v = {
    pc,
    app: null,
    camEnt: null,
    entity: null,
    canvas: null,
    model: { radius },
    trainer: { camMeta: [] },   // restore path fills this in
    frames: [],                 // and this
    training: false,
    sogUrl,
  };

  const ensureApp = async (canvas) => {
    if (v.app) return;
    v.canvas = canvas;
    const app = new pc.Application(canvas, {
      graphicsDeviceOptions: { antialias: false },
    });
    app.setCanvasFillMode(pc.FILLMODE_NONE);
    app.setCanvasResolution(pc.RESOLUTION_FIXED, canvas.width, canvas.height);
    // SH colors refresh only after the camera translates by tan(angle) x
    // splat distance — at the 10-degree engine default the view-dependent
    // color visibly SNAPS mid-move. 0 = refresh on any camera movement
    // (a static camera still costs nothing: the engine gates on
    // translationDelta > 0, and pure rotation cannot change SH).
    // ?colang=N overrides for perf testing on weak devices.
    if (app.scene.gsplat) {
      const q = new URLSearchParams(location.search).get('colang');
      app.scene.gsplat.colorUpdateAngle = q === null ? 0 : +q;
    }
    v.app = app;

    const cam = new pc.Entity('cam');
    cam.addComponent('camera', {
      clearColor: new pc.Color(0, 0, 0, 1),
      fov: 50,
      nearClip: Math.max(0.001, radius * 0.005),
      farClip: radius * 100,
    });
    app.root.addChild(cam);
    v.camEnt = cam;
    if (v._lastCam) setCam(v._lastCam);

    // the splat, straight from the SOG bundle — the frame loop starts only
    // once it is in, so the stand-in hero image stays visible while the
    // model streams
    const asset = new pc.Asset('shared.sog', 'gsplat', { url: sogUrl, filename: 'shared.sog' });
    await new Promise((resolve, reject) => {
      asset.on('load', resolve);
      asset.on('error', (err) => reject(new Error(`sog load failed: ${err}`)));
      app.assets.add(asset);
      app.assets.load(asset);
    });
    v.entity = new pc.Entity('splat');
    v.entity.addComponent('gsplat', { asset });
    // the SfM world is OpenCV y-down; PlayCanvas is y-up
    v.entity.setEulerAngles(180, 0, 0);
    app.root.addChild(v.entity);
    app.start();
  };

  // OpenCV-convention camera -> PC entity pose. R rows are the camera axes
  // in (y-down) world; the 180deg-X entity flip means world_pc = (x,-y,-z).
  const q = () => new pc.Quat();
  const m = () => new pc.Mat4();
  const setCam = (c) => {
    if (!v.camEnt) return;
    const { R, t } = c;
    const C = [
      -(R[0] * t[0] + R[3] * t[1] + R[6] * t[2]),
      -(R[1] * t[0] + R[4] * t[1] + R[7] * t[2]),
      -(R[2] * t[0] + R[5] * t[1] + R[8] * t[2])];
    const px = C[0], py = -C[1], pz = -C[2];
    // columns: right, up, back (in pc world)
    const right = [R[0], -R[1], -R[2]];
    const up = [-R[3], R[4], R[5]];
    const back = [-R[6], R[7], R[8]];
    const mat = m();
    mat.set([
      right[0], right[1], right[2], 0,
      up[0], up[1], up[2], 0,
      back[0], back[1], back[2], 0,
      px, py, pz, 1,
    ]);
    const rot = q().setFromMat4(mat);
    v.camEnt.setPosition(px, py, pz);
    v.camEnt.setRotation(rot);
    const fov = 2 * Math.atan(0.5 * c.h / c.f) * 180 / Math.PI;
    v.camEnt.camera.fov = fov;
    v.camEnt.camera.horizontalFov = false;
  };

  v.view = {
    attach(canvas) {
      if (!v.app) {
        // fire-and-forget: the loop keeps calling setCamera; frames start
        // once the asset is in
        v._boot = ensureApp(canvas).catch((e) => { v._bootError = e; console.error(e); });
      } else if (canvas === v.canvas) {
        v.app.setCanvasResolution(pc.RESOLUTION_FIXED, canvas.width, canvas.height);
      }
    },
    setCamera(c) {
      v._lastCam = c;
      if (v.canvas && v.app && (v.canvas.width !== v._rw || v.canvas.height !== v._rh)) {
        v._rw = v.canvas.width; v._rh = v.canvas.height;
        v.app.setCanvasResolution(pc.RESOLUTION_FIXED, v._rw, v._rh);
      }
      setCam(c);
    },
    renderNow() { /* the engine renders on its own frame loop */ },
    lookThrough(ci) {
      const meta = v.trainer.camMeta[ci];
      if (meta) v.view.setCamera(meta);
      return meta;
    },
  };

  /** Standard 3DGS PLY, decoded from the SOG on demand (download action). */
  v.exportPlyBlob = async () => {
    const { sogToGaussians } = await import('./session_io.js');
    const { gaussiansToPly } = await import('../../src/index.js');
    const bytes = new Uint8Array(await (await fetch(sogUrl)).arrayBuffer());
    const g = await sogToGaussians(bytes);
    return gaussiansToPly(g.data, g.n, g.sh, g.shK);
  };

  v.dispose = () => {
    try { v.app && v.app.destroy(); } catch (e) { /* torn down with the page */ }
    v.app = null;
  };

  return v;
}
