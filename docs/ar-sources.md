# WebXR implementation references

The fork uses the existing known-camera API from arrival-space/splat.js commit `a84620c`, documented in its README and `src/io/cameras.js`. Images and cameras are matched by basename. Camera intrinsics are converted from image pixels to feature-scale pixels by `useKnownCameras()` before training.

WebXR immersive AR alone does not provide camera pixels. Pose-synchronized photo capture requires the separately granted `camera-access` descriptor and `XRView.camera`, read with `XRWebGLBinding.getCameraImage()` inside the same active XR animation frame as the view pose. The browser owns the camera texture; it must be treated as read-only. Its camera image must have the same viewing frustum and pose as the XRView. Raw Camera Access additionally requires both camera and xr-spatial-tracking permissions policies, a secure origin, and device/browser support. Source: https://immersive-web.github.io/raw-camera-access/

Depth and hit-test are optional measurement sources; ARCore's native sparse point cloud is not generically exposed through WebXR. Do not claim access to a native ARCore feature-point cloud. Without measured seed points, splat.js may explicitly initialize random seed geometry in the viewed volume; that is not a captured point cloud.

The WebXR depth feature requires a `depthSensing` session dictionary even when optional. It contains ordered `usagePreference` and `dataFormatPreference` arrays; top-level `depthUsage` and `depthDataFormat` are session properties, not session initialization options. SplatXR requests CPU usage with matching depth view to use same-frame camera-plane depth measurements. Source: https://www.w3.org/TR/webxr-depth-sensing-1/

The reference UI and existing JavaScript training, preview and export workflow were obtained from the user's requested https://arrival.space/splat-js/index.html and MIT-licensed https://github.com/arrival-space/splat.js repository.
