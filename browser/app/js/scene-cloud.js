// Build the sparse point cloud displayed with camera poses in Details.
// Saved trainer reconstructions intentionally omit live `recon.points` on
// restore; their frozen display cloud lives in recon.json's `cloud` field.
export function buildDisplayCloud(capturedPoints = [], savedCloud = null, reconPoints = []) {
  const points = Array.isArray(capturedPoints) && capturedPoints.length ? capturedPoints : [];
  if (points.length) return cloudFromPoints(points);

  // Prefer the immutable, captured-time snapshot over solver points that may
  // have been mutated during training (or omitted entirely on restore).
  const cloudXYZ = savedCloud && savedCloud.xyz;
  if (cloudXYZ && cloudXYZ.length) {
    const xyz = Float32Array.from(cloudXYZ);
    const rgb = new Uint8Array(xyz.length);
    const savedRGB = savedCloud.rgb || [];
    for (let i = 0; i < Math.min(rgb.length, savedRGB.length); i++) rgb[i] = savedRGB[i];
    return { xyz, rgb };
  }

  return cloudFromPoints(Array.isArray(reconPoints) ? reconPoints : []);
}

function cloudFromPoints(points) {
  const xyz = new Float32Array(points.length * 3);
  const rgb = new Uint8Array(points.length * 3);
  points.forEach((point, index) => {
    if (point.X) xyz.set(point.X, index * 3);
    if (point.rgb) {
      rgb[index * 3] = point.rgb[0] * 255;
      rgb[index * 3 + 1] = point.rgb[1] * 255;
      rgb[index * 3 + 2] = point.rgb[2] * 255;
    }
  });
  return { xyz, rgb };
}
