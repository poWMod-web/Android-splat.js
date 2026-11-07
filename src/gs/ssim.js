// Structural similarity (SSIM) over normalized RGB pixels. The local statistics
// follow the common 11x11, sigma=1.5 Gaussian-window definition used by OpenCV.
const RADIUS = 5;
const SIGMA = 1.5;
const C1 = 0.01 ** 2;
const C2 = 0.03 ** 2;
const KERNEL = Float64Array.from({ length: RADIUS * 2 + 1 }, (_, i) => {
  const x = i - RADIUS;
  return Math.exp(-(x * x) / (2 * SIGMA * SIGMA));
});
{
  const sum = KERNEL.reduce((a, b) => a + b, 0);
  for (let i = 0; i < KERNEL.length; i++) KERNEL[i] /= sum;
}

function reflect101(i, size) {
  if (size <= 1) return 0;
  while (i < 0 || i >= size) i = i < 0 ? -i : size * 2 - 2 - i;
  return i;
}

function gaussianBlur(src, width, height) {
  const temp = new Float32Array(src.length);
  const out = new Float32Array(src.length);
  for (let y = 0; y < height; y++) {
    const row = y * width;
    for (let x = 0; x < width; x++) {
      let sum = 0;
      for (let k = -RADIUS; k <= RADIUS; k++) {
        sum += KERNEL[k + RADIUS] * src[row + reflect101(x + k, width)];
      }
      temp[row + x] = sum;
    }
  }
  for (let y = 0; y < height; y++) {
    const dst = y * width;
    for (let x = 0; x < width; x++) {
      let sum = 0;
      for (let k = -RADIUS; k <= RADIUS; k++) {
        sum += KERNEL[k + RADIUS] * temp[reflect101(y + k, height) * width + x];
      }
      out[dst + x] = sum;
    }
  }
  return out;
}

/**
 * Compute mean RGB SSIM for interleaved, linear-range [0,1] image data.
 * Invalid pixels may be excluded from the final mean with `valid` (one byte
 * per pixel); they still contribute zero to neighboring Gaussian windows.
 */
export function ssimIndexRgb(imageA, imageB, width, height, valid = null) {
  const pixelCount = width * height;
  if (!Number.isInteger(width) || !Number.isInteger(height) || width < 1 || height < 1 ||
      imageA.length !== pixelCount * 3 || imageB.length !== pixelCount * 3 ||
      (valid && valid.length !== pixelCount)) {
    throw new RangeError('SSIM inputs must be equal-sized interleaved RGB images');
  }
  let score = 0, samples = 0;
  for (let channel = 0; channel < 3; channel++) {
    const a = new Float32Array(pixelCount);
    const b = new Float32Array(pixelCount);
    const a2 = new Float32Array(pixelCount);
    const b2 = new Float32Array(pixelCount);
    const ab = new Float32Array(pixelCount);
    for (let p = 0, j = channel; p < pixelCount; p++, j += 3) {
      const x = imageA[j], y = imageB[j];
      a[p] = x; b[p] = y; a2[p] = x * x; b2[p] = y * y; ab[p] = x * y;
    }
    const muA = gaussianBlur(a, width, height);
    const muB = gaussianBlur(b, width, height);
    const ex2 = gaussianBlur(a2, width, height);
    const ey2 = gaussianBlur(b2, width, height);
    const exy = gaussianBlur(ab, width, height);
    for (let p = 0; p < pixelCount; p++) {
      if (valid && !valid[p]) continue;
      const mx = muA[p], my = muB[p];
      const varA = Math.max(0, ex2[p] - mx * mx);
      const varB = Math.max(0, ey2[p] - my * my);
      const cov = exy[p] - mx * my;
      const numerator = (2 * mx * my + C1) * (2 * cov + C2);
      const denominator = (mx * mx + my * my + C1) * (varA + varB + C2);
      score += denominator > 0 ? numerator / denominator : 1;
      samples++;
    }
  }
  return samples ? score / samples : null;
}
