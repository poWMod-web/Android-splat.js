// Distance only: no wall clock, XR timestamp or camera orientation affects capture cadence.
export const DEFAULT_CAPTURE_SPACING_M = 0.10;

export class DistanceCaptureTrigger {
  constructor(spacingMeters = DEFAULT_CAPTURE_SPACING_M, noiseFloorMeters = 0.002) {
    if (!Number.isFinite(spacingMeters) || spacingMeters <= 0) throw new RangeError('Capture spacing must be positive meters.');
    this.spacingMeters = spacingMeters;
    this.noiseFloorMeters = noiseFloorMeters;
    this.reset();
  }

  reset() {
    this.lastPosition = null;
    this.distanceSinceCapture = 0;
    this.totalDistance = 0;
  }

  setSpacingMeters(spacingMeters) {
    if (!Number.isFinite(spacingMeters) || spacingMeters <= 0) throw new RangeError('Capture spacing must be positive meters.');
    this.spacingMeters = spacingMeters;
    this.reset();
  }

  breakContinuity() {
    // Missing tracking is not measurable travel; the first recovered pose is a new baseline.
    this.lastPosition = null;
  }

  sample(position) {
    if (!Array.isArray(position) || position.length !== 3 || !position.every(Number.isFinite)) {
      this.breakContinuity(); return false;
    }
    if (!this.lastPosition) { this.lastPosition = position.slice(); return false; }
    const distance = Math.hypot(...position.map((value, i) => value - this.lastPosition[i]));
    if (distance > this.noiseFloorMeters) {
      this.distanceSinceCapture += distance;
      this.totalDistance += distance;
      this.lastPosition = position.slice();
    }
    return this.distanceSinceCapture + 1e-9 >= this.spacingMeters;
  }

  captured(position) {
    // Reset to the successful photo's position, including manually triggered photos.
    this.lastPosition = position.slice();
    this.distanceSinceCapture = 0;
  }
}
