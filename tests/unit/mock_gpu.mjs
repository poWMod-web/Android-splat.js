// Strict WebGPU mock for upload-size validation (not a test itself: run.mjs
// only executes test_*.mjs). Chromium validates every queue.writeBuffer
// against the source bytes synchronously ("Number of bytes to write is too
// large") and the destination buffer deeper (async device error) — this mock
// enforces both sides synchronously, with distinct messages, so oversized
// uploads fail deterministically in Node instead of only on one user's
// phone. An optional maxWriteBytes simulates drivers that reject very
// large single writes (mobile staging).

export function installWebGpuShims() {
  if (typeof globalThis.GPUBufferUsage === 'undefined') {
    globalThis.GPUBufferUsage = {
      MAP_READ: 1, MAP_WRITE: 2, COPY_SRC: 4, COPY_DST: 8,
      INDEX: 16, VERTEX: 32, UNIFORM: 64, STORAGE: 128,
      INDIRECT: 256, QUERY_RESOLVE: 512,
    };
  }
  if (typeof globalThis.GPUMapMode === 'undefined') {
    globalThis.GPUMapMode = { READ: 1, WRITE: 2 };
  }
  if (typeof globalThis.navigator === 'undefined') {
    globalThis.navigator = {};
  }
  if (!globalThis.navigator.gpu) {
    globalThis.navigator.gpu = { getPreferredCanvasFormat: () => 'bgra8unorm' };
  }
}

function byteLengthOf(data) {
  if (data instanceof ArrayBuffer) return data.byteLength;
  if (typeof SharedArrayBuffer !== 'undefined' && data instanceof SharedArrayBuffer) return data.byteLength;
  if (ArrayBuffer.isView(data)) return data.byteLength;
  throw new TypeError(`strictMockDevice: writeBuffer data must be a BufferSource, got ${typeof data}`);
}

/** A GPUDevice whose queue enforces the spec's writeBuffer bounds rules.
 *  opts: { bindingLimit (default 128MB, phone-like), maxBufferSize (256MB),
 *  maxWriteBytes (default Infinity), onWrite }. */
export function strictMockDevice(opts = {}) {
  const bindingLimit = opts.bindingLimit ?? 128 * 1024 * 1024;
  const maxBufferSize = opts.maxBufferSize ?? 256 * 1024 * 1024;
  const maxWriteBytes = opts.maxWriteBytes ?? Infinity;
  const onWrite = opts.onWrite || (() => {});
  const buffers = [];
  const writes = [];
  // mapFillers[label](arrayBuffer): stage readback contents (refine tests
  // shape the gather census through the 'refine-gatherRead' buffer)
  const mapFillers = {};

  // Chromium throws the "too large" OperationError synchronously for the
  // SOURCE side only (the size/count exceeds the source bytes — see Blink's
  // GPUQueue::WriteBufferImpl); a DESTINATION overrun fails deeper (Dawn
  // validation, async device error in the browser). The mock throws for
  // both — synchronously, so tests stay deterministic — but with distinct
  // messages so a test pins which side overflowed.
  const failSource = (label, have, want) =>
    new Error(`Failed to execute 'writeBuffer' on 'GPUQueue': Number of bytes to write is too large ` +
      `(${label}: ${want}B wanted, source holds ${have}B)`);
  const failDest = (label, size, bufferSize, bufferOffset) =>
    new Error(`Failed to execute 'writeBuffer' on 'GPUQueue': write of ${size}B @${bufferOffset} ` +
      `exceeds '${label}' (${bufferSize}B)`);

  const checkCopy = (label, src, srcOffset, dst, dstOffset, size, op) => {
    if (!Number.isInteger(size) || size < 0) {
      throw new Error(`strictMockDevice: ${op} '${label}' invalid size ${size}`);
    }
    if (srcOffset + size > src.size) {
      throw new Error(`strictMockDevice: ${op} '${label}' overruns source ` +
        `(${srcOffset} + ${size} > ${src.size})`);
    }
    if (dstOffset + size > dst.size) {
      throw new Error(`strictMockDevice: ${op} '${label}' overruns destination ` +
        `(${dstOffset} + ${size} > ${dst.size})`);
    }
  };

  const queue = {
    writes,
    writeBuffer(buffer, bufferOffset, data, dataOffset = 0, size = undefined) {
      if (!buffer || typeof buffer.size !== 'number') throw new TypeError('strictMockDevice: writeBuffer needs a mock buffer');
      if (!Number.isInteger(bufferOffset) || bufferOffset < 0 || bufferOffset % 4 !== 0) {
        throw new Error(`strictMockDevice: writeBuffer '${buffer.label}' misaligned bufferOffset ${bufferOffset}`);
      }
      // Blink takes dataOffset/size in ELEMENTS for TypedArray views, in
      // bytes for raw ArrayBuffers (GPUQueue::writeBuffer overloads) — the
      // mock mirrors that, or it would miss element-count overruns.
      const elSize = (ArrayBuffer.isView(data) && data.BYTES_PER_ELEMENT) || 1;
      const dataLen = byteLengthOf(data);
      if (!Number.isInteger(dataOffset) || dataOffset < 0) {
        throw new Error(`strictMockDevice: writeBuffer '${buffer.label}' invalid dataOffset ${dataOffset}`);
      }
      const dataOffBytes = dataOffset * elSize;
      const n = size === undefined ? dataLen - dataOffBytes : size * elSize;
      if (!Number.isInteger(n) || n < 0) {
        throw new Error(`strictMockDevice: writeBuffer '${buffer.label}' invalid size ${size}`);
      }
      if (dataOffBytes + n > dataLen) throw failSource(buffer.label, dataLen - dataOffBytes, n);
      if (bufferOffset + n > buffer.size) throw failDest(buffer.label, n, buffer.size, bufferOffset);
      if (n > maxWriteBytes) {
        throw new Error(`Failed to execute 'writeBuffer' on 'GPUQueue': single write of ${n}B ` +
          `exceeds this driver's ${maxWriteBytes}B ceiling ('${buffer.label}')`);
      }
      writes.push({ label: buffer.label, buffer, bufferOffset, size: n });
      onWrite(buffer.label, n);
    },
    submits: 0,
    submitted: [],
    submit(bufs) { this.submits++; this.submitted.push(...(bufs || [])); },
    onSubmittedWorkDone() { return Promise.resolve(); },
  };

  const device = {
    limits: {
      maxStorageBufferBindingSize: bindingLimit,
      maxBufferSize,
      maxComputeWorkgroupSizeX: 256,
      timestampPeriod: 1, // 1ns ticks, like the profileSteps assumption
    },
    features: new Set(opts.features ?? ['timestamp-query']),
    passes: [], // beginComputePass records { ts, enc, bgs, pipes }
    queue,
    buffers,
    mapFillers,
    createBuffer({ size, usage, label, mappedAtCreation }) {
      if (!Number.isInteger(size) || size < 0) {
        throw new Error(`Failed to execute 'createBuffer' on 'GPUDevice': invalid size ${size} ('${label}')`);
      }
      if (size > maxBufferSize) {
        throw new Error(`Failed to execute 'createBuffer' on 'GPUDevice': size ${size} exceeds maxBufferSize ${maxBufferSize} ('${label}')`);
      }
      const buf = {
        size, usage, label: label || 'unnamed',
        _map: mappedAtCreation ? new ArrayBuffer(size) : null,
        destroy() { this._destroyed = true; },
        async mapAsync() {
          this._map = new ArrayBuffer(this.size);
          const fill = mapFillers[this.label];
          if (fill) fill(this._map);
          if (this._resolvedFrom) {
            const { qs, first, count } = this._resolvedFrom;
            new BigUint64Array(this._map).set(qs._data.subarray(first, first + count));
          }
        },
        getMappedRange() {
          if (!this._map) throw new Error(`strictMockDevice: '${this.label}' getMappedRange before mapAsync`);
          return this._map;
        },
        unmap() { this._map = null; },
      };
      buffers.push(buf);
      return buf;
    },
    createShaderModule() { return {}; },
    createComputePipeline() { return { getBindGroupLayout: () => ({}) }; },
    createRenderPipeline() { return { getBindGroupLayout: () => ({}) }; },
    createBindGroup(desc) { return { entries: desc ? desc.entries : [] }; },
    createTexture() { return { createView: () => ({}), destroy() {} }; },
    createSampler() { return {}; },
    createQuerySet({ count }) {
      return { count, _data: new BigUint64Array(count), destroy() {} };
    },
    createCommandEncoder() {
      const enc = {
        copyBufferToBuffer(src, srcOffset, dst, dstOffset, size) {
          checkCopy(dst.label || src.label, src, srcOffset, dst, dstOffset, size, 'copyBufferToBuffer');
          if (src._resolvedFrom) {
            if (dstOffset !== 0) throw new Error('strictMockDevice: timestamp copies must land at 0');
            dst._resolvedFrom = {
              qs: src._resolvedFrom.qs,
              first: src._resolvedFrom.first + srcOffset / 8,
              count: size / 8,
            };
          }
        },
        resolveQuerySet(qs, first, count, dest, destOffset) {
          if (first + count > qs.count) {
            throw new Error(`strictMockDevice: resolve [${first}, ${first + count}) exceeds query set (${qs.count})`);
          }
          if (destOffset + count * 8 > dest.size) {
            throw new Error(`strictMockDevice: resolve overruns '${dest.label}'`);
          }
          dest._resolvedFrom = { qs, first, count };
        },
        beginComputePass(desc) {
          const rec = { ts: desc?.timestampWrites ?? null, enc, bgs: [], pipes: [] };
          device.passes.push(rec);
          return {
            setPipeline(pipe) { rec.pipes.push(pipe); },
            setBindGroup(i, bg) { rec.bgs.push(bg); },
            dispatchWorkgroups() {}, dispatchWorkgroupsIndirect() {}, end() {},
          };
        },
        finish() { return { enc }; },
      };
      return enc;
    },
    destroy() {},
  };
  return device;
}

/** A createGpu-shaped context around the strict device (hand it in as
 *  opts.device / opts.gpu wherever the session or GSTrainer.create runs). */
export function strictMockGpu(deviceOpts = {}) {
  const device = strictMockDevice(deviceOpts);
  return {
    device,
    adapter: null,
    info: { vendor: 'mock', architecture: 'mock-phone-gpu', device: 'strict-mock' },
    owned: false,
    onLost: null,
    dispose() {},
  };
}
