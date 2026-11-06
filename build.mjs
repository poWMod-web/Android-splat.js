import { cp, mkdir, rm } from 'node:fs/promises';

await rm('dist', { recursive: true, force: true });
await mkdir('dist', { recursive: true });
await cp('app', 'dist', { recursive: true });
await cp('src', 'dist/src', { recursive: true });
await cp('data', 'dist/data', { recursive: true });
await cp('LICENSE', 'dist/LICENSE');
console.log('Built SplatXR: static JavaScript app, WebXR capture, and WebGPU splat.js engine.');
