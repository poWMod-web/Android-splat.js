#!/usr/bin/env node

/**
 * Start an official Bubblewrap Trusted Web Activity project from a deployed
 * SplatXR manifest. This intentionally stops after `bubblewrap init`: the
 * website URL, Android package ID, signing key, and Digital Asset Links
 * fingerprint belong to the deployment owner and must not be guessed here.
 *
 * Usage:
 *   node android/package-android.mjs https://example.com/splatxr/
 */

import { spawnSync } from 'node:child_process';

const rawDeploymentUrl = process.argv[2];
if (!rawDeploymentUrl || rawDeploymentUrl === '--help' || rawDeploymentUrl === '-h') {
  console.error('Usage: node android/package-android.mjs https://your-domain.example/splatxr/');
  process.exit(rawDeploymentUrl ? 0 : 2);
}

let deploymentUrl;
try {
  deploymentUrl = new URL(rawDeploymentUrl);
} catch {
  console.error(`Invalid deployment URL: ${rawDeploymentUrl}`);
  process.exit(2);
}

if (deploymentUrl.protocol !== 'https:') {
  console.error('Bubblewrap/TWA packaging requires the deployed site to use HTTPS.');
  process.exit(2);
}
if (deploymentUrl.search || deploymentUrl.hash) {
  console.error('Pass the deployment origin/path without a query string or hash.');
  process.exit(2);
}

// Accept either the deployed directory or its index.html, then resolve the
// manifest using the same relative path the application uses in the browser.
if (deploymentUrl.pathname.endsWith('/index.html')) {
  deploymentUrl.pathname = deploymentUrl.pathname.slice(0, -'index.html'.length);
} else if (!deploymentUrl.pathname.endsWith('/')) {
  deploymentUrl.pathname += '/';
}
const manifestUrl = new URL('manifest.webmanifest', deploymentUrl).href;
const npx = process.platform === 'win32' ? 'npx.cmd' : 'npx';
const result = spawnSync(npx, [
  '--yes',
  '@bubblewrap/cli',
  'init',
  `--manifest=${manifestUrl}`,
], { stdio: 'inherit' });

if (result.error) {
  console.error(`Could not start the official Bubblewrap CLI: ${result.error.message}`);
  process.exit(1);
}
if (result.status !== 0) process.exit(result.status ?? 1);

console.log('\nBubblewrap init completed from the deployed manifest.');
console.log('Review its generated project configuration, then run the official `bubblewrap build` command after choosing your package ID and signing key.');
console.log('Do not publish a TWA until /.well-known/assetlinks.json contains the final app package name and signing-certificate SHA-256 fingerprint.');
