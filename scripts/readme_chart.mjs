// node scripts/readme_chart.mjs [statusA.json statusB.json ...]
//   writes docs/img/truck-psnr-vs-time-{light,dark}.svg, docs/img/truck-curve.json
//   and prints the README evolution table (minutes · cycles · held-out dB).
// Truck held-out PSNR as a training run unfolds: the mean of the bench cells'
// `curve` arrays (eval8 every `evalmin` minutes, ?evalmin=2) on a common
// 2-minute grid, the two locally measured trainers as marks at their measured
// times, the published methods as dashed levels (their times are not
// comparable). Without arguments the committed docs/img/truck-curve.json is drawn.
import { readFileSync, writeFileSync, existsSync } from 'node:fs';

const CURVE_JSON = 'docs/img/truck-curve.json';
const files = process.argv.slice(2);
let curve;   // [{ min, iter, psnr }]
let meta;
if (files.length) {
  const cells = files.map((f) => JSON.parse(readFileSync(f, 'utf8')));
  const step = 2;
  const end = Math.min(...cells.map((c) => c.trainMin));
  const lerp = (arr, key, m) => {   // linear in time inside a cell's checkpoints
    let a = arr[0], b = arr[arr.length - 1];
    for (let i = 1; i < arr.length; i++) if (arr[i].min >= m) { a = arr[i - 1]; b = arr[i]; break; }
    if (m <= arr[0].min) return arr[0][key];
    if (m >= arr[arr.length - 1].min) return arr[arr.length - 1][key];
    const t = (m - a.min) / (b.min - a.min);
    return a[key] + t * (b[key] - a[key]);
  };
  curve = [];
  for (let m = step; m <= end + 1e-9; m += step) {
    const ps = cells.map((c) => lerp(c.curve, 'psnr', m));
    const it = cells.map((c) => lerp(c.curve, 'iter', m));
    curve.push({ min: m, iter: Math.round(it.reduce((a, b) => a + b) / it.length),
      psnr: +(ps.reduce((a, b) => a + b) / ps.length).toFixed(3), seeds: ps.map((p) => +p.toFixed(3)) });
  }
  const finals = cells.map((c) => c.psnrTest);
  // the last point is the final held-out score of each cell (scored at the stop)
  curve.push({ min: +end.toFixed(1), iter: Math.round(cells.reduce((a, c) => a + c.iterDone, 0) / cells.length),
    psnr: +(finals.reduce((a, b) => a + b) / finals.length).toFixed(3), seeds: finals, final: true });
  meta = { date: new Date().toISOString().slice(0, 10), cells: files.map((f) => f.split('/').pop()),
    finals, iterDone: cells.map((c) => c.iterDone), trainMin: cells.map((c) => c.trainMin), splats: cells[0].splats };
  writeFileSync(CURVE_JSON, JSON.stringify({ meta, curve }, null, 1));
} else {
  if (!existsSync(CURVE_JSON)) { console.error('no status files given and no', CURVE_JSON); process.exit(1); }
  ({ meta, curve } = JSON.parse(readFileSync(CURVE_JSON, 'utf8')));
}
const last = curve[curve.length - 1];

const LOCAL = [   // measured on the same machine, same images, same holdout
  { min: 5.5, psnr: 26.14, name: 'LichtFeld Studio', sub: '5½ min · 26.14', dx: -50, anchor: 'start', at: 26.72 },   // label parked above the levels, leader down to the mark
  { min: 30, psnr: 26.10, name: 'Brush v0.3', sub: '30 min · 26.10', dx: -10, anchor: 'end', dy: 17 },
];
const PAPERS = [  // native CUDA, 30 k iterations; time not comparable, drawn as levels
  { psnr: 26.41, name: 'Student Splatting & Scooping', off: -4 },
  { psnr: 26.11, name: '3DGS-MCMC', off: 4 },
  { psnr: 25.77, name: 'Scaffold-GS', off: -3 },
  { psnr: 25.74, name: 'Mip-Splatting', off: 10 },
  { psnr: 25.18, name: '3DGS (2023)', off: 4 },
];

const W = 820, H = 420, L = 70, R = 590, T = 44, B = 356;
const X0 = 0, X1 = Math.max(32, Math.ceil(last.min / 5) * 5 + 2), Y0 = 23.0, Y1 = 27.0;
const x = (m) => L + (m - X0) / (X1 - X0) * (R - L);
const y = (p) => B - (p - Y0) / (Y1 - Y0) * (B - T);
const shown = curve.filter((d) => d.psnr >= Y0);   // the first minutes sit below the axis
const f2 = (v) => v.toFixed(2);
const esc = (t) => t.replace(/&/g, '&amp;');

function svg(theme) {
  const c = theme === 'dark'
    ? { text: '#c9d1d9', muted: '#8b949e', grid: '#30363d', axis: '#484f58', paper: '#8b949e', splat: '#2dd4bf', splatFill: '#0d1117', mark: '#d2a8ff' }
    : { text: '#1f2328', muted: '#59636e', grid: '#d8dee4', axis: '#8c959f', paper: '#6e7781', splat: '#0f9c8c', splatFill: '#ffffff', mark: '#8250df' };
  const font = `font-family="ui-sans-serif, -apple-system, 'Segoe UI', Helvetica, Arial, sans-serif"`;
  let s = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${W} ${H}" width="${W}" height="${H}" ${font} font-size="12">\n`;
  s += `<title>Truck: held-out PSNR as training proceeds</title>\n`;
  for (let p = Y0; p <= Y1 + 1e-9; p += 0.5) {
    s += `<line x1="${L}" x2="${R}" y1="${y(p).toFixed(1)}" y2="${y(p).toFixed(1)}" stroke="${c.grid}" stroke-width="1"/>\n`;
    s += `<text x="${L - 10}" y="${(y(p) + 4).toFixed(1)}" text-anchor="end" fill="${c.muted}" font-variant-numeric="tabular-nums">${p.toFixed(1)}</text>\n`;
  }
  for (let m = 0; m <= X1 - 2; m += 5) {
    s += `<line x1="${x(m).toFixed(1)}" x2="${x(m).toFixed(1)}" y1="${B}" y2="${B + 5}" stroke="${c.axis}"/>\n`;
    s += `<text x="${x(m).toFixed(1)}" y="${B + 20}" text-anchor="middle" fill="${c.muted}">${m}</text>\n`;
  }
  s += `<line x1="${L}" x2="${R}" y1="${B}" y2="${B}" stroke="${c.axis}"/>\n`;
  s += `<text x="${(L + R) / 2}" y="${B + 42}" text-anchor="middle" fill="${c.muted}">training minutes on one RTX 5080 (camera solve not included)</text>\n`;
  s += `<text transform="translate(22 ${(T + B) / 2}) rotate(-90)" text-anchor="middle" fill="${c.muted}">test PSNR, dB (32 held-out photos)</text>\n`;
  for (const p of PAPERS) {
    const yy = y(p.psnr).toFixed(1);
    s += `<line x1="${L}" x2="${R}" y1="${yy}" y2="${yy}" stroke="${c.paper}" stroke-dasharray="4 4" stroke-width="1"/>\n`;
    s += `<text x="${R + 10}" y="${(y(p.psnr) + 4 + p.off).toFixed(1)}" fill="${c.paper}">${esc(p.name)}</text>\n`;
  }
  // Splat.js curve: mean of the seeds, seed spread as a faint band
  if (shown[0].seeds && shown[0].seeds.length > 1) {
    const hi = shown.map((d) => `${x(d.min).toFixed(1)},${y(Math.min(Y1, Math.max(...d.seeds))).toFixed(1)}`);
    const lo = [...shown].reverse().map((d) => `${x(d.min).toFixed(1)},${y(Math.max(Y0, Math.min(...d.seeds))).toFixed(1)}`);
    s += `<polygon points="${[...hi, ...lo].join(' ')}" fill="${c.splat}" opacity="0.18"/>\n`;
  }
  const pts = shown.map((d) => `${x(d.min).toFixed(1)},${y(d.psnr).toFixed(1)}`).join(' ');
  s += `<polyline points="${pts}" fill="none" stroke="${c.splat}" stroke-width="2.5" stroke-linejoin="round"/>\n`;
  for (const d of shown) s += `<circle cx="${x(d.min).toFixed(1)}" cy="${y(d.psnr).toFixed(1)}" r="${d.final ? 5 : 2.5}" fill="${d.final ? c.splatFill : c.splat}" stroke="${c.splat}" stroke-width="2.5"/>\n`;
  // end label above the last point, right-aligned so it stays inside the plot
  s += `<text x="${(x(last.min) + 6).toFixed(1)}" y="${(y(last.psnr) - 14).toFixed(1)}" text-anchor="end" fill="${c.splat}" font-weight="700" font-size="13">Splat.js</text>\n`;
  for (const m of LOCAL) {
    const px = x(m.min), py = y(m.psnr);
    s += `<path d="M ${px} ${py - 6} L ${px + 6} ${py} L ${px} ${py + 6} L ${px - 6} ${py} Z" fill="${c.mark}"/>\n`;
    const ly = m.at ? y(m.at) : py + m.dy;
    if (m.at) s += `<line x1="${px}" x2="${px}" y1="${(py - 7).toFixed(1)}" y2="${(ly + 3).toFixed(1)}" stroke="${c.mark}" stroke-width="1" stroke-dasharray="2 3"/>
`;
    s += `<text x="${(px + m.dx).toFixed(1)}" y="${ly.toFixed(1)}" text-anchor="${m.anchor}" fill="${c.mark}" font-size="11"><tspan font-weight="600">${m.name}</tspan></text>\n`;
  }
  s += `<text x="${L}" y="22" fill="${c.text}" font-size="15" font-weight="700">Tanks &amp; Temples Truck — held-out PSNR as training proceeds</text>\n`;
  s += `</svg>\n`;
  return s;
}
for (const t of ['light', 'dark']) writeFileSync(`docs/img/truck-psnr-vs-time-${t}.svg`, svg(t));

// README table: every 4 minutes plus the end
const rows = curve.filter((d) => d.final || Math.round(d.min) % 4 === 0);
let md = '| minutes | cycles | Truck test PSNR |\n|---:|---:|---:|\n';
for (const d of rows) md += `| ${d.final ? Math.round(d.min) : d.min} | ${(Math.round(d.iter / 1000))} k | ${f2(d.psnr)} dB |\n`;
console.log(md);
console.log(`written docs/img/truck-psnr-vs-time-{light,dark}.svg (${curve.length} points, final ${last.psnr} at ${last.min} min, seeds ${JSON.stringify(meta.finals)})`);
