/**
 * dither.js — colour helpers for the Dither-CA domain (§11).
 *
 * The Dither-CA reinterprets the PID-CA in colour space: every expressible
 * state is a palette colour, the per-cell target is a colour T(c) drawn from
 * a solid fill, a gradient or an image, and each cell regulates the *mean
 * colour of its neighbourhood* toward T(c) with a three-channel PID.
 *
 * This module holds only the pure colour / geometry helpers: palette parsing
 * and presets, gradient + image sampling into a cell-resolution RGB field,
 * nearest-colour lookup and a small k-means quantiser used by "palette from
 * target". It knows nothing about the grid or the simulation loop and must
 * not import config.js (config.js imports *it*).
 */

export const MAX_PALETTE = 32;
export const DEFAULT_PALETTE = '#000000,#ffffff';

/** Ready-made palettes for the palette editor (§11). `colors` is a hex list. */
export const PALETTE_PRESETS = [
  { name: 'Black & white', colors: '#000000,#ffffff' },
  { name: 'Greys × 4', colors: '#000000,#555555,#aaaaaa,#ffffff' },
  { name: 'Greys × 8', colors: '#000000,#242424,#494949,#6d6d6d,#929292,#b6b6b6,#dbdbdb,#ffffff' },
  { name: 'Game Boy', colors: '#0f380f,#306230,#8bac0f,#9bbc0f' },
  {
    name: 'CGA (16)',
    colors:
      '#000000,#0000aa,#00aa00,#00aaaa,#aa0000,#aa00aa,#aa5500,#aaaaaa,' +
      '#555555,#5555ff,#55ff55,#55ffff,#ff5555,#ff55ff,#ffff55,#ffffff',
  },
  {
    name: 'PICO-8 (16)',
    colors:
      '#000000,#1d2b53,#7e2553,#008751,#ab5236,#5f574f,#c2c3c7,#fff1e8,' +
      '#ff004d,#ffa300,#ffec27,#00e436,#29adff,#83769c,#ff77a8,#ffccaa',
  },
  { name: 'Risograph (paper + 3 inks)', colors: '#f4efe6,#ff6f61,#0078bf,#ffe800' },
  { name: 'Newsprint (CMYK-ish)', colors: '#ffffff,#00aeef,#ec008c,#fff200,#231f20' },
  { name: 'Sepia × 5', colors: '#2b1d0e,#5c3d1e,#8b6b3d,#c4a774,#f3e6c8' },
  { name: 'Neon on black', colors: '#000000,#ff2079,#04d9ff,#ccff00' },
];

/** Coerce `#rgb` / `#rrggbb` / `rrggbb` to canonical `#rrggbb`, else null. */
export function normalizeHex(value) {
  if (typeof value !== 'string') return null;
  let s = value.trim().replace(/^#/, '');
  if (s.length === 3) s = s[0] + s[0] + s[1] + s[1] + s[2] + s[2];
  if (!/^[0-9a-fA-F]{6}$/.test(s)) return null;
  return '#' + s.toLowerCase();
}

/**
 * Parse a palette given as a comma / whitespace separated hex list (or an
 * array). Returns an array of at least two canonical `#rrggbb` strings.
 * `fallback === null` → return null when fewer than two colours parse;
 * an array fallback is returned (copied) in that case; otherwise the default.
 */
export function parsePalette(value, fallback) {
  const items = Array.isArray(value)
    ? value
    : String(value == null ? '' : value)
        .split(/[\s,;]+/)
        .filter(Boolean);
  const out = [];
  for (const item of items) {
    const hex = normalizeHex(item);
    if (hex) out.push(hex);
    if (out.length >= MAX_PALETTE) break;
  }
  if (out.length >= 2) return out;
  if (fallback === null) return null;
  if (Array.isArray(fallback) && fallback.length >= 2) return fallback.slice();
  return DEFAULT_PALETTE.split(',');
}

/** `#rrggbb` → [r, g, b] in 0..1. */
export function hexToUnit(hex, fallback = [0, 0, 0]) {
  const h = normalizeHex(hex);
  if (!h) return fallback;
  const n = parseInt(h.slice(1), 16);
  return [((n >> 16) & 255) / 255, ((n >> 8) & 255) / 255, (n & 255) / 255];
}

/** [r, g, b] in 0..1 → `#rrggbb`. */
export function unitToHex(rgb) {
  const c = (v) => {
    const n = Math.max(0, Math.min(255, Math.round((Number(v) || 0) * 255)));
    return (n < 16 ? '0' : '') + n.toString(16);
  };
  return '#' + c(rgb[0]) + c(rgb[1]) + c(rgb[2]);
}

/** Relative luminance of a 0..1 RGB triplet. */
export function luminance(rgb) {
  return 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2];
}

/** Palette (hex list) → flat Float32Array [r,g,b, r,g,b, …] in 0..1. */
export function paletteToFloats(list) {
  const out = new Float32Array(list.length * 3);
  for (let k = 0; k < list.length; k++) {
    const c = hexToUnit(list[k]);
    out[k * 3] = c[0];
    out[k * 3 + 1] = c[1];
    out[k * 3 + 2] = c[2];
  }
  return out;
}

export function sortPaletteByLuminance(list) {
  return list.slice().sort((a, b) => luminance(hexToUnit(a)) - luminance(hexToUnit(b)));
}

/**
 * Channel weights for the colour distance. 'perceptual' is 3× the classic
 * Rec. 601 luma weights so the scale matches plain RGB on grey ramps.
 */
export const COLOR_WEIGHTS = {
  rgb: [1, 1, 1],
  perceptual: [0.897, 1.761, 0.342],
};

/** Index of the palette entry nearest (r, g, b) under weights `w`. */
export function nearestPaletteIndex(pal, n, r, g, b, w = COLOR_WEIGHTS.rgb) {
  let best = 0;
  let bestD = Infinity;
  for (let k = 0, q = 0; k < n; k++, q += 3) {
    const dr = pal[q] - r;
    const dg = pal[q + 1] - g;
    const db = pal[q + 2] - b;
    const d = w[0] * dr * dr + w[1] * dg * dg + w[2] * db * db;
    if (d < bestD) {
      bestD = d;
      best = k;
    }
  }
  return best;
}

/**
 * Fill `out` (Float32Array w*h*3, 0..1) with a two-colour gradient.
 * kind: 'horizontal' | 'vertical' | 'diagonal' | 'radial' (a at centre).
 * Passing a === b yields a solid fill.
 */
export function fillGradientField(out, w, h, a, b, kind) {
  const mw = Math.max(1, w - 1);
  const mh = Math.max(1, h - 1);
  const cx = mw / 2;
  const cy = mh / 2;
  const corner = Math.hypot(cx, cy) || 1;
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      let t;
      switch (kind) {
        case 'vertical':
          t = y / mh;
          break;
        case 'diagonal':
          t = (x / mw + y / mh) / 2;
          break;
        case 'radial':
          t = Math.hypot(x - cx, y - cy) / corner;
          break;
        case 'horizontal':
        default:
          t = x / mw;
          break;
      }
      if (t < 0) t = 0;
      else if (t > 1) t = 1;
      const j = (y * w + x) * 3;
      out[j] = a[0] + (b[0] - a[0]) * t;
      out[j + 1] = a[1] + (b[1] - a[1]) * t;
      out[j + 2] = a[2] + (b[2] - a[2]) * t;
    }
  }
  return out;
}

/**
 * Resample an image (HTMLImageElement / ImageBitmap / canvas / video frame)
 * into a w×h cell-resolution RGB field (Float32Array, 0..1).
 * fit: 'cover' | 'contain' | 'stretch'. Letterbox area is `background`.
 * Returns null if the image has no size or the canvas is tainted (CORS).
 */
export function sampleImageToField(image, w, h, fit, background) {
  if (!image || typeof document === 'undefined') return null;
  const iw = image.naturalWidth || image.videoWidth || image.width;
  const ih = image.naturalHeight || image.videoHeight || image.height;
  if (!iw || !ih) return null;
  const canvas = document.createElement('canvas');
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext('2d');
  ctx.fillStyle = background || '#000000';
  ctx.fillRect(0, 0, w, h);
  let dw = w;
  let dh = h;
  let dx = 0;
  let dy = 0;
  if (fit !== 'stretch') {
    const s = fit === 'contain' ? Math.min(w / iw, h / ih) : Math.max(w / iw, h / ih);
    dw = iw * s;
    dh = ih * s;
    dx = (w - dw) / 2;
    dy = (h - dh) / 2;
  }
  ctx.imageSmoothingEnabled = true;
  if ('imageSmoothingQuality' in ctx) ctx.imageSmoothingQuality = 'high';
  try {
    ctx.drawImage(image, dx, dy, dw, dh);
  } catch (err) {
    return null;
  }
  let data;
  try {
    data = ctx.getImageData(0, 0, w, h).data;
  } catch (err) {
    // tainted canvas (cross-origin image without CORS headers)
    return null;
  }
  const out = new Float32Array(w * h * 3);
  for (let i = 0, p = 0, j = 0; i < w * h; i++, p += 4, j += 3) {
    out[j] = data[p] / 255;
    out[j + 1] = data[p + 1] / 255;
    out[j + 2] = data[p + 2] / 255;
  }
  return out;
}

/**
 * Deterministic k-means over an RGB field → `k` palette colours (hex list,
 * sorted by luminance, duplicates removed). Initial centres are luminance
 * quantiles of an evenly strided sample so the result is reproducible.
 */
export function quantizePalette(field, k, iterations = 16) {
  const n = Math.floor((field ? field.length : 0) / 3);
  if (!n) return null;
  const kk = Math.max(2, Math.min(MAX_PALETTE, k | 0));
  const stride = Math.max(1, Math.floor(n / 4096));
  const samples = [];
  for (let i = 0; i < n; i += stride) {
    samples.push([field[i * 3], field[i * 3 + 1], field[i * 3 + 2]]);
  }
  samples.sort((p, q) => luminance(p) - luminance(q));
  const centres = [];
  for (let c = 0; c < kk; c++) {
    const s = samples[Math.min(samples.length - 1, Math.floor(((c + 0.5) / kk) * samples.length))];
    centres.push([s[0], s[1], s[2]]);
  }
  for (let it = 0; it < iterations; it++) {
    const sum = centres.map(() => [0, 0, 0, 0]);
    for (let i = 0; i < samples.length; i++) {
      const p = samples[i];
      let best = 0;
      let bd = Infinity;
      for (let c = 0; c < kk; c++) {
        const q = centres[c];
        const d = (p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2 + (p[2] - q[2]) ** 2;
        if (d < bd) {
          bd = d;
          best = c;
        }
      }
      const s = sum[best];
      s[0] += p[0];
      s[1] += p[1];
      s[2] += p[2];
      s[3]++;
    }
    let moved = 0;
    for (let c = 0; c < kk; c++) {
      const s = sum[c];
      if (!s[3]) continue;
      const nc = [s[0] / s[3], s[1] / s[3], s[2] / s[3]];
      moved += Math.abs(nc[0] - centres[c][0]) + Math.abs(nc[1] - centres[c][1]) + Math.abs(nc[2] - centres[c][2]);
      centres[c] = nc;
    }
    if (moved < 1e-4) break;
  }
  centres.sort((p, q) => luminance(p) - luminance(q));
  const hexes = [];
  for (const c of centres) {
    const h = unitToHex(c);
    if (!hexes.includes(h)) hexes.push(h);
  }
  while (hexes.length < 2) hexes.push(hexes[0] === '#ffffff' ? '#000000' : '#ffffff');
  return hexes;
}

export default parsePalette;