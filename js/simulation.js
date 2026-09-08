/**
 * simulation.js — one discrete timestep per invocation + playback scheduling (§7.5, §8).
 *
 * Guarantees synchronous update semantics: every cell reads only the frozen
 * previous-step buffers, and all writes land in the "next" buffers, which are
 * swapped atomically at the end of the step.
 */

import {
   SCHEMA,
   targetAt,
   isTargetSpatiallyUniform,
   TEXT_FIELD_KEYS,
   DITHER_TARGET_KEYS,
} from './config.js';
import {
  Grid,
  populate,
  populateMembrane,
  neighborOffsets,
  countActiveNeighbors,
  sumNeighborStates,
  sumNeighborVoltageDelta,
   sumNeighborColors,
  makeActivePredicate,
  createRng,
  makeGaussianSampler,
} from './grid.js';
import { pidStep } from './controller.js';
import { expressState, expressBioelectrical } from './stateExpression.js';
import { membraneStep, makeMembraneInput, makeMembraneOutput } from './membrane.js';
import { forEachLineCell, rasterizeText, fitTextBlock, fontStack } from './raster.js';
import {
   parsePalette,
   paletteToFloats,
   hexToUnit,
   COLOR_WEIGHTS,
   nearestPaletteIndex,
   fillGradientField,
   sampleImageToField,
   quantizePalette,
} from './dither.js';

export class Simulation {
  constructor(config) {
    this.config = config;
    const cfg = config.all();

    this.grid = new Grid(cfg.gridWidth, cfg.gridHeight);
    this.time = 0;
    this.running = false;
    this.measuredRate = 0;
    this.stats = {
      step: 0,
      activeFraction: 0,
      meanAbsError: 0,
      energy: 0,
      meanIntegral: 0,
      firingFraction: 0,
      refractoryFraction: 0,
      meanV: 0,
    };

    this._listeners = Object.create(null);
    this._out = { p: 0, i: 0, d: 0, u: 0, error: 0 };
    this._mIn = makeMembraneInput();
    this._mOut = makeMembraneOutput();
    this._accumulator = 0;
    this._lastFrame = 0;
    this._raf = null;
    this._rateSteps = 0;
    this._rateT0 = 0;
    this._tick = this._tick.bind(this);
     // ---- Dither-CA scratch buffers + image sources (§11) -----------------
     this._colorAcc = new Float32Array(3);
     this._want = new Float32Array(3);
     this._distScratch = new Float32Array(2);
     this._colorWeights = COLOR_WEIGHTS.rgb;
     this._ditherUpload = null;
     this._ditherUploadLabel = '';
     this._ditherUrlImage = null;
     this._ditherUrl = null;
     this._ditherUrlError = null;
     /** Human-readable description of the current colour target (UI read-out). */
     this.ditherImageInfo = '';

    this._unsubscribe = config.subscribe((changed) => this._onConfigChange(changed));
    this.reset();
  }

  // ------------------------------------------------------------- event bus
  on(event, fn) {
    const set = this._listeners[event] || (this._listeners[event] = new Set());
    set.add(fn);
    return () => set.delete(fn);
  }

  emit(event, payload) {
    const set = this._listeners[event];
    if (!set) return;
    for (const fn of [...set]) {
      try {
        fn(payload, this);
      } catch (err) {
        console.error('simulation listener failed', err);
      }
    }
  }

  // ------------------------------------------------------- config reaction
  _onConfigChange(changed) {
     const cfg = this.config.all();
    const structural = changed.some((k) => SCHEMA[k] && SCHEMA[k].structural);
     if (cfg.mode === 'pid' && (changed.includes('stateMin') || changed.includes('stateMax'))) {
       this.grid.clampStates(cfg.stateMin, cfg.stateMax);
    }
    this._refreshDerived();
     if (cfg.mode === 'dither') {
       // Palette edits re-index the states; target edits regenerate T(c) live.
       if (changed.includes('ditherPalette')) this.grid.clampStates(0, this.paletteSize - 1);
       if (changed.some((k) => DITHER_TARGET_KEYS.includes(k))) {
         this._syncDitherImageUrl(cfg);
         this._refreshDitherTarget();
       }
     }
    if (structural) {
      this.reset();
      return;
    }
    // Live re-rasterisation of the text target field (§3.2).
    if (
      this.config.get('targetMode') === 'text' &&
      changed.some((k) => TEXT_FIELD_KEYS.includes(k))
    ) {
      this.renderTextField();
    }
    this.emit('change', changed);
  }

  _refreshDerived() {
    const cfg = this.config.all();
    this.offsets = neighborOffsets(cfg.neighborhood, cfg.radius, cfg.neighborhoodMask);
    this.isActive = makeActivePredicate(cfg.activePredicate, cfg);
    this.sumMode = cfg.neighborMetric === 'sum';
    this.maxNeighbors = this.offsets.length / 2;
     // Dither-CA palette (cheap; kept current in every mode so paint tools work).
     this.paletteHex = parsePalette(cfg.ditherPalette);
     this.paletteRGB = paletteToFloats(this.paletteHex);
     this.paletteSize = this.paletteHex.length;
     this._colorWeights = COLOR_WEIGHTS[cfg.ditherMetric] || COLOR_WEIGHTS.rgb;
     if (this._distScratch.length !== this.paletteSize) {
       this._distScratch = new Float32Array(this.paletteSize);
     }
  }
  /** N_t(c): either the active-neighbour count or the signed neighbour sum. */
  _neighborMeasure(states, x, y, boundary) {
    return this.sumMode
      ? sumNeighborStates(this.grid, states, x, y, this.offsets, boundary)
      : countActiveNeighbors(this.grid, states, x, y, this.offsets, boundary, this.isActive);
  }

  // -------------------------------------------------------------- lifecycle
  reset() {
    const cfg = this.config.all();
    if (this.grid.width !== cfg.gridWidth || this.grid.height !== cfg.gridHeight) {
      this.grid.resize(cfg.gridWidth, cfg.gridHeight);
    }
    this._refreshDerived();
    this.rng = createRng(cfg.seed);
    this.gaussian = makeGaussianSampler(this.rng);
    this.grid.clearControllerState();
    if (cfg.targetMode === 'text') this.renderTextField();
    else this.ensureTargetField();
    if (cfg.mode === 'pid') {
      populate(this.grid, cfg, this.rng);
      this._seedControllerState();
     } else if (cfg.mode === 'dither') {
       this.grid.ensureColorBuffers();
       this._syncDitherImageUrl(cfg);
       this._refreshDitherTarget();
       this._populateDither(cfg);
       this._seedDitherState();
    } else {
      populateMembrane(this.grid, cfg, this.rng);
      this._seedMembraneState();
    }
    this.time = 0;
    this._accumulator = 0;
    this._measure();
    this.emit('reset');
  }

  /** Empty the grid and its controller memory without touching the config. */
  clear() {
    const cfg = this.config.all();
    this.grid.clearStates();
    this.grid.clearControllerState();
    if (cfg.mode === 'pid') {
      this._seedControllerState();
     } else if (cfg.mode === 'dither') {
       // all cells = palette colour 0, controller memory re-seeded
       this._seedDitherState();
    } else {
      this.grid.clearMembrane(cfg.vRest);
      this._seedMembraneState();
    }
    this.time = 0;
    this._measure();
    this.emit('reset');
  }

  /** Seed e_(t-1) from the initial neighbourhood so the first D_t term is 0. */
  _seedControllerState() {
    const cfg = this.config.all();
    const g = this.grid;
    const uniform = isTargetSpatiallyUniform(cfg) ? targetAt(cfg, 0, 0, 0) : null;
    const painted = this._targetBuffer(cfg);
    const gaussian = cfg.perturbInit === 'normal' ? makeGaussianSampler(this.rng) : null;
    for (let y = 0; y < g.height; y++) {
      const row = y * g.width;
      for (let x = 0; x < g.width; x++) {
        const idx = row + x;
        const n = this._neighborMeasure(g.states, x, y, cfg.boundary);
        const T = painted ? painted[idx] : uniform !== null ? uniform : targetAt(cfg, x, y, 0);
        const e = T - n;
        let prevError = e;
        let integral = 0;
        if (gaussian) {
          prevError += gaussian() * cfg.perturbSigma;
          integral += gaussian() * cfg.perturbSigma;
        }
        g.prevError[idx] = prevError;
        g.error[idx] = e;
        g.integral[idx] = integral;
        g.u[idx] = 0;
      }
    }
  }
  /**
   * Derive the initial display state from (V, gate), and — in pid-homeostat
   * mode — seed e_(t-1) so the first derivative term is zero.
   */
  _seedMembraneState() {
    const cfg = this.config.all();
    const g = this.grid;
    const homeostat = cfg.mode === 'pid-homeostat';
    for (let i = 0; i < g.size; i++) {
      g.states[i] = expressBioelectrical(g.V[i], g.gate[i]);
      g.nextStates[i] = g.states[i];
      if (homeostat) {
        const e = cfg.vTarget - g.V[i];
        g.prevError[i] = e;
        g.nextPrevError[i] = e;
        g.error[i] = e;
        g.integral[i] = 0;
        g.nextIntegral[i] = 0;
        g.u[i] = 0;
      }
    }
  }

  /** Live intervention / painting (§7.7). Controller memory is preserved. */
  paintCell(x, y, value) {
     let v = value;
     if (this.config.get('mode') === 'dither') {
       v = Math.max(0, Math.min(this.paletteSize - 1, v | 0));
     }
     this.grid.setState(x, y, v);
    this.emit('paint', { x, y, value });
  }
   // ---------------------------------------------- Dither-CA: colour target T(c)
   /** Initial palette indices for the dither domain (§11). */
   _populateDither(cfg) {
     const g = this.grid;
     const n = this.paletteSize;
     g.clearStates();
     const states = g.states;
     switch (cfg.ditherInit) {
       case 'first':
         break;
       case 'nearest': {
         const T = g.targetRGB;
         const pal = this.paletteRGB;
         const w = this._colorWeights;
         for (let i = 0, j = 0; i < g.size; i++, j += 3) {
           states[i] = nearestPaletteIndex(pal, n, T[j], T[j + 1], T[j + 2], w);
         }
         break;
       }
       case 'random':
       default:
         for (let i = 0; i < g.size; i++) states[i] = Math.floor(this.rng() * n);
         break;
     }
   }
   /** Seed per-channel e_(t-1) from the initial neighbourhood so D_0 = 0. */
   _seedDitherState() {
     const cfg = this.config.all();
     const g = this.grid;
     g.ensureColorBuffers();
     const pal = this.paletteRGB;
     const acc = this._colorAcc;
     const T = g.targetRGB;
     for (let y = 0; y < g.height; y++) {
       for (let x = 0; x < g.width; x++) {
         const idx = y * g.width + x;
         acc[0] = acc[1] = acc[2] = 0;
         let count = sumNeighborColors(g, g.states, pal, x, y, this.offsets, cfg.boundary, acc);
         if (cfg.ditherIncludeSelf) {
           const s = g.states[idx] * 3;
           acc[0] += pal[s];
           acc[1] += pal[s + 1];
           acc[2] += pal[s + 2];
           count++;
         }
         const inv = count ? 1 / count : 0;
         let eSum = 0;
         for (let c = 0; c < 3; c++) {
           const j = idx * 3 + c;
           const e = T[j] - acc[c] * inv;
           g.prevErrorRGB[j] = e;
           g.errorRGB[j] = e;
           g.integralRGB[j] = 0;
           g.uRGB[j] = 0;
           eSum += e;
         }
         g.error[idx] = eSum / 3;
         g.prevError[idx] = eSum / 3;
         g.integral[idx] = 0;
         g.u[idx] = 0;
       }
     }
   }
   /**
    * (Re)generate the colour target field from the configured source. Image
    * sources fall back to the gradient until an image is available/readable.
    */
   _refreshDitherTarget() {
     const cfg = this.config.all();
     const g = this.grid;
     g.ensureColorBuffers();
     const a = hexToUnit(cfg.ditherColorA);
     const b = hexToUnit(cfg.ditherColorB);
     let info = '';
     if (cfg.ditherSource === 'image') {
       const image = this._ditherUpload || this._ditherUrlImage;
       if (image) {
         const field = sampleImageToField(image, g.width, g.height, cfg.ditherImageFit, cfg.ditherColorA);
         if (field) {
           g.targetRGB.set(field);
           const iw = image.naturalWidth || image.width;
           const ih = image.naturalHeight || image.height;
           this.ditherImageInfo =
             (this._ditherUpload ? this._ditherUploadLabel || 'uploaded image' : 'image URL') +
             ' ' + iw + '×' + ih + ' → ' + g.width + '×' + g.height + ' cells (' + cfg.ditherImageFit + ')';
           return;
         }
         info = 'image could not be read (cross-origin without CORS?) — showing the gradient instead';
       } else if (this._ditherUrlError) {
         info = this._ditherUrlError + ' — showing the gradient instead';
       } else if (this._ditherUrl) {
         info = 'loading ' + this._ditherUrl + ' …';
       } else {
         info = 'no image loaded — upload one or set an image URL; showing the gradient';
       }
     }
     fillGradientField(g.targetRGB, g.width, g.height, a, cfg.ditherSource === 'solid' ? a : b, cfg.ditherGradient);
     this.ditherImageInfo = info;
   }
   /** Start / cancel loading of `ditherImageUrl`; the result lands in T(c) asynchronously. */
   _syncDitherImageUrl(cfg) {
     const url = cfg.ditherSource === 'image' ? String(cfg.ditherImageUrl || '').trim() : '';
     if (url === this._ditherUrl) return;
     this._ditherUrl = url;
     this._ditherUrlImage = null;
     this._ditherUrlError = null;
     if (!url || typeof Image !== 'function') return;
     const img = new Image();
     img.crossOrigin = 'anonymous';
     img.onload = () => {
       if (this._ditherUrl !== url) return;
       this._ditherUrlImage = img;
       this._refreshDitherTarget();
       this.emit('paint', { target: true, dither: true });
     };
     img.onerror = () => {
       if (this._ditherUrl !== url) return;
       this._ditherUrlError = 'could not load image URL ' + url;
       this._refreshDitherTarget();
       this.emit('paint', { target: true, dither: true });
     };
     img.src = url;
   }
   /**
    * Use an uploaded image (HTMLImageElement / ImageBitmap / canvas) as the
    * colour target. Overrides `ditherImageUrl` until cleared. The controller
    * memory is kept, so the texture morphs toward the new picture.
    */
   setDitherImage(image, label) {
     this._ditherUpload = image || null;
     this._ditherUploadLabel = label ? String(label) : '';
     if (this.config.get('mode') === 'dither') this._refreshDitherTarget();
     this.emit('paint', { target: true, dither: true });
   }
   clearDitherImage() {
     this.setDitherImage(null);
   }
   /** k-means the current colour target into `k` palette colours (hex list). */
   extractPalette(k) {
     if (!this.grid.targetRGB) return null;
     return quantizePalette(this.grid.targetRGB, k);
   }
  // ------------------------------------------------ painted target field T(c)
  /** Per-cell target buffer for the field-backed modes, else null. */
  _targetBuffer(cfg) {
    return cfg.targetMode === 'painted' || cfg.targetMode === 'text' ? this.grid.targetField : null;
  }
  /** Lazily initialise the field to the scalar T so it is never all-zero. */
  ensureTargetField() {
    if (!this.grid.targetInitialized) this.grid.fillTargetField(this.config.get('target'));
  }
  /**
   * Rasterise the configured text block into T(c): the background is the
   * scalar T, the glyph ink is `textFieldValue`. The block is auto-centred and
   * auto-fitted so that the *greater* of its effective width % / height %
   * equals `textFieldFit` percent of the grid.
   */
  renderTextField() {
    const cfg = this.config.all();
    const g = this.grid;
    g.fillTargetField(cfg.target);
    const text = String(cfg.textFieldText == null ? '' : cfg.textFieldText);
    if (!text.replace(/\s/g, '')) {
      this.emit('paint', { target: true, text: true });
      return;
    }
    const frac = Math.max(0.01, Math.min(1, cfg.textFieldFit / 100));
    const glyph = fitTextBlock(text, g.width * frac, g.height * frac, {
      family: fontStack(cfg.textFieldFont),
      bold: cfg.textFieldBold,
      italic: cfg.textFieldItalic,
      lineHeight: cfg.textFieldLineHeight,
      align: cfg.textFieldAlign,
    });
    if (!glyph.width) {
      this.emit('paint', { target: true, text: true });
      return;
    }
    const x0 = Math.round((g.width - glyph.width) / 2 + (cfg.textFieldOffsetX / 100) * g.width);
    const y0 = Math.round((g.height - glyph.height) / 2 + (cfg.textFieldOffsetY / 100) * g.height);
    for (let gy = 0; gy < glyph.height; gy++) {
      const row = gy * glyph.width;
      for (let gx = 0; gx < glyph.width; gx++) {
        if (glyph.mask[row + gx]) g.setTarget(x0 + gx, y0 + gy, cfg.textFieldValue);
      }
    }
    this.textFieldMetrics = {
      fontSize: glyph.fontSize,
      width: glyph.width,
      height: glyph.height,
      widthPercent: (glyph.width / g.width) * 100,
      heightPercent: (glyph.height / g.height) * 100,
    };
    this.emit('paint', { target: true, text: true });
  }
  /** Flood the whole field with a single value. */
  fillTargetField(value) {
    this.grid.fillTargetField(value);
    this.emit('paint', { target: true, value });
  }
  /** Square stamp, no event (internal building block for the tools). */
  _brushTarget(cx, cy, value, size) {
    const s = Math.max(1, size | 0);
    const half = (s - 1) / 2;
    const x0 = Math.round(cx - half);
    const y0 = Math.round(cy - half);
    for (let dy = 0; dy < s; dy++) {
      for (let dx = 0; dx < s; dx++) this.grid.setTarget(x0 + dx, y0 + dy, value);
    }
  }
  /** Freehand dab. */
  paintTargetBrush(x, y, value, size) {
    this.ensureTargetField();
    this._brushTarget(x, y, value, size);
    this.emit('paint', { x, y, value, target: true });
  }
  /** Stroke a line of brush stamps (also used to interpolate freehand drags). */
  paintTargetLine(x0, y0, x1, y1, value, size) {
    this.ensureTargetField();
    forEachLineCell(x0, y0, x1, y1, (x, y) => this._brushTarget(x, y, value, size));
    this.emit('paint', { target: true, value });
  }
  /** Filled rectangle between two corners (inclusive). */
  paintTargetRect(a, b, value) {
    this.ensureTargetField();
    const g = this.grid;
    const x0 = Math.max(0, Math.min(a.x, b.x));
    const x1 = Math.min(g.width - 1, Math.max(a.x, b.x));
    const y0 = Math.max(0, Math.min(a.y, b.y));
    const y1 = Math.min(g.height - 1, Math.max(a.y, b.y));
    for (let y = y0; y <= y1; y++) {
      for (let x = x0; x <= x1; x++) g.setTarget(x, y, value);
    }
    this.emit('paint', { target: true, value });
  }
  /** Rasterise `text` at `fontSize` cells and stamp it centred on (x, y). */
  stampTargetText(x, y, text, fontSize, value, bold) {
    const label = String(text == null ? '' : text);
    if (!label.trim()) return;
    this.ensureTargetField();
    const glyph = rasterizeText(label, fontSize, { bold: Boolean(bold) });
    const g = this.grid;
    const x0 = Math.round(x - glyph.width / 2);
    const y0 = Math.round(y - glyph.height / 2);
    for (let gy = 0; gy < glyph.height; gy++) {
      const row = gy * glyph.width;
      for (let gx = 0; gx < glyph.width; gx++) {
        if (glyph.mask[row + gx]) g.setTarget(x0 + gx, y0 + gy, value);
      }
    }
    this.emit('paint', { x, y, value, target: true });
  }

  // ------------------------------------------- membrane-domain interventions
  /** Write into the stimulus field (bioelectrical.md §7 "paint stimulus"). */
  paintStimulus(x, y, amount) {
    const g = this.grid;
    if (!g.inBounds(x, y)) return;
    g.stimulus[g.index(x, y)] = amount;
    this.emit('paint', { x, y, value: amount });
  }
  /** Directly set a cell's membrane potential (manual excitation). */
  paintVoltage(x, y, V) {
    const cfg = this.config.all();
    const g = this.grid;
    if (!g.inBounds(x, y)) return;
    const idx = g.index(x, y);
    g.V[idx] = V < cfg.vMin ? cfg.vMin : V > cfg.vMax ? cfg.vMax : V;
    this.emit('paint', { x, y, value: g.V[idx] });
  }
  /** Pin / unpin a cell's potential — pacemakers, boundaries, conduction block. */
  paintClamp(x, y, on, V) {
    const cfg = this.config.all();
    const g = this.grid;
    if (!g.inBounds(x, y)) return;
    const idx = g.index(x, y);
    if (on) {
      const v = V < cfg.vMin ? cfg.vMin : V > cfg.vMax ? cfg.vMax : V;
      g.clamped[idx] = 1;
      g.clampV[idx] = v;
      g.V[idx] = v;
    } else {
      g.clamped[idx] = 0;
      g.clampV[idx] = cfg.vRest;
    }
    this.emit('paint', { x, y, value: g.clamped[idx] });
  }

  // -------------------------------------------------------------- one step
  step() {
    const cfg = this.config.all();
    if (cfg.mode === 'pid') this._stepPid(cfg);
     else if (cfg.mode === 'dither') this._stepDither(cfg);
    else this._stepMembrane(cfg);
    this.time++;
    this._measure();
    this._trackRate();
    this.emit('step');
  }
  _stepPid(cfg) {
    const g = this.grid;

    const st = g.states,
      pe = g.prevError,
      ig = g.integral;
    const nst = g.nextStates,
      npe = g.nextPrevError,
      nig = g.nextIntegral;
    const uBuf = g.u,
      eBuf = g.error;

    const offsets = this.offsets;
    const isActive = this.isActive;
    const sumMode = this.sumMode;
    const boundary = cfg.boundary;
    const expression = cfg.expression;
    const out = this._out;
    const rng = this.rng;
    const t = this.time;
    const uniformTarget = isTargetSpatiallyUniform(cfg) ? targetAt(cfg, 0, 0, t) : null;
    const painted = this._targetBuffer(cfg);

    for (let y = 0; y < g.height; y++) {
      const row = y * g.width;
      for (let x = 0; x < g.width; x++) {
        const idx = row + x;

        // (a) neighbour count from the frozen snapshot
        const n = sumMode
          ? sumNeighborStates(g, st, x, y, offsets, boundary)
          : countActiveNeighbors(g, st, x, y, offsets, boundary, isActive);

        // (b) error + PID terms
        const T = painted
          ? painted[idx]
          : uniformTarget !== null
            ? uniformTarget
            : targetAt(cfg, x, y, t);
        const e = T - n;
        pidStep(pe[idx], ig[idx], e, cfg, out);

        // (c) discretise u_t into the next expressed state
        nst[idx] = expressState(expression, out.u, out.p, out.i, out.d, cfg, rng);

        // carry forward controller memory + diagnostics
        npe[idx] = e;
        nig[idx] = out.i;
        uBuf[idx] = out.u;
        eBuf[idx] = e;
      }
    }

    // (d) atomic buffer swap
    g.commit();
  }
   /**
    * Dither-CA step (§11). For every cell: measure the mean palette colour of
    * the neighbourhood (frozen snapshot), run one PID update per channel on
    * e = T(c) − mean, then express the palette colour nearest T(c) + u_t.
    * Hysteresis biases the current colour, temperature makes the choice a
    * seeded softmin sample, and the update rate makes the CA asynchronous.
    */
   _stepDither(cfg) {
     const g = this.grid;
     g.ensureColorBuffers();
     const st = g.states,
       nst = g.nextStates;
     const pe = g.prevErrorRGB,
       npe = g.nextPrevErrorRGB;
     const ig = g.integralRGB,
       nig = g.nextIntegralRGB;
     const uRGB = g.uRGB,
       eRGB = g.errorRGB;
     // scalar (channel-mean) diagnostics keep the u / e / I overlays working
     const uS = g.u,
       eS = g.error,
       nigS = g.nextIntegral,
       npeS = g.nextPrevError;
     const T = g.targetRGB;
     const pal = this.paletteRGB;
     const n = this.paletteSize;
     const offsets = this.offsets;
     const boundary = cfg.boundary;
     const includeSelf = cfg.ditherIncludeSelf;
     const kp = cfg.kp,
       ki = cfg.ki,
       kd = cfg.kd;
     const clampI = cfg.integralClamp,
       iMin = cfg.integralMin,
       iMax = cfg.integralMax;
     const hyst = cfg.ditherHysteresis;
     const temp = cfg.ditherTemperature;
     const rate = cfg.ditherUpdateRate;
     const w = this._colorWeights;
     const rng = this.rng;
     const acc = this._colorAcc;
     const want = this._want;
     const dist = this._distScratch;
     for (let y = 0; y < g.height; y++) {
       const row = y * g.width;
       for (let x = 0; x < g.width; x++) {
         const idx = row + x;
         const j = idx * 3;
         // (a) mean neighbourhood colour from the frozen snapshot
         acc[0] = acc[1] = acc[2] = 0;
         let count = sumNeighborColors(g, st, pal, x, y, offsets, boundary, acc);
         const cur = st[idx];
         if (includeSelf) {
           const s = cur * 3;
           acc[0] += pal[s];
           acc[1] += pal[s + 1];
           acc[2] += pal[s + 2];
           count++;
         }
         const inv = count ? 1 / count : 0;
         // (b) per-channel PID; the desired colour is T + u
         let eSum = 0,
           uSum = 0,
           iSum = 0;
         for (let c = 0; c < 3; c++) {
           const t = T[j + c];
           const e = t - acc[c] * inv;
           let i = ig[j + c] + ki * e;
           if (clampI) {
             if (i < iMin) i = iMin;
             else if (i > iMax) i = iMax;
           }
           const u = kp * e + i + kd * (e - pe[j + c]);
           npe[j + c] = e;
           nig[j + c] = i;
           uRGB[j + c] = u;
           eRGB[j + c] = e;
           want[c] = t + u;
           eSum += e;
           uSum += u;
           iSum += i;
         }
         eS[idx] = eSum / 3;
         uS[idx] = uSum / 3;
         nigS[idx] = iSum / 3;
         npeS[idx] = eSum / 3;
         // (c) express: nearest palette colour (asynchronous / stochastic options)
         if (rate < 1 && rng() >= rate) {
           nst[idx] = cur;
           continue;
         }
         let best = 0;
         let bestD = Infinity;
         for (let k = 0, q = 0; k < n; k++, q += 3) {
           const dr = pal[q] - want[0];
           const dg = pal[q + 1] - want[1];
           const db = pal[q + 2] - want[2];
           let d = Math.sqrt(w[0] * dr * dr + w[1] * dg * dg + w[2] * db * db);
           if (k === cur && hyst > 0) d = d > hyst ? d - hyst : 0;
           dist[k] = d;
           if (d < bestD) {
             bestD = d;
             best = k;
           }
         }
         if (temp > 0) {
           let sum = 0;
           for (let k = 0; k < n; k++) {
             const wk = Math.exp(-(dist[k] - bestD) / temp);
             dist[k] = wk;
             sum += wk;
           }
           let r = rng() * sum;
           for (let k = 0; k < n; k++) {
             r -= dist[k];
             if (r <= 0) {
               best = k;
               break;
             }
           }
         }
         nst[idx] = best;
       }
     }
     // (d) atomic buffer swap
     g.commit();
   }
  /**
   * Bioelectrical membrane step (bioelectrical.md §4). Neighbour voltages are
   * read from the frozen snapshot only; no cell observes another cell's gate.
   */
  _stepMembrane(cfg) {
    const g = this.grid;
    const V = g.V,
      gate = g.gate,
      ot = g.openTicks,
      rt = g.restTicks;
    const nV = g.nextV,
      nGate = g.nextGate,
      nOt = g.nextOpenTicks,
      nRt = g.nextRestTicks;
    const offsets = this.offsets;
    const boundary = cfg.boundary;
    const input = this._mIn;
    const out = this._mOut;
    const pidOut = this._out;
    const homeostat = cfg.mode === 'pid-homeostat';
    const noisy = cfg.noiseAmplitude > 0;
    const gaussian = this.gaussian;
    const pulse = cfg.stimulusMode === 'pulse';
    for (let y = 0; y < g.height; y++) {
      const row = y * g.width;
      for (let x = 0; x < g.width; x++) {
        const idx = row + x;
        // 1. SENSE
        input.V = V[idx];
        input.gate = gate[idx];
        input.openTicks = ot[idx];
        input.restTicks = rt[idx];
        input.neighborSum = sumNeighborVoltageDelta(g, V, x, y, offsets, boundary);
        input.stimulus = g.stimulus[idx];
        input.noise = noisy ? gaussian() * cfg.noiseAmplitude : 0;
        input.clamp = g.clamped[idx] ? g.clampV[idx] : null;
        // optional homeostat: u_t replaces the fixed leak term (§6.3)
        if (homeostat) {
          const e = cfg.vTarget - input.V;
          pidStep(g.prevError[idx], g.integral[idx], e, cfg, pidOut);
          input.leak = pidOut.u;
          g.nextPrevError[idx] = e;
          g.nextIntegral[idx] = pidOut.i;
          g.error[idx] = e;
          g.u[idx] = pidOut.u;
        } else {
          input.leak = null;
        }
        // 2. INTEGRATE + 3. ADVANCE GATE
        membraneStep(input, cfg, out);
        nV[idx] = out.V;
        nGate[idx] = out.gate;
        nOt[idx] = out.openTicks;
        nRt[idx] = out.restTicks;
        // 4. EXPRESS
        g.nextStates[idx] = expressBioelectrical(out.V, out.gate);
        // painted stimulus is consumed unless explicitly held
        if (pulse && input.stimulus !== 0) g.stimulus[idx] = 0;
      }
    }
    g.commit();
  }

  _measure() {
     const mode = this.config.get('mode');
     if (mode === 'dither') {
       this._measureDither();
       return;
     }
     if (mode !== 'pid') {
      this._measureMembrane();
      return;
    }
    const g = this.grid;
    const isActive = this.isActive;
    let active = 0,
      sumAbs = 0,
      sumSq = 0,
      sumI = 0;
    for (let i = 0; i < g.size; i++) {
      if (isActive(g.states[i])) active++;
      const e = g.error[i];
      sumAbs += Math.abs(e);
      sumSq += e * e;
      sumI += g.integral[i];
    }
    this.stats = {
      step: this.time,
      activeFraction: active / g.size,
      meanAbsError: sumAbs / g.size,
      energy: sumSq,
      meanIntegral: sumI / g.size,
      firingFraction: 0,
      refractoryFraction: 0,
      meanV: 0,
    };
  }
   _measureDither() {
     const g = this.grid;
     g.ensureColorBuffers();
     const pal = this.paletteRGB;
     const st = g.states,
       e = g.errorRGB,
       I = g.integralRGB,
       T = g.targetRGB;
     let sumAbs = 0,
       sumSq = 0,
       sumI = 0,
       nonzero = 0;
     let gr = 0,
       gg = 0,
       gb = 0,
       tr = 0,
       tg = 0,
       tb = 0;
     for (let i = 0, j = 0; i < g.size; i++, j += 3) {
       if (st[i] !== 0) nonzero++;
       const s = st[i] * 3;
       gr += pal[s];
       gg += pal[s + 1];
       gb += pal[s + 2];
       tr += T[j];
       tg += T[j + 1];
       tb += T[j + 2];
       for (let c = 0; c < 3; c++) {
         const v = e[j + c];
         sumAbs += Math.abs(v);
         sumSq += v * v;
         sumI += I[j + c];
       }
     }
     const inv = 1 / g.size;
     const gridMean = [gr * inv, gg * inv, gb * inv];
     const targetMean = [tr * inv, tg * inv, tb * inv];
     this.stats = {
       step: this.time,
       activeFraction: nonzero / g.size,
       meanAbsError: sumAbs / (g.size * 3),
       energy: sumSq,
       meanIntegral: sumI / (g.size * 3),
       firingFraction: 0,
       refractoryFraction: 0,
       meanV: 0,
       gridMean,
       targetMean,
       globalDelta: Math.hypot(
         gridMean[0] - targetMean[0],
         gridMean[1] - targetMean[1],
         gridMean[2] - targetMean[2]
       ),
     };
   }
  _measureMembrane() {
    const g = this.grid;
    let open = 0,
      refractory = 0,
      sumV = 0,
      sumI = 0;
    for (let i = 0; i < g.size; i++) {
      const gate = g.gate[i];
      if (gate === 1) open++;
      else if (gate === 2) refractory++;
      sumV += g.V[i];
      sumI += g.integral[i];
    }
    this.stats = {
      step: this.time,
      activeFraction: open / g.size,
      firingFraction: open / g.size,
      refractoryFraction: refractory / g.size,
      meanV: sumV / g.size,
      meanAbsError: 0,
      energy: 0,
      meanIntegral: sumI / g.size,
    };
  }

  _trackRate() {
    const now = performance.now();
    if (this._rateT0 === 0) {
      this._rateT0 = now;
      this._rateSteps = 0;
    }
    this._rateSteps++;
    const dt = now - this._rateT0;
    if (dt >= 500) {
      this.measuredRate = (this._rateSteps * 1000) / dt;
      this._rateSteps = 0;
      this._rateT0 = now;
    }
  }

  // --------------------------------------------------------------- playback
  play() {
    if (this.running) return;
    this.running = true;
    this._lastFrame = performance.now();
    this._accumulator = 0;
    this._rateT0 = 0;
    this._raf = requestAnimationFrame(this._tick);
    this.emit('running', true);
  }

  pause() {
    if (!this.running) return;
    this.running = false;
    if (this._raf !== null) cancelAnimationFrame(this._raf);
    this._raf = null;
    this.measuredRate = 0;
    this.emit('running', false);
  }

  toggle() {
    if (this.running) this.pause();
    else this.play();
  }

  _tick(now) {
    if (!this.running) return;
    const cfg = this.config.all();
    const dt = Math.min(0.25, (now - this._lastFrame) / 1000);
    this._lastFrame = now;
    this._accumulator += dt * cfg.stepsPerSecond;

    let steps = Math.floor(this._accumulator);
    if (steps > cfg.maxStepsPerFrame) {
      steps = cfg.maxStepsPerFrame;
      this._accumulator = 0;
    } else {
      this._accumulator -= steps;
    }
    for (let k = 0; k < steps; k++) this.step();

    this._raf = requestAnimationFrame(this._tick);
  }

  dispose() {
    this.pause();
    if (this._unsubscribe) this._unsubscribe();
    this._listeners = Object.create(null);
  }
}