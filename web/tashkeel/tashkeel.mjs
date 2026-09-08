/**
 * Arabic diacritizer (tashkeel) for the browser — a hand-written port of the model
 * piper runs before espeak on Arabic text.
 *
 * Piper restores the short vowels that Arabic script omits and that espeak needs.
 * Without this step the browser feeds the voice a phoneme distribution it was never
 * distilled on. This module reproduces piper/tashkeel/__init__.py and the ONNX graph
 * of libtashkeel's model exactly, with no ONNX runtime.
 *
 * Source model and id maps: libtashkeel (https://github.com/mush42/libtashkeel), MIT,
 * as vendored by piper (MIT). Weights are re-encoded by tools/export_tashkeel_web.py.
 */

const CHAR_LIMIT = 12000;
const NUMERAL_SYMBOL = '#';
const D_MODEL = 56;
const N_HEADS = 8;
const HEAD_DIM = 7;
const N_LAYERS = 3;
const EPS_LN = 1e-5;
const EPS_BN = 9.999999747378752e-6;
const CHAR_EMB_SCALE = 16.0;
const DIAC_EMB_SCALE = 16.0;

const NUMERALS = new Set('0123456789٠١٢٣٤٥٦٧٨٩');
// U+064B..U+0652: the eight marks the model treats as diacritics.
const ARABIC_DIACRITICS = new Set(
  [0x64b, 0x64c, 0x64d, 0x64e, 0x64f, 0x650, 0x651, 0x652].map((c) => String.fromCharCode(c)),
);
// haraka+shadda in the order text usually carries it -> the shadda-first order the
// hint vocabulary is keyed on.
const NORMALIZED_DIAC_MAP = new Map([
  ['َّ', 'َّ'],
  ['ًّ', 'ًّ'],
  ['ُّ', 'ُّ'],
  ['ٌّ', 'ٌّ'],
  ['ِّ', 'ِّ'],
  ['ٍّ', 'ٍّ'],
]);

// Characters Python's str.strip() removes. JS trim() differs at both ends of this
// set (it keeps \x1c-\x1f and \x85, and it strips ﻿), so strip explicitly.
const PY_SPACE = new Set(
  [
    0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x1c, 0x1d, 0x1e, 0x1f, 0x20, 0x85, 0xa0, 0x1680, 0x2000,
    0x2001, 0x2002, 0x2003, 0x2004, 0x2005, 0x2006, 0x2007, 0x2008, 0x2009, 0x200a, 0x2028,
    0x2029, 0x202f, 0x205f, 0x3000,
  ].map((c) => String.fromCodePoint(c)),
);

export class TashkeelError extends Error {}

function pyStrip(text) {
  let a = 0;
  let b = text.length;
  while (a < b && PY_SPACE.has(text[a])) a += 1;
  while (b > a && PY_SPACE.has(text[b - 1])) b -= 1;
  return text.slice(a, b);
}

// ------------------------------------------------------------------ weight blob

function f16to32(bits) {
  const sign = (bits & 0x8000) ? -1 : 1;
  const exp = (bits >> 10) & 0x1f;
  const frac = bits & 0x3ff;
  if (exp === 0) return sign * frac * 5.960464477539063e-8;
  if (exp === 31) return frac ? NaN : sign * Infinity;
  return sign * Math.pow(2, exp - 25) * (1024 + frac);
}

function numel(shape) {
  return shape.reduce((a, b) => a * b, 1);
}

/**
 * Dequantise one manifest entry into a Float32Array. Every storage format is
 * expanded at load time so the forward pass only ever sees f32.
 */
function readTensor(entry, buf) {
  const n = numel(entry.shape);
  const out = new Float32Array(n);
  if (entry.dtype === 'f32') {
    out.set(new Float32Array(buf.slice(entry.offset, entry.offset + n * 4)));
    return out;
  }
  if (entry.dtype === 'f16') {
    const src = new Uint16Array(buf.slice(entry.offset, entry.offset + n * 2));
    for (let i = 0; i < n; i += 1) out[i] = f16to32(src[i]);
    return out;
  }
  if (entry.dtype !== 'q8' && entry.dtype !== 'q16') {
    throw new TashkeelError(`unsupported tensor dtype ${entry.dtype}`);
  }
  const q = entry.dtype === 'q8'
    ? new Int8Array(buf.slice(entry.offset, entry.offset + n))
    : new Int16Array(buf.slice(entry.offset, entry.offset + n * 2));
  const scaleN = numel(entry.scaleShape);
  const scale = new Float32Array(buf.slice(entry.scaleOffset, entry.scaleOffset + scaleN * 4));
  // The scale tensor is the weight shape with the quantised axes collapsed to 1, so
  // walk the weight in row-major order and index the scale with the same strides.
  const shape = entry.shape;
  const sShape = entry.scaleShape;
  const strides = new Array(shape.length);
  let acc = 1;
  for (let d = shape.length - 1; d >= 0; d -= 1) {
    strides[d] = sShape[d] === 1 ? 0 : acc;
    acc *= sShape[d];
  }
  const idx = new Array(shape.length).fill(0);
  for (let i = 0; i < n; i += 1) {
    let s = 0;
    for (let d = 0; d < shape.length; d += 1) s += idx[d] * strides[d];
    out[i] = q[i] * scale[s];
    for (let d = shape.length - 1; d >= 0; d -= 1) {
      idx[d] += 1;
      if (idx[d] < shape[d]) break;
      idx[d] = 0;
    }
  }
  return out;
}

// ------------------------------------------------------------------- math kernels

function layernorm(x, t, d, w, b, out) {
  for (let i = 0; i < t; i += 1) {
    const o = i * d;
    let mean = 0;
    for (let j = 0; j < d; j += 1) mean += x[o + j];
    mean /= d;
    let v = 0;
    for (let j = 0; j < d; j += 1) {
      const e = x[o + j] - mean;
      v += e * e;
    }
    const inv = 1 / Math.sqrt(v / d + EPS_LN);
    for (let j = 0; j < d; j += 1) out[o + j] = (x[o + j] - mean) * inv * w[j] + b[j];
  }
}

/** out(t,n) = x(t,m) @ W(m,n) + bias(n); bias may be null. */
function matmul(x, t, m, W, n, bias, out) {
  for (let i = 0; i < t; i += 1) {
    const xo = i * m;
    const oo = i * n;
    if (bias) for (let j = 0; j < n; j += 1) out[oo + j] = bias[j];
    else out.fill(0, oo, oo + n);
    for (let k = 0; k < m; k += 1) {
      const xv = x[xo + k];
      if (xv === 0) continue;
      const wo = k * n;
      for (let j = 0; j < n; j += 1) out[oo + j] += xv * W[wo + j];
    }
  }
}

function sigmoid(v) {
  return 1 / (1 + Math.exp(-v));
}

function siluInPlace(a, n) {
  for (let i = 0; i < n; i += 1) a[i] *= sigmoid(a[i]);
}

/** 'same'-padded 1-D convolution. x is (t, cin) and W is (cout, cin, k). */
function conv1d(x, t, cin, W, cout, k, bias, out) {
  const pad = (k - 1) >> 1;
  for (let i = 0; i < t; i += 1) {
    const oo = i * cout;
    for (let o = 0; o < cout; o += 1) {
      let s = bias[o];
      const wo = o * cin * k;
      for (let j = 0; j < k; j += 1) {
        const p = i + j - pad;
        if (p < 0 || p >= t) continue;
        const xo = p * cin;
        for (let c = 0; c < cin; c += 1) s += x[xo + c] * W[wo + c * k + j];
      }
      out[oo + o] = s;
    }
  }
}

/**
 * ONNX bidirectional GRU with linear_before_reset=1.
 * x is (t, cin); out is (t, 2*h) laid out [forward | backward] on the last axis.
 */
function biGru(x, t, cin, W, R, B, h, out) {
  const h3 = 3 * h;
  const gates = new Float64Array(t * h3);
  const state = new Float64Array(h);
  for (let d = 0; d < 2; d += 1) {
    const Wo = d * h3 * cin;
    const Ro = d * h3 * h;
    const Bo = d * 6 * h;
    // input side for the whole sequence, with both bias halves folded in for z and r
    for (let i = 0; i < t; i += 1) {
      const xo = i * cin;
      const go = i * h3;
      for (let r = 0; r < h3; r += 1) {
        let s = B[Bo + r] + (r < 2 * h ? B[Bo + 3 * h + r] : 0);
        const wr = Wo + r * cin;
        for (let c = 0; c < cin; c += 1) s += x[xo + c] * W[wr + c];
        gates[go + r] = s;
      }
    }
    state.fill(0);
    for (let n = 0; n < t; n += 1) {
      const i = d === 0 ? n : t - 1 - n;
      const go = i * h3;
      for (let j = 0; j < h; j += 1) {
        let z = gates[go + j];
        let r = gates[go + h + j];
        const rz = Ro + j * h;
        const rr = Ro + (h + j) * h;
        for (let c = 0; c < h; c += 1) {
          z += state[c] * R[rz + c];
          r += state[c] * R[rr + c];
        }
        out[i * 2 * h + d * h + j] = sigmoid(z);      // stash z
        gates[go + h + j] = sigmoid(r);               // stash r in place
      }
      for (let j = 0; j < h; j += 1) {
        let hh = 0;
        const rh = Ro + (2 * h + j) * h;
        for (let c = 0; c < h; c += 1) hh += state[c] * R[rh + c];
        hh = (hh + B[Bo + 5 * h + j]) * gates[go + h + j] + gates[go + 2 * h + j];
        const z = out[i * 2 * h + d * h + j];
        out[i * 2 * h + d * h + j] = (1 - z) * Math.tanh(hh) + z * state[j];
      }
      for (let j = 0; j < h; j += 1) state[j] = out[i * 2 * h + d * h + j];
    }
  }
}

// ---------------------------------------------------------------- the model

class TashkeelModel {
  constructor(manifest, buffer) {
    this.manifest = manifest;
    this.w = new Map();
    for (const entry of manifest.tensors) this.w.set(entry.name, readTensor(entry, buffer));
    this.posCache = null;
  }

  get(name) {
    const t = this.w.get(name);
    if (!t) throw new TashkeelError(`missing tensor ${name}`);
    return t;
  }

  posEnc(t) {
    if (this.posCache && this.posCache.t >= t) return this.posCache.a;
    const scale = this.get('pos_scale')[0];
    const a = new Float64Array(t * D_MODEL);
    for (let i = 0; i < t; i += 1) {
      for (let j = 0; j < 28; j += 1) {
        const ang = i * Math.pow(10000, -j / 28);
        a[i * D_MODEL + j] = Math.sin(ang) * scale;
        a[i * D_MODEL + 28 + j] = Math.cos(ang) * scale;
      }
    }
    this.posCache = { t, a };
    return a;
  }

  /** Returns Uint8Array of argmax class ids, one per input position. */
  predict(charIds, diacIds) {
    const t = charIds.length;
    const D = D_MODEL;
    const charEmb = this.get('char_emb');
    const diacEmb = this.get('diac_emb');
    const hintScale = this.get('hint_scale')[0] * DIAC_EMB_SCALE;

    let x = new Float64Array(t * D);
    for (let i = 0; i < t; i += 1) {
      const co = charIds[i] * D;
      const dof = diacIds[i] * D;
      for (let j = 0; j < D; j += 1) {
        x[i * D + j] = charEmb[co + j] * CHAR_EMB_SCALE + diacEmb[dof + j] * hintScale;
      }
    }

    // three stacked linear layers, no activation between them
    let buf = new Float64Array(t * 256);
    matmul(x, t, D, this.get('dense0_w'), 256, this.get('dense0_b'), buf);
    let buf2 = new Float64Array(t * 128);
    matmul(buf, t, 256, this.get('dense1_w'), 128, this.get('dense1_b'), buf2);
    matmul(buf2, t, 128, this.get('dense2_w'), D, this.get('dense2_b'), x);

    // four stacked convolutions, kernels 1/3/5/7, no activation between them
    let y = new Float64Array(t * D);
    for (let c = 0; c < 4; c += 1) {
      conv1d(x, t, D, this.get(`conv${c}_w`), D, 2 * c + 1, this.get(`conv${c}_b`), y);
      const tmp = x;
      x = y;
      y = tmp;
    }

    const residual = Float64Array.from(x);
    const pos = this.posEnc(t);
    let h = x;
    const tmpD = new Float64Array(t * D);
    const tmp224 = new Float64Array(t * 224);
    const g1 = new Float64Array(t * 224);
    const g2 = new Float64Array(t * 224);
    const gsmall = new Float64Array(t * 112);
    const q = new Float64Array(t * D);
    const kk = new Float64Array(t * D);
    const vv = new Float64Array(t * D);
    const att = new Float64Array(t);
    const ao = new Float64Array(t * D);

    for (let L = 0; L < N_LAYERS; L += 1) {
      // macaron feed-forward, half-weighted residual
      layernorm(h, t, D, this.get(`L${L}_ffm1_ln_w`), this.get(`L${L}_ffm1_ln_b`), tmpD);
      matmul(tmpD, t, D, this.get(`L${L}_ffm1_w1`), 224, this.get(`L${L}_ffm1_b1`), tmp224);
      siluInPlace(tmp224, t * 224);
      matmul(tmp224, t, 224, this.get(`L${L}_ffm1_w2`), D, this.get(`L${L}_ffm1_b2`), tmpD);
      for (let i = 0; i < t * D; i += 1) h[i] += 0.5 * tmpD[i];

      // self-attention over layernorm(h) + sinusoidal positions, 8 heads of 7
      layernorm(h, t, D, this.get(`L${L}_attn_ln_w`), this.get(`L${L}_attn_ln_b`), tmpD);
      for (let i = 0; i < t * D; i += 1) tmpD[i] += pos[i];
      matmul(tmpD, t, D, this.get(`L${L}_attn_q`), D, null, q);
      matmul(tmpD, t, D, this.get(`L${L}_attn_k`), D, null, kk);
      matmul(tmpD, t, D, this.get(`L${L}_attn_v`), D, null, vv);
      const s = Math.sqrt(1 / Math.sqrt(HEAD_DIM));
      ao.fill(0);
      for (let hd = 0; hd < N_HEADS; hd += 1) {
        const off = hd * HEAD_DIM;
        for (let i = 0; i < t; i += 1) {
          let mx = -Infinity;
          for (let j = 0; j < t; j += 1) {
            let dot = 0;
            for (let c = 0; c < HEAD_DIM; c += 1) {
              dot += q[i * D + off + c] * s * (kk[j * D + off + c] * s);
            }
            att[j] = dot;
            if (dot > mx) mx = dot;
          }
          let sum = 0;
          for (let j = 0; j < t; j += 1) {
            att[j] = Math.exp(att[j] - mx);
            sum += att[j];
          }
          for (let j = 0; j < t; j += 1) {
            const a = att[j] / sum;
            for (let c = 0; c < HEAD_DIM; c += 1) ao[i * D + off + c] += a * vv[j * D + off + c];
          }
        }
      }
      matmul(ao, t, D, this.get(`L${L}_attn_o`), D, null, tmpD);
      for (let i = 0; i < t * D; i += 1) h[i] += tmpD[i];

      // convolution/recurrence module: LN -> biGRU -> GLU -> biGRU -> BN -> SiLU -> biGRU
      layernorm(h, t, D, this.get(`L${L}_ccm_ln_w`), this.get(`L${L}_ccm_ln_b`), tmpD);
      biGru(tmpD, t, D, this.get(`L${L}_gru1_W`), this.get(`L${L}_gru1_R`),
            this.get(`L${L}_gru1_B`), 112, g1);
      for (let i = 0; i < t; i += 1) {
        for (let j = 0; j < 112; j += 1) {
          gsmall[i * 112 + j] = Math.tanh(g1[i * 224 + j] + g1[i * 224 + 112 + j]);
        }
      }
      for (let i = 0; i < t; i += 1) {
        for (let j = 0; j < D; j += 1) {
          tmpD[i * D + j] = gsmall[i * 112 + j] * sigmoid(gsmall[i * 112 + D + j]);
        }
      }
      biGru(tmpD, t, D, this.get(`L${L}_gru2_W`), this.get(`L${L}_gru2_R`),
            this.get(`L${L}_gru2_B`), 112, g2);
      const bw = this.get(`L${L}_bn_w`);
      const bb = this.get(`L${L}_bn_b`);
      const bm = this.get(`L${L}_bn_m`);
      const bv = this.get(`L${L}_bn_v`);
      for (let i = 0; i < t; i += 1) {
        for (let j = 0; j < 112; j += 1) {
          let v = Math.tanh(g2[i * 224 + j] + g2[i * 224 + 112 + j]);
          v = (v - bm[j]) / Math.sqrt(bv[j] + EPS_BN) * bw[j] + bb[j];
          gsmall[i * 112 + j] = v * sigmoid(v);
        }
      }
      biGru(gsmall, t, 112, this.get(`L${L}_gru3_W`), this.get(`L${L}_gru3_R`),
            this.get(`L${L}_gru3_B`), D, g1);
      for (let i = 0; i < t; i += 1) {
        for (let j = 0; j < D; j += 1) {
          h[i * D + j] += Math.tanh(g1[i * 2 * D + j] + g1[i * 2 * D + D + j]);
        }
      }

      layernorm(h, t, D, this.get(`L${L}_ffm2_ln_w`), this.get(`L${L}_ffm2_ln_b`), tmpD);
      matmul(tmpD, t, D, this.get(`L${L}_ffm2_w1`), 224, this.get(`L${L}_ffm2_b1`), tmp224);
      siluInPlace(tmp224, t * 224);
      matmul(tmp224, t, 224, this.get(`L${L}_ffm2_w2`), D, this.get(`L${L}_ffm2_b2`), tmpD);
      for (let i = 0; i < t * D; i += 1) h[i] += 0.5 * tmpD[i];

      layernorm(h, t, D, this.get(`L${L}_post_w`), this.get(`L${L}_post_b`), tmpD);
      for (let i = 0; i < t * D; i += 1) h[i] = Math.tanh(tmpD[i]);
    }

    for (let i = 0; i < t * D; i += 1) h[i] += residual[i];
    layernorm(h, t, D, this.get('res_ln_w'), this.get('res_ln_b'), tmpD);
    const logits = new Float64Array(t * 15);
    matmul(tmpD, t, D, this.get('fc_w'), 15, this.get('fc_b'), logits);
    const preds = new Uint8Array(t);
    for (let i = 0; i < t; i += 1) {
      let best = 0;
      let bv = -Infinity;
      for (let j = 0; j < 15; j += 1) {
        if (logits[i * 15 + j] > bv) {
          bv = logits[i * 15 + j];
          best = j;
        }
      }
      preds[i] = best;
    }
    return preds;
  }
}

// --------------------------------------------------------------- text pipeline

export class TashkeelDiacritizer {
  constructor(manifest, buffer) {
    this.model = new TashkeelModel(manifest, buffer);
    this.inputIdMap = new Map(Object.entries(manifest.input_id_map));
    this.hintIdMap = new Map(Object.entries(manifest.hint_id_map));
    this.idTargetMap = new Map();
    for (const [ch, id] of Object.entries(manifest.target_id_map)) this.idTargetMap.set(id, ch);
    this.metaTargetIds = new Set([manifest.target_id_map['_']]);
  }

  /** Keep model-known characters, fold digits to '#', and report what was dropped. */
  toValidChars(text) {
    let valid = '';
    const removed = new Set();
    for (const c of text) {
      if (this.inputIdMap.has(c) || ARABIC_DIACRITICS.has(c)) valid += c;
      else if (NUMERALS.has(c)) valid += NUMERAL_SYMBOL;
      else removed.add(c);
    }
    return { valid, removed };
  }

  /** Split into base characters and the (normalised) diacritic that follows each. */
  extractCharsAndDiacritics(text) {
    let start = 0;
    while (start < text.length && ARABIC_DIACRITICS.has(text[start])) start += 1;
    const body = text.slice(start);

    const clean = [];
    const diacritics = [];
    let pending = '';
    for (const c of `${body} `) {
      if (ARABIC_DIACRITICS.has(c)) {
        pending += c;
      } else {
        clean.push(c);
        diacritics.push(pending);
        pending = '';
      }
    }
    if (clean.length) clean.pop();
    if (diacritics.length) diacritics.shift();

    for (let i = 0; i < diacritics.length; i += 1) {
      if (!this.hintIdMap.has(diacritics[i])) {
        diacritics[i] = NORMALIZED_DIAC_MAP.get(diacritics[i]) ?? '';
      }
    }
    return { chars: clean.join(''), diacritics };
  }

  /**
   * Restore the short vowels. Mirrors piper's TashkeelDiacritizer.diacritize with
   * taskeen_threshold left at None, which is what piper actually runs: its __call__
   * takes the threshold and then drops it on the floor.
   */
  diacritize(text) {
    const stripped = pyStrip(text);
    if (stripped.length > CHAR_LIMIT) {
      throw new TashkeelError(`Text length cannot exceed ${CHAR_LIMIT}`);
    }
    const { valid, removed } = this.toValidChars(stripped);
    const { chars, diacritics } = this.extractCharsAndDiacritics(valid);
    if (chars.length === 0) return stripped;

    const charIds = new Int32Array(chars.length);
    const diacIds = new Int32Array(chars.length);
    let i = 0;
    for (const c of chars) {
      charIds[i] = this.inputIdMap.get(c);
      diacIds[i] = this.hintIdMap.get(diacritics[i]);
      i += 1;
    }

    const preds = this.model.predict(charIds, diacIds);
    // piper filters the padding class out of the prediction stream rather than
    // aligning it to a position, so the stream can be shorter than the text.
    const out = [];
    for (let j = 0; j < preds.length; j += 1) {
      if (!this.metaTargetIds.has(preds[j])) out.push(this.idTargetMap.get(preds[j]));
    }
    return this.annotate(stripped, out, removed);
  }

  annotate(text, diacritics, removed) {
    let n = 0;
    let result = '';
    for (const c of text) {
      if (ARABIC_DIACRITICS.has(c)) continue;
      if (removed.has(c)) {
        result += c;
      } else {
        result += c + (n < diacritics.length ? diacritics[n] : '');
        n += 1;
      }
    }
    return result;
  }
}

/** Fetch the manifest and weights from `baseUrl` and build a diacritizer. */
export async function loadTashkeel(baseUrl, fetchImpl = fetch) {
  const base = baseUrl.endsWith('/') ? baseUrl : `${baseUrl}/`;
  const manifestRes = await fetchImpl(`${base}tashkeel.json`);
  if (!manifestRes.ok) throw new TashkeelError(`cannot load tashkeel.json: ${manifestRes.status}`);
  const manifest = await manifestRes.json();
  const binRes = await fetchImpl(`${base}${manifest.weights}`);
  if (!binRes.ok) throw new TashkeelError(`cannot load ${manifest.weights}: ${binRes.status}`);
  return new TashkeelDiacritizer(manifest, await binRes.arrayBuffer());
}
