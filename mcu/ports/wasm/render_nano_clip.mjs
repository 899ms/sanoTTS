// render_nano_clip.mjs -- render a gallery clip for a snt_nano voice through
// the SAME C runtime the browser page runs, headless under Node.
//
// WHY THIS EXISTS
//
// web/index.html says of the heart-nano row: "What plays below is the same
// int8 arithmetic a microcontroller would run." That sentence is only true if
// the mp3 in web/samples-release/ came out of mcu/src/snt_nano.c. The Python
// runtime (pypkg/sanotts/nano.py) is NOT that arithmetic -- its own module
// docstring says so: it dequantises the int8 rows (w = q * scale) and runs
// float math, deliberately, because "a Python caller has floats". A pypkg
// render sounds fine and would silently falsify the page. So the gallery
// clips are rendered here instead.
//
// This driver reproduces synthNano() in web/index.html step for step:
//
//   web/trellis_frontend.js over web/snt_g2p.js   (espeak-ng 1.52.0 wasm ->
//     misaki E2M -> the frozen 62-symbol character-level vocabulary; NOT the
//     Piper id table the piperlite voices use)
//   -> web/voices/<voice>/meta.json + front/model blobs
//   -> web/snt_nano_<voice>.js  _snt_nano_wasm_synthesize()
//        durs = NULL            (the duration student runs, as in the browser)
//        seed = sha256(text)[:8] via _snt_nano_wasm_seed_from_text()
//        16 MB arena, 24000*30 sample cap -- the page's NANO_ARENA/NANO_OUT_CAP
//
// Output is a 32-bit float mono WAV at the module's own sample rate, so no
// requantisation happens before the mastering step.
//
//   node mcu/ports/wasm/render_nano_clip.mjs <voice> <out.wav> <text> [--split]
//
//   node mcu/ports/wasm/render_nano_clip.mjs heartnano /tmp/clip.wav \
//     "Welcome! This entire voice runs in under four hundred kilobytes, with no cloud at all."
//
// Default is ONE synthesize call for the whole text, which is what the
// shipped gallery clips are. --split applies the page's splitChunks() and
// concatenates, which is what a visitor typing the same text hears; it is
// audibly different (each chunk carries its own leading and trailing silence,
// so sentence gaps roughly double). Use --split only to reproduce the live
// page, never to master a gallery clip.
//
// Master the result to match the rest of the gallery (-24.5 LUFS, 24 kHz
// mono, libmp3lame 64 kbps), two-pass so the measurement is real:
//
//   ffmpeg -i clip.wav -af loudnorm=I=-24.5:TP=-1.5:LRA=11:print_format=json -f null -
//   ffmpeg -i clip.wav -af "loudnorm=I=-24.5:TP=-1.5:LRA=11:measured_I=…:\
//     measured_TP=…:measured_LRA=…:measured_thresh=…:offset=…:linear=true" \
//     -ar 24000 -ac 1 -c:a libmp3lame -b:a 64k clip.mp3
//
// Related: verify_nano_node.mjs is the correctness gate for the same modules
// (frozen fixture durations, correlation against the PyTorch reference). Run
// it first; this script assumes the module already passes.
//
// Exit 0 on success, 1 on a render failure, 2 on bad arguments or a missing
// input file.
import { readFileSync, writeFileSync, existsSync } from "node:fs";
import { fileURLToPath, pathToFileURL } from "node:url";
import { createRequire } from "node:module";
import { dirname, resolve } from "node:path";
import vm from "node:vm";

const require = createRequire(import.meta.url);
const here = dirname(fileURLToPath(import.meta.url));
const repo = resolve(here, "../../..");
const web = resolve(repo, "web");

const NANO_ARENA = 16 * 1024 * 1024;   // web/index.html NANO_ARENA
const NANO_OUT_CAP = 24000 * 30;       // web/index.html NANO_OUT_CAP

function usage(msg) {
  if (msg) console.error(`error: ${msg}`);
  console.error("usage: node render_nano_clip.mjs <voice> <out.wav> <text> [--split]");
  process.exit(2);
}

function die(msg) {          // an input we cannot work with
  console.error(`error: ${msg}`);
  process.exit(2);
}

const argv = process.argv.slice(2);
const split = argv.includes("--split");
const positional = argv.filter(a => a !== "--split");
if (positional.length !== 3) usage(`expected 3 positional arguments, got ${positional.length}`);
const [voice, outArg, text] = positional;
// resolved against the caller's CWD now, because we chdir to web/ below
const outPath = resolve(process.cwd(), outArg);
if (!voice.length) usage("empty voice");
if (!outArg.length) usage("empty output path");
if (!text.trim().length) usage("empty text");

/* web/index.html splitChunks(), copied verbatim. Kept in sync by eye: it is
 * 20 lines and changes about once a year. */
function splitChunks(text) {
  const MAX = 120, MIN = 40;
  const sents = text.match(/[^.!?।。！？]+[.!?।。！？]*\s*/g) || [text];
  const chunks = [];
  for (let s of sents) {
    s = s.trim();
    if (!s) continue;
    while (s.length > MAX) {
      let cut = -1;
      for (const m of s.matchAll(/[,;:、，；：]\s*/g)) {
        const end = m.index + m[0].length;
        if (end >= MIN && end <= MAX) cut = end;
      }
      if (cut < 0) {
        const sp = s.lastIndexOf(" ", MAX);
        cut = sp > MIN ? sp + 1 : MAX;
      }
      chunks.push(s.slice(0, cut).trim());
      s = s.slice(cut).trim();
    }
    if (s) chunks.push(s);
  }
  return chunks.length ? chunks : [text];
}

function toHeap(mod, bytes) {
  const p = mod._malloc(bytes.length);
  if (!p) throw new Error(`_malloc(${bytes.length}) returned NULL -- raise INITIAL_MEMORY in build_nano.sh`);
  mod.HEAPU8.set(bytes, p);
  return p;
}

/* 32-bit float mono WAV (IEEE float, format tag 3). */
function writeWavF32(path, samples, sampleRate) {
  const hdr = Buffer.alloc(44);
  hdr.write("RIFF", 0); hdr.writeUInt32LE(36 + samples.length * 4, 4); hdr.write("WAVE", 8);
  hdr.write("fmt ", 12); hdr.writeUInt32LE(16, 16); hdr.writeUInt16LE(3, 20);
  hdr.writeUInt16LE(1, 22); hdr.writeUInt32LE(sampleRate, 24);
  hdr.writeUInt32LE(sampleRate * 4, 28); hdr.writeUInt16LE(4, 32); hdr.writeUInt16LE(32, 34);
  hdr.write("data", 36); hdr.writeUInt32LE(samples.length * 4, 40);
  writeFileSync(path, Buffer.concat([hdr, Buffer.from(samples.buffer, samples.byteOffset, samples.length * 4)]));
}

/* ---- frontend: the page's two <script> tags -------------------------- */

// snt_g2p.js resolves snt_g2p.data relative to the process CWD, so run from
// web/ no matter where the caller invoked us.
const frontendPath = resolve(web, "trellis_frontend.js");
const g2pPath = resolve(web, "snt_g2p.js");
for (const p of [frontendPath, g2pPath, resolve(web, "snt_g2p.data")]) {
  if (!existsSync(p)) die(`missing ${p} -- run mcu/ports/wasm/build_g2p.sh`);
}
process.chdir(web);

vm.runInThisContext(readFileSync(frontendPath, "utf8"));
const Frontend = globalThis.SaanoTrellisFrontend;
if (!Frontend) die("trellis_frontend.js did not define SaanoTrellisFrontend");

const G2P = await require(g2pPath)();
if (typeof G2P._snt_g2p_init === "function") {   // the page calls this once at load
  const rc = G2P._snt_g2p_init();
  if (rc !== 0) die(`snt_g2p_init() failed rc=${rc}`);
}
const frontend = Frontend.createFrontend({ module: G2P });

/* ---- the voice: meta.json + blobs + its own module ------------------- */

const dir = resolve(web, "voices", voice);
const metaPath = resolve(dir, "meta.json");
if (!existsSync(metaPath)) die(`missing ${metaPath} -- is "${voice}" a snt_nano voice?`);

let meta;
try {
  meta = JSON.parse(readFileSync(metaPath, "utf8"));
} catch (err) {
  die(`${metaPath} is not valid JSON: ${err.message}`);
}
if (meta.runtime !== "snt_nano") die(`${voice} has runtime "${meta.runtime}", not snt_nano`);
for (const k of ["module", "front", "dec", "front_bytes", "dec_bytes", "weights", "export_name"]) {
  if (meta[k] === undefined) die(`${metaPath} has no "${k}"`);
}

const frontBlob = new Uint8Array(readFileSync(resolve(dir, meta.front)));
const decBlob = new Uint8Array(readFileSync(resolve(dir, meta.dec)));
// the page's own guard: a truncated download must fail loudly, not render mush
if (frontBlob.length !== meta.front_bytes || decBlob.length !== meta.dec_bytes) {
  die(`${voice}: blob sizes ${frontBlob.length}/${decBlob.length} do not match ` +
      `meta.json ${meta.front_bytes}/${meta.dec_bytes}`);
}

const modulePath = resolve(web, meta.module);
if (!existsSync(modulePath)) die(`missing ${modulePath} -- run mcu/ports/wasm/build_nano.sh`);
const N = await (await import(pathToFileURL(modulePath).href)).default();

// NANO_WEIGHT_FORMAT: 0 = int8 rows and int8 activations (the device path),
// 1 = float32 rows. A mismatch here means the module and the blobs disagree
// about how to read every weight, which is silent garbage, not an error.
const fmt = N._snt_nano_wasm_weight_format();
if ((fmt === 1) !== (meta.weights === "f32")) {
  die(`${voice}: module weight format ${fmt} does not match meta.json "${meta.weights}"`);
}
const sr = N._snt_nano_wasm_sample_rate() || meta.sample_rate;
if (!sr) die(`${voice}: module reported no sample rate and meta.json has none`);

console.log(`voice   : ${voice} (${meta.export_name}, lineage ${meta.lineage})`);
console.log(`module  : ${modulePath}`);
console.log(`weights : ${meta.weights} (NANO_WEIGHT_FORMAT ${fmt}: ${fmt === 1 ? "float32 rows" : "int8 rows + int8 activations, the device path"})`);
console.log(`blobs   : ${meta.front} ${frontBlob.length} B + ${meta.dec} ${decBlob.length} B = ${frontBlob.length + decBlob.length} B`);
console.log(`rate    : ${sr} Hz`);
console.log(`text    : ${JSON.stringify(text)}`);

/* ---- synthesize ------------------------------------------------------ */

const chunks = split ? splitChunks(text) : [text];
console.log(`chunks  : ${chunks.length}${split ? " (--split: the page's splitChunks)" : " (single pass, the gallery convention)"}`);

const frontP = toHeap(N, frontBlob);
const decP = toHeap(N, decBlob);
const arenaP = N._malloc(NANO_ARENA);
const outP = N._malloc(NANO_OUT_CAP * 4);
if (!arenaP || !outP) die("could not allocate the arena / output buffer inside the module");

const pcms = [];
try {
  for (let i = 0; i < chunks.length; i++) {
    const chunk = chunks[i];
    let ids, phonemes;
    try {
      ({ ids, phonemes } = frontend.textToIds(chunk));
    } catch (err) {
      // FrontendError carries .kind (wasm / empty / limit / type)
      throw new Error(`frontend rejected chunk ${i + 1} ${JSON.stringify(chunk)}: ` +
                      `${err.kind ? err.kind + ": " : ""}${err.message}`);
    }
    if (!ids.length) throw new Error(`frontend produced no ids for chunk ${i + 1} ${JSON.stringify(chunk)}`);
    const ids32 = Int32Array.from(ids);

    // decoder-noise seed: sha256(chunk)[:8], the renderer's own convention,
    // derived inside the module so the C SHA-256 is the one that decides
    const nbytes = N.lengthBytesUTF8(chunk) + 1;
    const textP = N._malloc(nbytes);
    const seedP = N._malloc(8);
    const idsP = toHeap(N, new Uint8Array(ids32.buffer, 0, ids32.length * 4));
    let n, lo, hi, wall;
    try {
      if (!textP || !seedP) throw new Error("out of module memory deriving the seed");
      N.stringToUTF8(chunk, textP, nbytes);
      const src = N._snt_nano_wasm_seed_from_text(textP, seedP);
      if (src !== 0) throw new Error(`snt_nano_wasm_seed_from_text failed rc=${src}`);
      lo = N.HEAPU32[seedP >> 2];
      hi = N.HEAPU32[(seedP >> 2) + 1];
      const t0 = performance.now();
      // durs = 0 (NULL) -> the duration student runs, exactly as the page does
      n = N._snt_nano_wasm_synthesize(frontP, decP, idsP, ids32.length, 0, lo, hi,
                                      arenaP, NANO_ARENA, outP, NANO_OUT_CAP);
      wall = (performance.now() - t0) / 1000;
    } finally {
      [textP, seedP, idsP].forEach(p => { if (p) N._free(p); });
    }
    if (n < 0) {
      const why = { "-1": "bad arguments", "-2": "arena too small", "-3": "token count outside 1..1024",
                    "-4": "phoneme id outside the vocabulary", "-5": "output cap too small",
                    "-6": "snt_nano_synthesize failed" }[String(n)] || "unknown";
      throw new Error(`chunk ${i + 1} failed: rc=${n} (${why}), core rc ${N._snt_nano_wasm_last_rc()}`);
    }
    const pcm = new Float32Array(n);
    pcm.set(N.HEAPF32.subarray(outP >> 2, (outP >> 2) + n));
    pcms.push(pcm);
    console.log(`chunk ${i + 1}/${chunks.length} ${JSON.stringify(chunk)}`);
    console.log(`   phonemes ${JSON.stringify(phonemes)}`);
    console.log(`   ids ${ids32.length}  seed ${(BigInt(hi) << 32n) | BigInt(lo)}  ` +
                `frames ${N._snt_nano_wasm_last_frames()}  samples ${n}  ` +
                `${(n / sr).toFixed(3)} s in ${wall.toFixed(3)} s (${(n / sr / wall).toFixed(1)}x RT)  ` +
                `arena peak ${N._snt_nano_wasm_last_arena_peak()} B`);
  }
} catch (err) {
  console.error(`error: ${err.message}`);
  process.exit(1);
} finally {
  [frontP, decP, arenaP, outP].forEach(p => { if (p) N._free(p); });
}

const total = pcms.reduce((a, p) => a + p.length, 0);
if (!total) { console.error("error: rendered zero samples"); process.exit(1); }
const all = new Float32Array(total);
{ let off = 0; for (const p of pcms) { all.set(p, off); off += p.length; } }
let peak = 0;
for (let i = 0; i < all.length; i++) { const a = Math.abs(all[i]); if (a > peak) peak = a; }
if (!Number.isFinite(peak)) { console.error("error: waveform contains NaN or Inf"); process.exit(1); }

try {
  writeWavF32(outPath, all, sr);
} catch (err) {
  console.error(`error: could not write ${outPath}: ${err.message}`);
  process.exit(1);
}
console.log(`total   : ${total} samples, ${(total / sr).toFixed(3)} s, float peak ${peak.toFixed(6)}`);
console.log(`wrote   : ${outPath}`);
