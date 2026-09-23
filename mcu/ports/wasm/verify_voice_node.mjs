// verify_voice_node.mjs -- end-to-end gate for the all-voices live browser
// synth chain, run headless under Node.
//
// For each voice under test: loads web/voices/<key>/meta.json, drives
// web/snt_g2p.js (SaanoG2P) with that voice's espeak_voice + g2p_voice_slot
// to turn a sentence into phoneme ids, feeds those ids plus
// web/voices/<key>/{front_f32.bin,dec_f32.bin} into web/snt_voice.js
// (SaanoVoice)'s snt_voice_synthesize, and asserts:
//   - synth returns a positive sample count
//   - the resulting audio is longer than 1 second at 22.05kHz
//   - every sample is finite (no NaN/Inf from a mis-parsed blob or a
//     shape mismatch that silently walked off a weight tensor)
//   - the audio is not silence (some samples exceed a tiny amplitude floor)
//
// A voice whose meta.json says "weights": "f16" is widened here the way
// web/index.html widens it after download, using the SAME routine, and the
// widened bytes are checked against meta.json's *_widened_sha256. That digest
// is written by tools/shrink_voice_bundle_f16.py from numpy, so agreement
// gates the browser's hand-rolled half-float decode against numpy's -- the
// step that stands between a 3 MB download and the f32 blob the runtime has
// always been handed. Before this, the harness opened front_f32.bin by name
// and so could not run against any f16 voice at all, which is the ten
// languages added on 2026-09-08 plus Polish.
//
//   node mcu/ports/wasm/verify_voice_node.mjs             # amy + hindi
//   node mcu/ports/wasm/verify_voice_node.mjs amy kristin hfc ...  # subset
//
// Exit 0 when all requested voices PASS, 1 otherwise.
import { readFileSync } from "node:fs";
import { createHash } from "node:crypto";
import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";

const here = dirname(fileURLToPath(import.meta.url));
const repo = resolve(here, "../../..");
const web = resolve(repo, "web");

const SR = 22050;
const MAX_IDS = 1024;
const OUT_CAP = 22050 * 20; // 20s ceiling, generous for a short test sentence

const SENTENCES = {
  amy: "Hello! I am a tiny voice living entirely in your browser.",
  kristin: "Hello! I am a tiny voice living entirely in your browser.",
  hfc: "Hello! I am a tiny voice living entirely in your browser.",
  vietnamese: "Xin chào! Rất vui được gặp bạn hôm nay.",
  indonesian: "Halo! Senang bertemu dengan Anda hari ini.",
  nepali: "नमस्ते! तपाईंलाई भेटेर खुशी लाग्यो।",
  hindi: "नमस्ते! आपसे मिलकर बहुत खुशी हुई।",
  chinese: "你好！很高兴今天见到你。",
  // The ten languages added 2026-09-08 and Polish. All ship f16, so none of
  // them could be driven from here until this harness learned to widen.
  // Sentences are the ones web/index.html puts in the box for that voice, so
  // a failure here is a failure a visitor would hear on the first click.
  german: "Hallo! Ich bin eine winzige Stimme in deinem Browser.",
  french: "Bonjour ! Je suis une toute petite voix dans votre navigateur.",
  spanish: "\u00a1Hola! Soy una voz diminuta que vive en tu navegador.",
  italian: "Ciao! Sono una vocina che vive nel tuo browser.",
  portuguese: "Ol\u00e1! Sou uma vozinha que mora no seu navegador.",
  // Russian is deliberately absent, for the same reason it is absent from
  // verify_g2p_node.mjs: ru_dict is not in the .data bundle (8.2MB raw), so
  // espeak here answers "Can't read dictionary file: /espeak/ru_dict" and
  // emits 5 ids. The browser assembles it per utterance from web/g2p-lazy/ru/
  // and compiles it in-module; verify_g2p_ru_shards_node.mjs gates THAT path.
  czech: "Ahoj! Jsem mal\u00fd hlas ve tv\u00e9m prohl\u00ed\u017ee\u010di.",
  romanian: "Salut! Sunt o voce mic\u0103 ce tr\u0103ie\u0219te \u00een browserul t\u0103u.",
  turkish: "Merhaba! Taray\u0131c\u0131n\u0131zda ya\u015fayan k\u00fc\u00e7\u00fck bir sesim.",
  arabic: "\u0645\u0631\u062d\u0628\u0627! \u0623\u0646\u0627 \u0635\u0648\u062a \u0635\u063a\u064a\u0631 \u064a\u0639\u064a\u0634 \u0641\u064a \u0645\u062a\u0635\u0641\u062d\u0643.",
  polish: "Cze\u015b\u0107! Jestem ma\u0142ym g\u0142osem \u017cyj\u0105cym w twojej przegl\u0105darce.",
};

const args = process.argv.slice(2);
const voices = args.length ? args : ["amy", "hindi"];

const SaanoVoice = (await import(resolve(web, "snt_voice.js"))).default;
const SaanoG2P = (await import(resolve(web, "snt_g2p.js"))).default;

const V = await SaanoVoice();
const G = await SaanoG2P({ locateFile: (f) => resolve(web, f) });

const setVoice = G.cwrap("snt_g2p_set_voice", "number", ["string", "number"]);
const g2pIds = G.cwrap("snt_g2p_text_to_ids", "number", ["number", "number", "number"]);
const synth = V.cwrap("snt_voice_synthesize", "number",
  ["number", "number", "number", "number", "number", "number", "number"]);

// Byte-for-byte the routine in web/index.html; see the note above on why this
// is duplicated rather than reimplemented with Node's own Float16Array.
function widenF16(bytes, dims) {
  const head = Number(dims.meta_bytes), n = Number(dims.weight_floats);
  if (!Number.isInteger(head) || !Number.isInteger(n) || head < 0 || n <= 0)
    throw new Error("f16 blob: meta.json has no usable meta_bytes/weight_floats");
  if (bytes.length !== head + 2 * n)
    throw new Error(`f16 blob is ${bytes.length} bytes, meta.json implies ${head + 2 * n}`);
  const out = new Uint8Array(head + 4 * n);
  out.set(bytes.subarray(0, head), 0);
  const src = new DataView(bytes.buffer, bytes.byteOffset + head, 2 * n);
  const dst = new DataView(out.buffer, head, 4 * n);
  for (let i = 0; i < n; i++) {
    const b = src.getUint16(i * 2, true);
    const exp = (b >> 10) & 0x1f, frac = b & 0x3ff, sign = (b & 0x8000) ? -1 : 1;
    let v;
    if (exp === 0)       v = sign * frac * 5.960464477539063e-8;
    else if (exp === 31) v = frac ? NaN : sign * Infinity;
    else                 v = sign * Math.pow(2, exp - 25) * (1024 + frac);
    dst.setFloat32(i * 4, v, true);
  }
  return out;
}

// Returns the f32 bytes the runtime must be handed, widening first when the
// voice ships f16. `expectSha` is meta.json's digest OF THE WIDENED BYTES.
function voiceWeights(dir, name, dims, f16, expectSha) {
  let bytes = new Uint8Array(readFileSync(resolve(dir, name)));
  if (f16) bytes = widenF16(bytes, dims);
  if (expectSha) {
    const got = createHash("sha256").update(bytes).digest("hex");
    if (got !== expectSha)
      throw new Error(`${name}: widened sha256 ${got} != meta.json ${expectSha}`);
  }
  return bytes;
}

function loadBlobBytes(mod, bytes) {
  const p = mod._malloc(bytes.length);
  mod.HEAPU8.set(bytes, p);
  return { p, len: bytes.length };
}

function loadBlob(mod, path) {
  const bytes = new Uint8Array(readFileSync(path));
  const p = mod._malloc(bytes.length);
  mod.HEAPU8.set(bytes, p);
  return { p, len: bytes.length };
}

function textToIds(text) {
  const nBytes = G.lengthBytesUTF8(text) + 1;
  const textP = G._malloc(nBytes);
  G.stringToUTF8(text, textP, nBytes);
  const outP = G._malloc(MAX_IDS * 4);
  const n = g2pIds(textP, outP, MAX_IDS);
  G._free(textP);
  if (n <= 0) { G._free(outP); throw new Error(`g2p failed: rc=${n}`); }
  const ids = Int32Array.from(G.HEAP32.subarray(outP >> 2, (outP >> 2) + n));
  G._free(outP);
  return ids;
}

let allOk = true;
for (const key of voices) {
  const dir = resolve(web, "voices", key);
  const meta = JSON.parse(readFileSync(resolve(dir, "meta.json"), "utf8"));
  const text = SENTENCES[key];
  if (!text) throw new Error(`no test sentence for voice ${key}`);

  const rc = setVoice(meta.espeak_voice, meta.g2p_voice_slot);
  if (rc !== 0) {
    console.log(`${key}: FAIL (snt_g2p_set_voice("${meta.espeak_voice}", ${meta.g2p_voice_slot}) rc=${rc})`);
    allOk = false;
    continue;
  }
  const ids = textToIds(text);

  const f16 = meta.weights === "f16";
  const front = loadBlobBytes(V, voiceWeights(
    dir, meta.front || "front_f32.bin", meta.front_dims, f16, meta.front_widened_sha256));
  const dec = loadBlobBytes(V, voiceWeights(
    dir, meta.dec || "dec_f32.bin", meta.dec_dims, f16, meta.dec_widened_sha256));
  const idsBytes = new Uint8Array(ids.buffer, ids.byteOffset, ids.byteLength);
  const idsBlob = { p: V._malloc(idsBytes.length), len: idsBytes.length };
  V.HEAPU8.set(idsBytes, idsBlob.p);
  const outP = V._malloc(OUT_CAP * 4);

  const t0 = performance.now();
  const n = synth(front.p, dec.p, idsBlob.p, ids.length, meta.length_scale, outP, OUT_CAP);
  const wall = (performance.now() - t0) / 1000;

  let ok = n > 0;
  let secs = 0, finite = true, peak = 0;
  if (ok) {
    secs = n / SR;
    const pcm = V.HEAPF32.subarray(outP >> 2, (outP >> 2) + n);
    for (let i = 0; i < n; i++) {
      const v = pcm[i];
      if (!Number.isFinite(v)) { finite = false; break; }
      const a = Math.abs(v);
      if (a > peak) peak = a;
    }
    ok = finite && secs > 1.0 && peak > 1e-4;
  }

  [front.p, dec.p, idsBlob.p, outP].forEach((p) => V._free(p));

  console.log(
    `${key}: ${ok ? "PASS" : "FAIL"}  n=${n} ids=${ids.length} ` +
    `secs=${secs.toFixed(2)} finite=${finite} peak=${peak.toFixed(4)} ` +
    `wall=${wall.toFixed(3)}s${wall > 0 ? ` (${(secs / wall).toFixed(1)}x RT)` : ""}`
  );
  if (!ok) allOk = false;
}

console.log(allOk ? "ALL PASS" : "FAIL");
process.exit(allOk ? 0 : 1);
