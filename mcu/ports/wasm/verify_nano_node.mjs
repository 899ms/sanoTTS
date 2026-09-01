// verify_nano_node.mjs -- the WASM golden gate for a snt_nano lineage, run
// headless under Node.
//
// Loads web/snt_nano_<voice>.js (built by build_nano.sh against
// mcu/models/<lineage>/) and the versioned fixture under
// mcu/test/fixtures/<lineage>/, then runs every fixture row through
// snt_nano_wasm_synthesize with the row's FROZEN durations and seed -- the
// exact procedure of mcu/test/nano_golden_main.c -- and reports the waveform
// Pearson correlation against the float PyTorch reference (rNN_audio.bin).
//
// The gate is the MINIMUM correlation over the rows, held to the same 0.98
// the host gate uses. An int8 module should print the same per-row numbers
// as `make test-nano`; a float-weight module (build_nano.sh ... f32) the same
// as `make test-nano-wf32`. Any difference between the two harnesses is a
// finding, not noise: same C, same inputs.
//
//   bash mcu/ports/wasm/build_nano.sh en_us_e13b heartnano SaanoNanoHeartNano int8
//   node mcu/ports/wasm/verify_nano_node.mjs heartnano en_us_e13b
//   node mcu/ports/wasm/verify_nano_node.mjs heart en_us_r227f32
//
// Exit 0 on PASS, 1 otherwise.
import { readFileSync, existsSync } from "node:fs";
import { fileURLToPath, pathToFileURL } from "node:url";
import { dirname, resolve } from "node:path";

const here = dirname(fileURLToPath(import.meta.url));
const repo = resolve(here, "../../..");
const web = resolve(repo, "web");

const GATE = 0.98;
const ARENA = 16 * 1024 * 1024;
const OUT_CAP = 24000 * 30;

function usage() {
  console.error("usage: node verify_nano_node.mjs <voice> <lineage>");
  process.exit(2);
}
const [voice, lineage] = process.argv.slice(2);
if (!voice || !lineage) usage();

const fixture = resolve(repo, "mcu/test/fixtures", lineage);
const modulePath = resolve(web, `snt_nano_${voice}.js`);
if (!existsSync(modulePath)) { console.error(`missing ${modulePath} -- run build_nano.sh`); process.exit(2); }
if (!existsSync(resolve(fixture, "rows.txt"))) { console.error(`missing fixture ${fixture}`); process.exit(2); }

function loadBytes(name) { return new Uint8Array(readFileSync(resolve(fixture, name))); }
function loadI32(name) { const b = readFileSync(resolve(fixture, name)); return new Int32Array(b.buffer, b.byteOffset, b.byteLength >> 2); }
function loadF32(name) { const b = readFileSync(resolve(fixture, name)); return new Float32Array(b.buffer, b.byteOffset, b.byteLength >> 2); }

function pearson(a, b) {
  const n = Math.min(a.length, b.length);
  let sa = 0, sb = 0, saa = 0, sbb = 0, sab = 0;
  for (let i = 0; i < n; i++) {
    const x = a[i], y = b[i];
    sa += x; sb += y; saa += x * x; sbb += y * y; sab += x * y;
  }
  const cov = sab - sa * sb / n, va = saa - sa * sa / n, vb = sbb - sb * sb / n;
  return { corr: cov / Math.sqrt(va * vb), rmsRatio: Math.sqrt(saa / n) / Math.sqrt(sbb / n), n };
}

const factory = (await import(pathToFileURL(modulePath).href)).default;
const M = await factory();

const fmt = M._snt_nano_wasm_weight_format();
const tag = fmt === 1 ? "f32" : "q8";
const frontName = `front_${tag}.bin`, decName = `model_${tag}.bin`;
if (!existsSync(resolve(fixture, frontName))) {
  console.error(`module is a ${tag} build but ${fixture} has no ${frontName}`);
  process.exit(2);
}
const front = loadBytes(frontName), dec = loadBytes(decName);

const rows = readFileSync(resolve(fixture, "rows.txt"), "utf8").trim().split("\n").map(l => {
  const [rowId, tokens, frames, samples, seed] = l.trim().split(/\s+/);
  return { rowId, tokens: +tokens, frames: +frames, samples: +samples, seed: BigInt(seed) };
});

console.log(`module  : ${modulePath} (${tag} weights, ${M._snt_nano_wasm_sample_rate()} Hz)`);
console.log(`fixture : ${fixture} (${rows.length} rows)`);

// seed derivation must match the fixture's recorded sha256(row_id)[:8]
{
  const text = rows[0].rowId;
  const n = M.lengthBytesUTF8(text) + 1;
  const tp = M._malloc(n); M.stringToUTF8(text, tp, n);
  const sp = M._malloc(8);
  const rc = M._snt_nano_wasm_seed_from_text(tp, sp);
  const lo = BigInt(M.HEAPU32[sp >> 2]), hi = BigInt(M.HEAPU32[(sp >> 2) + 1]);
  const derived = (hi << 32n) | lo;
  M._free(tp); M._free(sp);
  const ok = rc === 0 && derived === rows[0].seed;
  console.log(`seed    : sha256("${text}")[:8] = ${derived}, fixture says ${rows[0].seed} ${ok ? "OK" : "-- SHA-256 PORT IS WRONG"}`);
  if (!ok) process.exit(1);
}

const frontP = M._malloc(front.length); M.HEAPU8.set(front, frontP);
const decP = M._malloc(dec.length); M.HEAPU8.set(dec, decP);
const arenaP = M._malloc(ARENA);
const outP = M._malloc(OUT_CAP * 4);

console.log("\nrow            frames  samples       corr  rms_ratio        ms");
let minCorr = 2, sum = 0, worst = null, hardFail = false;
for (let r = 0; r < rows.length; r++) {
  const tag2 = String(r).padStart(2, "0");
  const ids = loadI32(`r${tag2}_ids.bin`), durs = loadI32(`r${tag2}_durs.bin`), gold = loadF32(`r${tag2}_audio.bin`);
  const idsP = M._malloc(ids.length * 4); M.HEAP32.set(ids, idsP >> 2);
  const dursP = M._malloc(durs.length * 4); M.HEAP32.set(durs, dursP >> 2);
  const lo = Number(rows[r].seed & 0xffffffffn), hi = Number(rows[r].seed >> 32n);
  const t0 = performance.now();
  const n = M._snt_nano_wasm_synthesize(frontP, decP, idsP, ids.length, dursP, lo, hi, arenaP, ARENA, outP, OUT_CAP);
  const ms = performance.now() - t0;
  M._free(idsP); M._free(dursP);
  if (n < 0) {
    console.log(`${rows[r].rowId.padEnd(14)} FAILED rc=${n} (core rc ${M._snt_nano_wasm_last_rc()})`);
    hardFail = true; continue;
  }
  const pcm = M.HEAPF32.subarray(outP >> 2, (outP >> 2) + n);
  if (n !== gold.length) console.log(`  note: ${n} samples emitted vs ${gold.length} in the reference`);
  const { corr, rmsRatio } = pearson(pcm, gold);
  console.log(`${rows[r].rowId.padEnd(14)} ${String(M._snt_nano_wasm_last_frames()).padStart(6)} ${String(n).padStart(8)}   ${corr.toFixed(6)}   ${rmsRatio.toFixed(6)} ${ms.toFixed(0).padStart(9)}`);
  sum += corr;
  if (corr < minCorr) { minCorr = corr; worst = rows[r].rowId; }
}
[frontP, decP, arenaP, outP].forEach(p => M._free(p));

console.log(`\nrows ${rows.length}   mean corr ${(sum / rows.length).toFixed(6)}   MIN corr ${minCorr.toFixed(6)} (${worst})`);
if (hardFail) { console.log("FAIL: a row did not synthesize"); process.exit(1); }
if (minCorr > GATE) { console.log("PASS"); process.exit(0); }
console.log(`FAIL: corr ${minCorr.toFixed(4)} < ${GATE}`);
process.exit(1);
