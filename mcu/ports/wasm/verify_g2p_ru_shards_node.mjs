// verify_g2p_ru_shards_node.mjs -- the gate for per-utterance Russian.
//
// Russian ships no dictionary. web/ru_lexicon.js fetches a handful of lexicon
// shards for the words in the text and compiles a dictionary in-module with
// espeak's own compiler. The phoneme ids MUST come out identical to the full
// 8.6MB dictionary, because ru_RU-irina-medium was distilled on those exact
// ids -- 99.9% is a fail, not a pass.
//
// The reference is built from the shipped shards themselves: concatenating all
// of web/g2p-lazy/ru/shard/ reproduces the complete ru_listx, so this gate has
// no external dependency beyond the corpus. (That full dictionary's own
// equivalence to python PiperVoice is established in
// experiments/evidence/ru-dict-trim-20260908.json: 0 changed ids of 637,730.)
//
//   node mcu/ports/wasm/verify_g2p_ru_shards_node.mjs [corpus.jsonl]
//
// Exit 0 only when agreement is exactly 100.000%.
import { readFileSync, existsSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";
import { RuLexicon, candidates } from "../../../web/ru_lexicon.js";

const here = dirname(fileURLToPath(import.meta.url));
const repo = resolve(here, "../../..");
const web = resolve(repo, "web");
const ruDir = resolve(web, "g2p-lazy/ru");

const RU_SLOT = 10, MAX_IDS = 8192;
const corpusPath = process.argv[2] || resolve(repo, "data/textsets/flores-distill-v1/ru_RU.jsonl");
if (!existsSync(corpusPath)) {
  console.error(`missing corpus ${corpusPath}\n` +
    `pass one as argv[1]; the reference set is FLORES Russian ` +
    `(k2:~/saanotts/data/textsets/flores-distill-v1/ru_RU.jsonl)`);
  process.exit(1);
}
const texts = readFileSync(corpusPath, "utf8").trim().split("\n").map((l) => JSON.parse(l).text);
const manifest = JSON.parse(readFileSync(resolve(ruDir, "manifest.json"), "utf8"));

const factory = (await import(resolve(web, "snt_g2p.js"))).default;
async function boot() {
  const Mod = await factory({ locateFile: (f) => resolve(web, f), print: () => {}, printErr: () => {} });
  const setVoice = Mod.cwrap("snt_g2p_set_voice", "number", ["string", "number"]);
  if (setVoice("en-us", 1) !== 0) throw new Error("en-us warmup failed");
  if (setVoice("ru", RU_SLOT) !== 0) throw new Error("ru voice failed -- is lang/zle/ru in espeak-data-multi?");
  const idsFn = Mod.cwrap("snt_g2p_text_to_ids", "number", ["number", "number", "number"]);
  const outP = Mod._malloc(MAX_IDS * 4);
  const ids = (t) => {
    const nb = Mod.lengthBytesUTF8(t) + 1, p = Mod._malloc(nb);
    Mod.stringToUTF8(t, p, nb);
    const n = idsFn(p, outP, MAX_IDS); Mod._free(p);
    if (n < 0) throw new Error(`snt_g2p_text_to_ids rc=${n}`);
    return Array.from(Mod.HEAP32.subarray(outP >> 2, (outP >> 2) + n));
  };
  return { Mod, ids };
}

// ---- reference: the complete lexicon, compiled from the shipped shards ------
const refRun = await boot();
{
  const compile = refRun.Mod.cwrap("espeak_ng_CompileDictionary", "number",
    ["string", "string", "number", "number", "number"]);
  refRun.Mod.FS.mkdirTree("/dsrc");
  for (const name of manifest.src_files)
    refRun.Mod.FS.writeFile(`/dsrc/${name}`, readFileSync(resolve(ruDir, "src", name)));
  const all = [];
  for (let i = 0; i < manifest.shards; i++)
    all.push(readFileSync(resolve(ruDir, "shard", `${i.toString(16).padStart(4, "0")}.txt`), "utf8"));
  refRun.Mod.FS.writeFile("/dsrc/ru_listx", all.join(""));
  const t0 = performance.now();
  const rc = compile("/dsrc/", "ru", 0, 0, 0);
  if (rc !== 0) throw new Error(`reference compile rc=${rc}`);
  console.log(`reference: full lexicon from ${manifest.shards} shards, ` +
    `${refRun.Mod.FS.stat("/espeak/ru_dict").size} byte dict, ${(performance.now() - t0).toFixed(0)} ms`);
}
const ref = texts.map(refRun.ids);
const totalIds = ref.reduce((n, a) => n + a.length, 0);

// ---- the real path: per-utterance shard fetch + compile ---------------------
const run = await boot();
const lex = new RuLexicon({
  baseUrl: ruDir,
  fetchText: async (p) => readFileSync(resolve(ruDir, p), "utf8"),
});
await lex.init(run.Mod);
console.log(`base tier: ${lex.entries.size} headwords with entries, ${lex.known.size} settled without a request`);

let same = 0, firstBad = null, maxCompile = 0;
const perUtterance = [];
for (let i = 0; i < texts.length; i++) {
  const st = await lex.prepare(run.Mod, texts[i]);
  maxCompile = Math.max(maxCompile, st.compileMs);
  perUtterance.push(st.shardsFetched);
  const got = run.ids(texts[i]);
  if (JSON.stringify(got) === JSON.stringify(ref[i])) same++;
  else if (!firstBad) firstBad = { i, text: texts[i], got, want: ref[i] };
}

const pct = (same * 100) / texts.length;
console.log(`\nsentences identical to the full dictionary: ${same}/${texts.length} = ${pct.toFixed(3)}%`);
console.log(`ids compared: ${totalIds}`);
console.log(`shard requests: ${lex.stats.shardRequests} total, ` +
  `${(perUtterance.reduce((a, b) => a + b, 0) / texts.length).toFixed(2)} per utterance (cache warms)`);
console.log(`bytes fetched: base ${lex.stats.baseBytes}, sources ${lex.stats.srcBytes}, ` +
  `shards ${lex.stats.shardBytes}`);
console.log(`compile: ${(lex.stats.compileMs / lex.stats.compiles).toFixed(2)} ms mean, ` +
  `${maxCompile.toFixed(2)} ms max, ${lex.stats.compiles} compiles`);
if (firstBad) {
  console.log(`\nfirst mismatch, row ${firstBad.i}: ${firstBad.text}`);
  console.log(`  got : [${firstBad.got.join(", ")}]`);
  console.log(`  want: [${firstBad.want.join(", ")}]`);
}
const ok = same === texts.length;
console.log(ok ? "PASS" : "FAIL");
process.exit(ok ? 0 : 1);
