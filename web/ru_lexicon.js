// ru_lexicon.js -- assemble espeak's Russian dictionary per utterance.
//
// espeak-ng's ru_dict is 8.2MB raw / 3.5MB brotli, and 99.5% of it is an
// 811k-entry stress lexicon (dictsource/extra/ru_listx). Russian stress is
// lexical, not derivable: with no lexicon at all 1 phoneme id in 12 changes,
// and trimming it by word frequency is no better -- keeping the 500k most
// frequent forms still costs 625KB brotli and changes 1 id in 681. Those ids
// are the training-time contract for ru_RU-irina-medium, so "close" is not
// usable. See experiments/evidence/ru-dict-trim-20260908.json.
//
// Any single utterance, though, needs only a sliver. So: ship the lexicon
// SOURCE split into shards, fetch the handful a sentence touches, and let
// espeak's own compiler -- espeak_ng_CompileDictionary, already linked into
// snt_g2p.wasm because build_g2p.sh has always compiled compiledict.c -- build
// a dictionary in the module's own filesystem. The phoneme ids come out
// identical to the full 8.6MB dictionary, not merely close.
//
// Everything here is static files: no server logic, no range requests.
//
//   import { RuLexicon } from "./ru_lexicon.js";
//   const ru = new RuLexicon({ baseUrl: "g2p-lazy/ru" });
//   await ru.init(Mod);              // manifest + rules/list/emoji + base tier
//   await ru.prepare(Mod, text);     // fetch missing shards, compile
//   // ... then snt_g2p_set_voice("ru", 10) and snt_g2p_text_to_ids as usual
//
// Node callers pass their own `fetchText` (fs) instead of window.fetch.

const DSRC = "/dsrc/";               // where the dictionary source is staged
const SRC_FILES = ["ru_rules", "ru_list", "ru_emoji"];

/** FNV-1a 32-bit over UTF-8 bytes. Must match tools/gen_ru_lexicon_shards.py. */
export function fnv1a32(str) {
  const bytes = new TextEncoder().encode(str);
  let h = 0x811c9dc5;
  for (let i = 0; i < bytes.length; i++) {
    h ^= bytes[i];
    h = Math.imul(h, 0x01000193) >>> 0;
  }
  return h >>> 0;
}

/**
 * Headwords espeak might look up for `text`.
 *
 * Deliberately a SUPERSET. A word whose entry we fail to fetch does not error
 * -- it silently falls back to the letter rules and comes out with the wrong
 * stress, which is exactly the failure this whole mechanism exists to avoid.
 * An extra candidate costs a few hundred bytes; a missing one costs a
 * mispronounced word.
 *
 * espeak splits words at case boundaries (translate.c: "lower case followed by
 * upper case, possibly CamelCase", plus the upper->lower split at the last
 * uppercase). A run like "туризмЭто" -- a missing space, which real text does
 * contain -- is ONE match for a naive /[а-яё]+/ but TWO lookups for espeak, and
 * getting that wrong was the single failure in the first end-to-end run.
 */
export function candidates(text) {
  const out = new Set();
  for (const run of text.match(/[а-яёА-ЯЁ]+/g) || []) {
    out.add(run.toLowerCase());
    for (const piece of run.split(/(?<=[а-яё])(?=[А-ЯЁ])/)) {
      out.add(piece.toLowerCase());
      for (const p2 of piece.split(/(?<=[А-ЯЁ])(?=[А-ЯЁ][а-яё])/)) out.add(p2.toLowerCase());
    }
  }
  out.delete("");
  return [...out];
}

export class RuLexicon {
  /**
   * @param {object} opts
   * @param {string} opts.baseUrl        directory holding manifest.json
   * @param {function} [opts.fetchText]  (path) => Promise<string>; defaults to fetch()
   */
  constructor({ baseUrl, fetchText } = {}) {
    if (!baseUrl) throw new Error("RuLexicon: baseUrl is required");
    this.baseUrl = baseUrl.replace(/\/+$/, "");
    this.fetchText = fetchText || (async (p) => {
      const r = await fetch(`${this.baseUrl}/${p}`);
      if (!r.ok) throw new Error(`RuLexicon: ${p} -> HTTP ${r.status}`);
      return r.text();
    });
    this.manifest = null;
    this.entries = new Map();   // headword -> source lines
    this.known = new Set();     // headwords we have a definitive answer for
    this.shardsLoaded = new Set();
    this.stats = { shardRequests: 0, shardBytes: 0, baseBytes: 0, srcBytes: 0, compiles: 0, compileMs: 0 };
  }

  /** Fetch the manifest, the always-compiled sources and the base tier, and
   *  stage the sources in the module filesystem. Idempotent. */
  async init(Mod) {
    if (this.manifest) return;
    const manifest = JSON.parse(await this.fetchText("manifest.json"));
    if (manifest.hash !== "fnv1a-32-mod-shards")
      throw new Error(`RuLexicon: unsupported shard hash ${manifest.hash}`);
    this.manifest = manifest;

    Mod.FS.mkdirTree(DSRC.slice(0, -1));
    // The manifest is the authority on which sources get compiled; SRC_FILES is
    // only a fallback, so regenerating with a different set cannot silently
    // leave the client compiling the wrong thing.
    for (const name of (manifest.src_files || SRC_FILES)) {
      const text = await this.fetchText(`src/${name}`);
      this.stats.srcBytes += text.length;
      Mod.FS.writeFile(DSRC + name, text);
    }

    const base = await this.fetchText("base.txt");
    this.stats.baseBytes = base.length;
    let absent = false;
    for (const line of base.split("\n")) {
      if (line === manifest.absent_marker) { absent = true; continue; }
      const s = line.trim();
      if (!s) continue;
      if (absent) { this.known.add(s); continue; }   // frequent, and has NO entry
      this.#add(line);
    }
  }

  #add(line) {
    const s = line.trim();
    if (!s || s.startsWith("//")) return;
    const w = s.split(/\s+/)[0].toLowerCase();
    if (!this.entries.has(w)) this.entries.set(w, []);
    this.entries.get(w).push(line);
    this.known.add(w);
  }

  /** Fetch whatever shards `text` needs, then compile a dictionary for it.
   *  Returns the compile stats for this utterance. */
  async prepare(Mod, text) {
    if (!this.manifest) throw new Error("RuLexicon: call init() first");
    const words = candidates(text);

    const wanted = new Set();
    for (const w of words) {
      if (this.known.has(w)) continue;                  // base tier already settled it
      const sid = fnv1a32(w) % this.manifest.shards;
      if (!this.shardsLoaded.has(sid)) wanted.add(sid);
    }
    await Promise.all([...wanted].map(async (sid) => {
      const path = this.manifest.shard_path.replace("{id:04x}", sid.toString(16).padStart(4, "0"));
      const text = await this.fetchText(path);
      this.stats.shardRequests++;
      this.stats.shardBytes += text.length;
      this.shardsLoaded.add(sid);
      for (const line of text.split("\n")) this.#add(line);
    }));
    // Anything still unmatched genuinely has no entry: remember that so the
    // same word never triggers a second request, and let it fall to the rules.
    for (const w of words) this.known.add(w);

    const lines = [];
    for (const w of words) {
      const e = this.entries.get(w);
      if (e) lines.push(...e);
    }
    Mod.FS.writeFile(DSRC + "ru_listx", lines.join("\n") + "\n");

    const compile = this._compile || (this._compile = Mod.cwrap(
      "espeak_ng_CompileDictionary", "number", ["string", "string", "number", "number", "number"]));
    const t0 = (globalThis.performance || Date).now();
    const rc = compile(DSRC, "ru", 0, 0, 0);
    const ms = (globalThis.performance || Date).now() - t0;
    if (rc !== 0) throw new Error(`RuLexicon: espeak_ng_CompileDictionary rc=${rc}`);
    this.stats.compiles++;
    this.stats.compileMs += ms;
    return { entries: lines.length, shardsFetched: wanted.size, compileMs: ms };
  }
}
