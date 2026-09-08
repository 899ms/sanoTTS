/**
 * Gate the browser tashkeel port against piper's own onnxruntime output.
 *
 * Ground truth is produced on k2 (see experiments/evidence/ar-tashkeel-port-*.json):
 * per FLORES ar_JO sentence it records the model's char/diac input ids, the
 * onnxruntime argmax, and the diacritized text piper hands to espeak.
 *
 *   node web/tashkeel/verify_tashkeel_node.mjs <blobDir> <groundtruth.json> [limit]
 *
 * Prints the character-level agreement and writes the port's diacritized text to
 * <blobDir>/diacritized.json so the phoneme-id gate can run espeak on k2.
 */

import { readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { TashkeelDiacritizer } from './tashkeel.mjs';

const [blobDir, gtPath, limitArg] = process.argv.slice(2);
if (!blobDir || !gtPath) {
  console.error('usage: verify_tashkeel_node.mjs <blobDir> <groundtruth.json> [limit]');
  process.exit(2);
}

const manifest = JSON.parse(readFileSync(join(blobDir, 'tashkeel.json'), 'utf8'));
const bin = readFileSync(join(blobDir, manifest.weights));
const buffer = bin.buffer.slice(bin.byteOffset, bin.byteOffset + bin.byteLength);
const d = new TashkeelDiacritizer(manifest, buffer);

const rows = JSON.parse(readFileSync(gtPath, 'utf8'));
const limit = limitArg ? Number(limitArg) : rows.length;

let chars = 0;
let charsOk = 0;
let sentOk = 0;
let textOk = 0;
const bad = [];
const out = [];
const t0 = Date.now();

for (let i = 0; i < Math.min(limit, rows.length); i += 1) {
  const r = rows[i];
  const preds = d.model.predict(Int32Array.from(r.char_ids), Int32Array.from(r.diac_ids));
  if (preds.length !== r.predictions.length) throw new Error(`length mismatch at ${i}`);
  let m = 0;
  for (let j = 0; j < preds.length; j += 1) if (preds[j] === r.predictions[j]) m += 1;
  chars += preds.length;
  charsOk += m;
  if (m === preds.length) sentOk += 1;
  else bad.push(i);

  const text = d.diacritize(r.text);
  out.push(text);
  if (text === r.diacritized) textOk += 1;
}

const n = Math.min(limit, rows.length);
const secs = (Date.now() - t0) / 1000;
writeFileSync(join(blobDir, 'diacritized.json'), JSON.stringify(out));
console.log(`dtype=${manifest.dtype} blob=${manifest.blobBytes} bytes`);
console.log(`chars       ${charsOk}/${chars} = ${((100 * charsOk) / chars).toFixed(6)}%`);
console.log(`sentences   ${sentOk}/${n} clean at the class level`);
console.log(`text        ${textOk}/${n} byte-identical to piper's diacritized string`);
console.log(`time        ${secs.toFixed(1)}s for ${n} sentences (${((1000 * secs) / n).toFixed(1)} ms/sentence)`);
if (bad.length) console.log(`first bad sentence indices: ${bad.slice(0, 10).join(', ')}`);
