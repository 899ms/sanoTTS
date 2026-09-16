#!/usr/bin/env python3
"""Whisper CER for an int8 render set against its fp32 twin.

tools/audition_voice_package.py answers "did the student keep what the teacher
had" for a SHIPPED PACKAGE, rendering through pypkg's numpy runtime. It cannot
answer this question, because the thing under test here is not a package: it is
the C runtime's int8 path, and the only way to hear that is to let the C
runtime render it. So this takes the wav pairs mcu/test/piperlite_e2e_main.c
writes and scores both with the SAME normalisation and the same ASR that
audition_voice_package.py uses -- `error_rates` and `load_for_whisper` are
imported from tools/audition_piper_teacher.py, not reimplemented.

Read the GAP between the two columns, not the absolute CER. The absolute
number carries the voice's own distillation loss plus Whisper's opinion of the
sentence; the difference between int8 and fp32 on the same sentence, rendered
by the same code, is the cost of quantisation and nothing else.

  python tools/score_int8_vs_fp32_renders.py \\
      --wav-dir artifacts/.../amy/wav --texts artifacts/.../amy/ids/texts.json \\
      --language en --out int8-cer.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from audition_piper_teacher import error_rates, load_for_whisper  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--wav-dir", type=Path, required=True,
                    help="dir holding NNN-int8.wav / NNN-f32.wav pairs")
    ap.add_argument("--texts", type=Path, required=True,
                    help="JSON list of the reference sentences, row order")
    ap.add_argument("--language", type=str, required=True)
    ap.add_argument("--whisper-model", default="small")
    ap.add_argument("--suffix", default="", help='e.g. "-locked" for the '
                    "duration-locked render set")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    import whisper  # noqa: PLC0415

    texts = json.loads(args.texts.read_text(encoding="utf-8"))
    if not isinstance(texts, list) or not texts:
        raise SystemExit(f"{args.texts}: expected a non-empty JSON list")

    print(f"loading whisper '{args.whisper_model}' ...", flush=True)
    asr = whisper.load_model(args.whisper_model)

    rows = []
    for i, sentence in enumerate(texts):
        paths = {tag: args.wav_dir / f"{i:03d}{args.suffix}-{tag}.wav"
                 for tag in ("int8", "f32")}
        missing = [str(p) for p in paths.values() if not p.is_file()]
        if missing:
            print(f"  row {i}: missing {missing}")
            continue
        row = {"i": i, "text": sentence}
        for tag, path in paths.items():
            heard = asr.transcribe(load_for_whisper(path),
                                   language=args.language, fp16=False)["text"]
            cer, wer = error_rates(sentence, str(heard))
            row[f"{tag}_cer"] = cer
            row[f"{tag}_wer"] = wer
            row[f"{tag}_heard"] = str(heard).strip()
        rows.append(row)
        print(f"  {i:02d} int8 cer={row['int8_cer']:.4f}  fp32 cer={row['f32_cer']:.4f}"
              f"  delta={row['int8_cer'] - row['f32_cer']:+.4f}", flush=True)

    if not rows:
        raise SystemExit("nothing scored")
    summary = {}
    for tag in ("int8", "f32"):
        summary[f"{tag}_cer_mean"] = sum(r[f"{tag}_cer"] for r in rows) / len(rows)
        wers = [r[f"{tag}_wer"] for r in rows if r[f"{tag}_wer"] == r[f"{tag}_wer"]]
        summary[f"{tag}_wer_mean"] = sum(wers) / len(wers) if wers else None
    summary["cer_delta_mean"] = summary["int8_cer_mean"] - summary["f32_cer_mean"]
    summary["rows"] = len(rows)
    summary["identical_transcripts"] = sum(
        1 for r in rows if r["int8_heard"] == r["f32_heard"])
    summary["rows_int8_worse"] = sum(1 for r in rows if r["int8_cer"] > r["f32_cer"])
    summary["rows_int8_better"] = sum(1 for r in rows if r["int8_cer"] < r["f32_cer"])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(
        {"wav_dir": str(args.wav_dir), "language": args.language,
         "whisper_model": args.whisper_model, "suffix": args.suffix,
         "summary": summary, "rows": rows}, indent=1, ensure_ascii=False))
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
