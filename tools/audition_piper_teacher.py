#!/usr/bin/env python3
"""Score a Piper teacher's intelligibility with ASR, before distilling it.

    python3 tools/audition_piper_teacher.py \
        --voices de_DE-thorsten-medium ru_RU-irina-medium \
        --texts flores24.json --out audition.json

A distilled student cannot be more intelligible than the teacher it copies, so
the teacher's word/character error rate is the ceiling for anything we ship in
that language. Measuring it costs minutes and decides whether a language is
worth a training run at all -- which is the whole point of running this first.

It also removes the reviewer bottleneck. Judging a German or Korean voice by
ear needs a native speaker; ASR gives a number in every language Whisper
covers, and the number is comparable across languages when the prompts are
translations of each other (FLORES-200 devtest is, which is why the companion
text set is built from it).

WHAT THE NUMBER IS NOT. A low CER means the words are recoverable, not that
the voice sounds good -- prosody, naturalness and artefacts are invisible to
it. Use this to REJECT teachers, and SCOREQ plus a listener to rank the ones
that pass. The multilingual gate already works this way: it only counts a
student's ASR score when the teacher itself came in under 10% CER.

CER is the headline rather than WER because Chinese, Japanese and Thai are not
whitespace-delimited, so a word error rate is not comparable across the set.

Renders with sampling off (noise 0, length 1), matching how the distillation
pipeline renders its packs, so the audio scored here is the audio the students
would be trained against.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import unicodedata
import wave
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))


def fold_chinese(text: str) -> str:
    """Traditional -> simplified, for scoring only.

    Whisper answers in whichever script it feels like, and Tatoeba mixes both,
    so an unfolded CER charges a Chinese voice for orthography it never
    pronounced -- 烏 against 乌 is one error per character and the audio is
    identical. This is not cosmetic: it is the difference between rejecting a
    teacher and accepting it. zh_CN-xiao_ya-medium scores 0.307 unfolded and
    0.104 folded on the same transcripts, i.e. fail against pass on our own
    0.10 gate, and we rejected it once on the unfolded number.

    Left out deliberately: homophones. 他们 and 它们 are both tamen and no
    listener could tell them apart either, so ASR cannot score that distinction
    and we do not pretend to. It inflates every Chinese CER here by a little.
    """
    try:
        from zhconv import convert  # noqa: PLC0415
    except ImportError:
        return text
    return convert(text, "zh-cn")


def normalise(text: str) -> str:
    """Fold away differences ASR should not be penalised for.

    Case, punctuation and whitespace runs are not what this is measuring, and
    the FLORES prompts carry quotation marks and typographic dashes that no
    recogniser reproduces consistently. Digits are LEFT ALONE on purpose: how a
    voice reads "4" is exactly the kind of thing that should count against it.
    """
    text = unicodedata.normalize("NFKC", text)
    text = "".join(" " if unicodedata.category(c).startswith("P") else c for c in text)
    return re.sub(r"\s+", " ", text).strip().lower()


def error_rates(reference: str, hypothesis: str) -> tuple[float, float]:
    """(CER, WER) after normalisation, via jiwer."""
    import jiwer  # noqa: PLC0415

    ref, hyp = normalise(reference), normalise(hypothesis)
    if any("\u4e00" <= c <= "\u9fff" for c in ref):
        ref, hyp = fold_chinese(ref), fold_chinese(hyp)
    if not ref:
        return float("nan"), float("nan")
    cer = jiwer.cer(ref, hyp)
    # A whitespace WER is meaningless for scripts that do not use spaces; the
    # caller decides whether to look at it.
    wer = jiwer.wer(ref, hyp) if len(ref.split()) > 1 else float("nan")
    return float(cer), float(wer)


def load_for_whisper(path: Path) -> Any:
    """Wav -> float32 mono at 16 kHz, in memory.

    whisper.load_audio() shells out to ffmpeg, which is not on every training
    box and is a system change we do not need: these files are our own 16-bit
    PCM, so decoding them here keeps the tool self-contained. Whisper accepts a
    float32 array in place of a path.
    """
    import numpy as np  # noqa: PLC0415
    from scipy.signal import resample_poly  # noqa: PLC0415

    with wave.open(str(path), "rb") as handle:
        rate = handle.getframerate()
        channels = handle.getnchannels()
        width = handle.getsampwidth()
        frames = handle.readframes(handle.getnframes())
    if width != 2:
        raise RuntimeError(f"{path}: expected 16-bit PCM, got {width * 8}-bit")
    audio = np.frombuffer(frames, dtype="<i2").astype("float32") / 32768.0
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    if rate != 16000:
        from math import gcd  # noqa: PLC0415
        g = gcd(int(rate), 16000)
        audio = resample_poly(audio, 16000 // g, int(rate) // g).astype("float32")
    return np.ascontiguousarray(audio)


def render(voice_obj: Any, text: str, path: Path) -> float:
    """Synthesize one sentence to a wav; returns duration in seconds."""
    from piper import SynthesisConfig  # noqa: PLC0415

    # Sampling off: the packs are rendered this way, so the audition hears what
    # the students would actually be trained on.
    config = SynthesisConfig(noise_scale=0.0, length_scale=1.0, noise_w_scale=0.0)
    chunks = list(voice_obj.synthesize(text, config))
    if not chunks:
        raise RuntimeError("piper returned no audio")
    rate = chunks[0].sample_rate
    width = chunks[0].sample_width
    channels = chunks[0].sample_channels
    payload = b"".join(c.audio_int16_bytes for c in chunks)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(width)
        handle.setframerate(rate)
        handle.writeframes(payload)
    return len(payload) / (rate * width * channels)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--voices", nargs="+", required=True,
                        help="Rhasspy voice keys, e.g. de_DE-thorsten-medium")
    parser.add_argument("--texts", type=Path, required=True,
                        help='JSON: {"<whisper lang code>": ["sentence", ...]}')
    parser.add_argument("--rows", type=int, default=24)
    parser.add_argument("--whisper-model", default="small",
                        help="A bigger model lowers ASR's own error floor, at a cost "
                             "in time; 'small' is enough to separate a broken teacher "
                             "from a working one.")
    parser.add_argument("--teacher-dir", type=Path, default=ROOT / "models" / "teachers")
    parser.add_argument("--work-dir", type=Path, default=ROOT / "artifacts" / "auditions")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cer-gate", type=float, default=0.10,
                        help="A teacher above this is not worth distilling (the "
                             "multilingual gate's own threshold).")
    args = parser.parse_args()

    from train_voice_from_piper import resolve_teacher  # noqa: PLC0415
    from piper import PiperVoice  # noqa: PLC0415
    import whisper  # noqa: PLC0415

    texts = json.loads(args.texts.read_text(encoding="utf-8"))
    args.work_dir.mkdir(parents=True, exist_ok=True)

    print(f"loading whisper '{args.whisper_model}' …", flush=True)
    asr = whisper.load_model(args.whisper_model)

    results: list[dict[str, Any]] = []
    for key in args.voices:
        print(f"\n=== {key}", flush=True)
        onnx, config_path = resolve_teacher(key, args.teacher_dir)
        config = json.loads(config_path.read_text())
        lang = ((config.get("espeak") or {}).get("voice") or "").split("-")[0].lower()
        # espeak calls Mandarin 'cmn'; Whisper calls it 'zh'.
        lang = {"cmn": "zh"}.get(lang, lang)
        rows = texts.get(lang)
        if not rows:
            print(f"  SKIP: no prompts for language {lang!r} in {args.texts.name}")
            continue

        voice_obj = PiperVoice.load(str(onnx), config_path=str(config_path))
        out_dir = args.work_dir / key
        out_dir.mkdir(parents=True, exist_ok=True)

        per_row, started = [], time.time()
        for i, sentence in enumerate(rows[: args.rows]):
            wav = out_dir / f"{i:03d}.wav"
            try:
                seconds = render(voice_obj, sentence, wav)
            except Exception as exc:  # noqa: BLE001 - one bad row must not kill the sweep
                print(f"  row {i}: render failed: {exc}")
                continue
            heard = asr.transcribe(load_for_whisper(wav), language=lang, fp16=False)["text"]
            cer, wer = error_rates(sentence, str(heard))
            per_row.append({"i": i, "text": sentence, "heard": str(heard).strip(),
                            "cer": cer, "wer": wer, "seconds": seconds})
            print(f"  {i:02d} cer={cer:.3f} {str(heard).strip()[:60]}", flush=True)

        if not per_row:
            print("  no rows scored")
            continue
        cers = sorted(r["cer"] for r in per_row)
        wers = [r["wer"] for r in per_row if r["wer"] == r["wer"]]
        median = cers[len(cers) // 2]
        mean = sum(cers) / len(cers)
        entry = {
            "voice": key, "language": lang, "rows": len(per_row),
            "cer_mean": mean, "cer_median": median,
            "wer_mean": (sum(wers) / len(wers)) if wers else None,
            "audio_seconds": sum(r["seconds"] for r in per_row),
            "elapsed_s": time.time() - started,
            "verdict": "pass" if mean <= args.cer_gate else "fail",
            "per_row": per_row,
        }
        results.append(entry)
        print(f"  -> CER mean {mean:.3f} median {median:.3f}  [{entry['verdict']}]",
              flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "generated": time.strftime("%Y-%m-%d %H:%M"),
        "whisper_model": args.whisper_model,
        "cer_gate": args.cer_gate,
        "prompts": str(args.texts),
        "note": "FLORES-200 devtest; prompts are translations of each other, so CER "
                "is comparable across languages. Low CER means intelligible, not good.",
        "results": results,
    }, ensure_ascii=False, indent=1), encoding="utf-8")

    print("\n" + "=" * 62)
    print(f"{'voice':30} {'lang':5} {'CER':>7} {'verdict':>8}")
    for r in sorted(results, key=lambda x: x["cer_mean"]):
        print(f"{r['voice']:30} {r['language']:5} {r['cer_mean']:7.3f} {r['verdict']:>8}")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
