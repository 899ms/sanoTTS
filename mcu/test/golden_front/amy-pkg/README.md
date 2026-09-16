# amy-pkg — the int8 front fixture, rebuilt from the published package

`golden_front/{amy,vi,hindi}` predate `tools/export_front_q8.py` and cannot be
given int8 blobs: the training checkpoints their `golden.json` names no longer
exist anywhere. This directory is the same amy voice rebuilt from the
**published** package, which is the only route a user of a shipped voice has —
and, usefully, the exact route Ampixa/sanoTTS#12 is about.

The `.bin` payloads are not versioned (the repo's `*.bin` rule; only
`mcu/test/fixtures/**` is exempt). Regenerate them on a box that may load
models — never this Mac. `W` is any scratch directory.

```bash
python tools/repack_package_to_checkpoints.py ~/.cache/sanotts/amy-en-1p46m --out $W/ckpt

# calibration text and scored text must be disjoint
python tools/make_front_latent_pack.py \
    --duration $W/ckpt/duration-student.pt --acoustic $W/ckpt/latent-student.pt \
    --piper-model models/teachers/en_US-amy-medium/en_US-amy-medium.onnx \
    --texts en_US.train13k.jsonl --rows 16 --length-scale 1.08 --out $W/pack
python tools/make_front_latent_pack.py \
    --duration $W/ckpt/duration-student.pt --acoustic $W/ckpt/latent-student.pt \
    --piper-model models/teachers/en_US-amy-medium/en_US-amy-medium.onnx \
    --texts en_US.diverse-heldout24.jsonl --rows 24 --length-scale 1.08 --out $W/evalpack

python tools/export_front_golden.py $W/ckpt/duration-student.pt \
    $W/ckpt/latent-student.pt --pack $W/evalpack --chunk-row 0 --out $W/front
python tools/export_front_q8.py $W/ckpt/duration-student.pt \
    $W/ckpt/latent-student.pt --pack $W/pack --calib-n 16 --length-scale 1.08 \
    --out $W/front

python tools/export_piperlite_golden.py $W/ckpt/decoder-student.pt \
    --pack $W/evalpack --chunk-row 0 --out $W/dec
python tools/export_piperlite_q8.py $W/ckpt/decoder-student.pt --pack $W/pack --out $W/dec
```

Then copy into place (the stage goldens the decoder exporter also writes are
several MB each and no gate here needs them):

```
golden_front/amy-pkg     <- meta.bin front_weights_f32.bin front_meta_q8.bin
                            front_weights_q8.bin ids.bin durations.bin
                            durations_ls125.bin latent.bin golden.json
                            front_calib_q8.json
golden_piperlite/amy-pkg <- meta.bin weights_f32.bin meta_q8.bin weights_q8.bin
                            z.bin audio.bin golden.json calib_q8.json
```

The golden row is held-out row 0 of `en_US.diverse-heldout24.jsonl`; the
activation clips are calibrated on 16 rows of `en_US.train13k.jsonl`. The
package is fp16, so `front_weights_f32.bin` here is fp16 widened back to fp32 —
that is the model the published voice actually is, not an approximation of it.

Numbers this fixture reproduces are in
`experiments/evidence/piperlite-int8-20260916.json`.
