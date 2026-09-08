"""Numpy-only grapheme -> phoneme model for words the dictionary does not have.

The checkpoint is PeterReid/graphemes_to_phonemes_en_us (Apache-2.0), a
751,551-parameter BART with one encoder layer, one decoder layer, one attention
head and d_model 128, trained on misaki's own us_gold/us_silver dictionaries.
It was not distilled from espeak-ng, which is the whole point: it is the OOV
path for a front end that must not carry a GPL dependency.

`tools/build_nano_g2p_assets.py` converts the published safetensors to
`g2p_data/oov_bart_en_us.npz`; this module is the forward pass, transcribed
from transformers' `BartModel` so it can run under numpy alone:

  * post-norm blocks (residual add, then LayerNorm), which is BART, not the
    pre-norm arrangement most later decoders use;
  * learned positional embeddings with the offset of 2 that BART reserves for
    pad and bos, hence the 66-row table for 64 positions;
  * `scale_embedding=false`, so token embeddings are used unscaled;
  * exact erf gelu, not the tanh approximation;
  * greedy decoding from `decoder_start_token_id`, which is what the upstream
    reference script gets from its generation config (no beams, no sampling).

Decoding is per word and words are short, so there is no KV cache: the decoder
is re-run over the whole prefix each step. A cache would save microseconds on
sequences that are at most a few dozen tokens and is not worth the surface area.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).resolve().parent / "g2p_data"
MODEL_PATH = DATA_DIR / "oov_bart_en_us.npz"

# The tokenizer reserves 0..3 for <pad>/<s>/</s>/<unk> and then indexes the
# character lists; the shipped config records both lists already padded with
# four underscores so the character at index i is simply chars[i].
SPECIAL_TOKEN_COUNT = 4
UNK_ID = 3


class OOVModelError(RuntimeError):
    """Raised when the fallback model is unusable; never swallowed by callers."""


def _layer_norm(x: np.ndarray, weight: np.ndarray, bias: np.ndarray, eps: float) -> np.ndarray:
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps) * weight + bias


def _erf(x: np.ndarray) -> np.ndarray:
    """Abramowitz-Stegun 7.1.26 erf, max abs error 1.5e-7.

    transformers maps the "gelu" activation to the erf form, not the tanh
    approximation. numpy has no erf and scipy is not a dependency of this
    package, so the approximation is inlined. 1.5e-7 is far below the margin between competing
    characters in a 63-way argmax; the k2 gate test compares this port's greedy
    output against transformers on the dictionary and finds no disagreement.
    """
    sign = np.sign(x)
    ax = np.abs(x)
    t = 1.0 / (1.0 + 0.3275911 * ax)
    poly = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741
                + t * (-1.453152027 + t * 1.061405429))))
    return sign * (1.0 - poly * np.exp(-ax * ax))


def _gelu_exact(x: np.ndarray) -> np.ndarray:
    return 0.5 * x * (1.0 + _erf(x / math.sqrt(2.0)))


def _softmax(x: np.ndarray) -> np.ndarray:
    shifted = x - x.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


class _Attention:
    """Single-head BartAttention. num_heads is 1 in this checkpoint, asserted
    by the asset builder, so the head reshape collapses to a plain matmul."""

    def __init__(self, weights: dict[str, np.ndarray], prefix: str, scaling: float) -> None:
        self.q_w = weights[f"{prefix}.q_proj.weight"].T
        self.q_b = weights[f"{prefix}.q_proj.bias"]
        self.k_w = weights[f"{prefix}.k_proj.weight"].T
        self.k_b = weights[f"{prefix}.k_proj.bias"]
        self.v_w = weights[f"{prefix}.v_proj.weight"].T
        self.v_b = weights[f"{prefix}.v_proj.bias"]
        self.o_w = weights[f"{prefix}.out_proj.weight"].T
        self.o_b = weights[f"{prefix}.out_proj.bias"]
        self.scaling = scaling

    def __call__(self, query: np.ndarray, memory: np.ndarray,
                 causal: bool = False) -> np.ndarray:
        q = (query @ self.q_w + self.q_b) * self.scaling
        k = memory @ self.k_w + self.k_b
        v = memory @ self.v_w + self.v_b
        scores = q @ k.T
        if causal:
            mask = np.triu(np.ones((scores.shape[0], scores.shape[1]), dtype=bool), k=1)
            scores = np.where(mask, np.float32(-np.inf), scores)
        return _softmax(scores) @ v @ self.o_w + self.o_b


class _Layer:
    """One BartEncoderLayer, or one BartDecoderLayer when `cross` is present."""

    def __init__(self, weights: dict[str, np.ndarray], prefix: str,
                 scaling: float, eps: float, cross: bool) -> None:
        self.eps = eps
        self.self_attn = _Attention(weights, f"{prefix}.self_attn", scaling)
        self.self_ln = (weights[f"{prefix}.self_attn_layer_norm.weight"],
                        weights[f"{prefix}.self_attn_layer_norm.bias"])
        if cross:
            self.cross_attn = _Attention(weights, f"{prefix}.encoder_attn", scaling)
            self.cross_ln = (weights[f"{prefix}.encoder_attn_layer_norm.weight"],
                             weights[f"{prefix}.encoder_attn_layer_norm.bias"])
        else:
            self.cross_attn = None
            self.cross_ln = None
        self.fc1_w = weights[f"{prefix}.fc1.weight"].T
        self.fc1_b = weights[f"{prefix}.fc1.bias"]
        self.fc2_w = weights[f"{prefix}.fc2.weight"].T
        self.fc2_b = weights[f"{prefix}.fc2.bias"]
        self.final_ln = (weights[f"{prefix}.final_layer_norm.weight"],
                         weights[f"{prefix}.final_layer_norm.bias"])

    def __call__(self, hidden: np.ndarray, memory: np.ndarray | None,
                 causal: bool) -> np.ndarray:
        hidden = _layer_norm(hidden + self.self_attn(hidden, hidden, causal=causal),
                             *self.self_ln, self.eps)
        if self.cross_attn is not None:
            if memory is None:
                raise OOVModelError("decoder layer reached with no encoder memory")
            hidden = _layer_norm(hidden + self.cross_attn(hidden, memory),
                                 *self.cross_ln, self.eps)
        ffn = _gelu_exact(hidden @ self.fc1_w + self.fc1_b) @ self.fc2_w + self.fc2_b
        return _layer_norm(hidden + ffn, *self.final_ln, self.eps)


class OOVPhonemizer:
    """Greedy grapheme -> phoneme decoding for one word at a time."""

    def __init__(self, path: Path | str = MODEL_PATH) -> None:
        path = Path(path)
        if not path.is_file():
            raise OOVModelError(
                f"fallback model not found at {path}; run "
                "tools/build_nano_g2p_assets.py to fetch it"
            )
        try:
            with np.load(path, allow_pickle=False) as bundle:
                weights = {name: np.asarray(bundle[name], dtype=np.float32)
                           for name in bundle.files if name != "meta_json"}
                meta_raw = str(bundle["meta_json"])
        except (OSError, ValueError, KeyError) as exc:
            raise OOVModelError(f"could not read {path}: {exc}") from exc
        try:
            meta = json.loads(meta_raw)
        except json.JSONDecodeError as exc:
            raise OOVModelError(f"{path}: meta_json is not valid JSON: {exc}") from exc

        self.grapheme_chars: str = meta["grapheme_chars"]
        self.phoneme_chars: str = meta["phoneme_chars"]
        self.max_positions = int(meta["max_position_embeddings"])
        self.bos_id = int(meta["bos_token_id"])
        self.eos_id = int(meta["eos_token_id"])
        self.start_id = int(meta["decoder_start_token_id"])
        self.offset = int(meta["position_offset"])
        eps = float(meta["layer_norm_eps"])
        heads = int(meta["num_heads"])
        d_model = int(meta["d_model"])
        if heads != 1:
            raise OOVModelError(f"this port assumes one attention head, checkpoint has {heads}")
        scaling = (d_model // heads) ** -0.5

        self.grapheme_to_id = {
            ch: i for i, ch in enumerate(self.grapheme_chars) if i >= SPECIAL_TOKEN_COUNT
        }
        self.embed = weights["model.shared.weight"]
        self.logit_bias = weights["final_logits_bias"][0]
        self.enc_pos = weights["model.encoder.embed_positions.weight"]
        self.dec_pos = weights["model.decoder.embed_positions.weight"]
        self.enc_ln = (weights["model.encoder.layernorm_embedding.weight"],
                       weights["model.encoder.layernorm_embedding.bias"])
        self.dec_ln = (weights["model.decoder.layernorm_embedding.weight"],
                       weights["model.decoder.layernorm_embedding.bias"])
        self.encoder = _Layer(weights, "model.encoder.layers.0", scaling, eps, cross=False)
        self.decoder = _Layer(weights, "model.decoder.layers.0", scaling, eps, cross=True)
        self.eps = eps

    def encode_word(self, word: str) -> list[int]:
        """Character ids with bos/eos, unknown characters mapped to <unk>."""
        return [self.bos_id, *(self.grapheme_to_id.get(c, UNK_ID) for c in word), self.eos_id]

    def __call__(self, word: str) -> str:
        """Phonemes for `word`, or '' if the model emits nothing usable."""
        if not word:
            raise OOVModelError("cannot phonemize an empty word")
        ids = self.encode_word(word)
        if len(ids) > self.max_positions:
            raise OOVModelError(
                f"word {word!r} needs {len(ids)} positions; the model has {self.max_positions}"
            )
        memory = self._encode(np.asarray(ids, dtype=np.int64))

        out: list[int] = [self.start_id]
        for _ in range(self.max_positions - 1):
            logits = self._decode_step(np.asarray(out, dtype=np.int64), memory)
            nxt = int(np.argmax(logits))
            if nxt == self.eos_id:
                break
            out.append(nxt)
        return "".join(
            self.phoneme_chars[i] for i in out[1:]
            if SPECIAL_TOKEN_COUNT <= i < len(self.phoneme_chars)
        )

    def _encode(self, ids: np.ndarray) -> np.ndarray:
        hidden = self.embed[ids] + self.enc_pos[self.offset : self.offset + len(ids)]
        hidden = _layer_norm(hidden, *self.enc_ln, self.eps)
        return self.encoder(hidden, None, causal=False)

    def _decode_step(self, ids: np.ndarray, memory: np.ndarray) -> np.ndarray:
        hidden = self.embed[ids] + self.dec_pos[self.offset : self.offset + len(ids)]
        hidden = _layer_norm(hidden, *self.dec_ln, self.eps)
        hidden = self.decoder(hidden, memory, causal=True)
        return hidden[-1] @ self.embed.T + self.logit_bias


_SHARED: OOVPhonemizer | None = None


def shared_phonemizer() -> OOVPhonemizer:
    """One process-wide instance; loading the npz costs ~3 MB and a few ms."""
    global _SHARED
    if _SHARED is None:
        _SHARED = OOVPhonemizer()
    return _SHARED
