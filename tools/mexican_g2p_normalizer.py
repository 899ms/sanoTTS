"""Mexican-Spanish pre-phonemization normalizer (training-side re-export).

    from mexican_g2p_normalizer import normalize_mexican_g2p
    text = normalize_mexican_g2p(text, espeak_voice)

The implementation lives in ``pypkg/sanotts/mexican_g2p.py`` because the fix
has to happen at INFERENCE to matter: the student maps phoneme ids to audio, so
rewriting the text only at pack-build time never teaches it the word "Oaxaca" --
a user typing it still gets /oaksaka/ back. Training and inference must apply
the identical rewrite, so there is one implementation and this re-exports it.

`espeak_voice` is required and everything is a no-op unless it is es-419. See
the module docstring there for why that gate is not optional.
"""

from __future__ import annotations

import sys
from pathlib import Path

_PYPKG = Path(__file__).resolve().parents[1] / "pypkg"
if str(_PYPKG) not in sys.path:
    sys.path.insert(0, str(_PYPKG))

from sanotts.mexican_g2p import applies_to, normalize_mexican_g2p  # noqa: E402,F401

__all__ = ["applies_to", "normalize_mexican_g2p"]
