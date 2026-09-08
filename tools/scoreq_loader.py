#!/usr/bin/env python3
"""Load the SCOREQ model class without triggering its package side effects.

    from scoreq_loader import load_scoreq_class
    scoreq = load_scoreq_class()(data_domain="synthetic", mode="nr", use_onnx=True)

The installed package's __init__.py is:

    from .scoreq import Scoreq
    scoreq = Scoreq()

so a plain `import scoreq` CONSTRUCTS a default model at import time -- before
the caller can say which domain or mode it wants, and paying for weights we are
about to discard. Every scoring tool here wants `data_domain="synthetic",
mode="nr"`, so we load scoreq/scoreq.py directly by file path, which skips
__init__.py and hands back the bare class.

This lived in tools/diagnose_roota_sourcefilter_codebook.py, a 697-line LPC
oracle that ten unrelated tools imported purely for these few lines -- and that
pulled build_roota_lpc_targets and the decoder trainer in behind it. That is
what broke the public tree: tools/eval_mos_all.py shipped, its accidental
697-line dependency did not, and SCOREQ scoring raised ImportError for anyone
reproducing our numbers. Import it from here, not from a diagnostic.
"""

from __future__ import annotations

import functools
import importlib.metadata
import importlib.util
from pathlib import Path
from typing import Any

SCOREQ_MODULE_NAME = "scoreq_core_no_init"


@functools.lru_cache(maxsize=1)
def load_scoreq_class() -> type[Any]:
    """Return the SCOREQ `Scoreq` class, bypassing the package __init__.

    Cached: the module is exec'd on first call only. Without this, two callers
    would exec scoreq.py twice and get two distinct class objects, so an
    isinstance check across them would quietly fail.
    """
    try:
        package_files = importlib.metadata.files("scoreq") or []
    except importlib.metadata.PackageNotFoundError as exc:
        # Reached by anyone running the eval tools without the optional scoring
        # extra. Say so, rather than surfacing a bare PackageNotFoundError from
        # three frames down.
        raise RuntimeError(
            "scoreq is not installed in this interpreter (pip install scoreq); "
            "it is optional, and only the SCOREQ metric needs it") from exc

    scoreq_file: Path | None = None
    for package_file in package_files:
        if str(package_file).endswith("scoreq/scoreq.py"):
            scoreq_file = Path(str(package_file.locate()))
            break
    if scoreq_file is None:
        raise RuntimeError(
            "scoreq is installed but its metadata lists no scoreq/scoreq.py; "
            "the package layout changed and this loader needs updating")
    if not scoreq_file.is_file():
        raise RuntimeError(f"{scoreq_file}: listed by scoreq's metadata but missing on disk")

    spec = importlib.util.spec_from_file_location(SCOREQ_MODULE_NAME, scoreq_file)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not build a module spec for {scoreq_file}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    scoreq_class = getattr(module, "Scoreq", None)
    if scoreq_class is None:
        raise RuntimeError(f"{scoreq_file}: no `Scoreq` class found (upstream layout changed?)")
    return scoreq_class
