#!/usr/bin/env python3
"""Assert the Home Assistant voice list matches the sanotts package registry.

`custom_components/sanotts/const.py` hardcodes its voice table so the
integration imports cleanly without reaching into the installed package at
module load. That duplication is only safe if something checks it, which is
this script: a voice added to `pypkg/sanotts/tables/voices.json` but not to the
integration becomes a failure here rather than a voice that silently never
appears in Home Assistant.

    python3 tools/check_sanotts_ha_voices.py

Exit 0 when the two agree, 1 otherwise.
"""

from __future__ import annotations

import ast
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
PACKAGE_TABLE = ROOT / "pypkg" / "sanotts" / "tables" / "voices.json"
HA_CONST = ROOT / "custom_components" / "sanotts" / "const.py"


def package_voices() -> dict[str, str]:
    """Alias -> Home Assistant language tag, from the package registry."""
    data = json.loads(PACKAGE_TABLE.read_text())
    return {
        alias: entry["language"].replace("_", "-")
        for alias, entry in data["voices"].items()
    }


def integration_voices() -> dict[str, str]:
    """Alias -> language tag, parsed out of the integration's VOICES tuple.

    Parsed with `ast` rather than imported, so this runs without Home Assistant
    installed and without executing integration code.
    """
    tree = ast.parse(HA_CONST.read_text())
    for node in ast.walk(tree):
        if not isinstance(node, ast.AnnAssign) or not isinstance(node.target, ast.Name):
            continue
        if node.target.id != "VOICES" or node.value is None:
            continue
        voices: dict[str, str] = {}
        for call in node.value.elts:  # type: ignore[attr-defined]
            if not isinstance(call, ast.Call) or len(call.args) != 3:
                raise SystemExit(f"unexpected VOICES entry: {ast.dump(call)[:120]}")
            alias, _label, language = (ast.literal_eval(a) for a in call.args)
            voices[alias] = language
        return voices
    raise SystemExit(f"no VOICES assignment found in {HA_CONST}")


def main() -> int:
    """Compare the two tables and report every difference."""
    pkg = package_voices()
    ha = integration_voices()

    problems: list[str] = []
    for alias in sorted(set(pkg) - set(ha)):
        problems.append(
            f"  {alias!r} is in the package registry but missing from the integration"
        )
    for alias in sorted(set(ha) - set(pkg)):
        problems.append(
            f"  {alias!r} is in the integration but not in the package registry"
        )
    for alias in sorted(set(pkg) & set(ha)):
        if pkg[alias] != ha[alias]:
            problems.append(
                f"  {alias!r} language differs: package {pkg[alias]!r} vs "
                f"integration {ha[alias]!r}"
            )

    if problems:
        print(f"FAIL: {PACKAGE_TABLE.name} and {HA_CONST.name} disagree:")
        print("\n".join(problems))
        return 1

    print(f"PASS: {len(pkg)} voices agree between the package and the integration")
    for alias in sorted(pkg):
        print(f"  {alias:<10} {pkg[alias]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
