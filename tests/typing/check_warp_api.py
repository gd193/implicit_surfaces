"""Run strict contract checks and verify every negative diagnostic explicitly.

Usage: python tests/typing/check_warp_api.py /absolute/path/to/pyright
"""
import collections
import json
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "pyrightconfig.warp-api.json"
FIXTURES = ROOT / "tests/typing/warp_api"


def run(executable, *paths):
    result = subprocess.run(
        [executable, "--project", str(CONFIG), "--outputjson",
         *(["--pythonpath", os.environ["PYRIGHT_PYTHONPATH"]] if "PYRIGHT_PYTHONPATH" in os.environ else []),
         *map(str, paths)],
        cwd=ROOT, capture_output=True, text=True,
    )
    if result.returncode not in (0, 1):
        raise AssertionError(result.stderr or result.stdout)
    return json.loads(result.stdout)["generalDiagnostics"]


def main():
    executable = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("PYRIGHT", "pyright")
    valid = run(executable)
    assert not valid, json.dumps(valid, indent=2)
    total = 0
    for negative in sorted(FIXTURES.glob("invalid*.py")):
        expected = collections.Counter(
            (index, match.group(1))
            for index, line in enumerate(negative.read_text().splitlines())
            for match in re.finditer(r"# expect: (\w+)", line)
        )
        assert expected, f"No expected diagnostics in {negative}"
        diagnostics = run(executable, negative)
        actual = collections.Counter(
            (entry["range"]["start"]["line"], entry.get("rule"))
            for entry in diagnostics if Path(entry["file"]) == negative
        )
        unexpected_files = [entry for entry in diagnostics if Path(entry["file"]) != negative]
        assert not unexpected_files, json.dumps(unexpected_files, indent=2)
        assert actual == expected, json.dumps({"fixture": negative.name,
                                               "expected": list(expected.elements()),
                                               "actual": list(actual.elements()),
                                               "diagnostics": diagnostics}, indent=2)
        total += sum(expected.values())
    print(f"Strict valid contract: 0 diagnostics; negative contracts: {total} expected diagnostics")


if __name__ == "__main__":
    main()
