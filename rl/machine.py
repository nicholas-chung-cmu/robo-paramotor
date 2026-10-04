"""Per-machine settings: machine.toml at the repo root, with any values in a
git-ignored machine.local.toml next to it taking precedence on that computer.

Imported before JAX: limit_jax_memory() must run before JAX initializes.
Run `python3 -m rl.machine gpus` for the GPU list (used by docker/train.sh).
"""
import sys
import os
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "machine.toml"
LOCAL = ROOT / "machine.local.toml"


def load():
    """The settings as a dict: machine.toml, overridden by machine.local.toml."""
    out = {}
    for path in (PATH, LOCAL):
        if path.exists():
            with path.open("rb") as f:
                out.update(tomllib.load(f))
    return out


def limit_jax_memory(key="jax_memory_fraction"):
    """Cap JAX's GPU memory at machine.toml's fraction, unless the variable is
    already exported or the setting is absent (no cap)."""
    fraction = load().get(key)
    if fraction is not None:
        os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", str(fraction))


if __name__ == "__main__" and sys.argv[1:] == ["gpus"]:
    print(",".join(map(str, load().get("gpus", [0]))))
