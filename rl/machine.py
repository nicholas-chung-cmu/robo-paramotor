"""Per-machine settings from machine.toml at the repo root.

Imported before JAX: limit_jax_memory() must run before JAX initializes.
"""
import os
import tomllib
from pathlib import Path

PATH = Path(__file__).resolve().parents[1] / "machine.toml"


def load():
    """The settings as a dict; empty if the file is missing."""
    if not PATH.exists():
        return {}
    with PATH.open("rb") as f:
        return tomllib.load(f)


def limit_jax_memory(key="jax_memory_fraction"):
    """Cap JAX's GPU memory at machine.toml's fraction, unless the variable is
    already exported or the setting is absent (no cap)."""
    fraction = load().get(key)
    if fraction is not None:
        os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", str(fraction))
