"""End-to-end checks that aero failures cannot become green test results."""

import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parent


def run_python(arguments, cwd):
    environment = dict(os.environ, PYTHONPATH=str(ROOT),
                       PYTHONDONTWRITEBYTECODE="1", PYTEST_DISABLE_PLUGIN_AUTOLOAD="1")
    return subprocess.run([sys.executable, "-B", *arguments], cwd=cwd,
                          env=environment, capture_output=True, text=True, timeout=60)


@pytest.mark.parametrize("flight, status, summary", [
    ((6.0, 0.0, 0.0), 0, "1 passed"),
    ((6.0, -1.0, 40.0), 1, "1 failed"),
])
def test_pytest_reports_flight_acceptance_result(tmp_path, flight, status, summary):
    # Exercise a real aero acceptance test with controlled flight measurements,
    # then verify pytest's process status, not just the helper's printed output.
    probe = tmp_path / "test_flight_reporting.py"
    probe.write_text(
        "import test_aero as aero\n"
        "def test_flight_acceptance():\n"
        f"    aero._free_flight = lambda *args, **kwargs: {flight!r}\n"
        "    aero.test_powered_climb()\n", encoding="utf-8")
    result = run_python(["-m", "pytest", str(probe), "-q"], tmp_path)
    assert result.returncode == status, result.stdout + result.stderr
    assert summary in result.stdout, result.stdout + result.stderr


def test_standalone_reports_failure_and_continues(tmp_path):
    probe = tmp_path / "run_aero_reporting.py"
    probe.write_text(
        "import test_aero as aero\n"
        "flight_test = aero.test_powered_climb\n"
        "for name in vars(aero).copy():\n"
        "    if name.startswith('test_'):\n"
        "        setattr(aero, name, lambda: None)\n"
        "aero.test_powered_climb = flight_test\n"
        "aero._free_flight = lambda *args, **kwargs: (6.0, -1.0, 40.0)\n"
        "aero.test_validated_thrust_envelope = lambda: aero.check('later check executed', True)\n"
        "raise SystemExit(aero.main())\n", encoding="utf-8")
    result = run_python([str(probe)], tmp_path)
    assert result.returncode == 1, result.stdout + result.stderr
    assert "later check executed" in result.stdout
    assert "1/2 passed" in result.stdout
