#!/bin/bash
# Open the model in the MuJoCo viewer. Does NOT rebuild unless asked.
#
#   ./view.sh              LIVE: free flight, powered, camera tracks the pod
#   ./view.sh --sweep      live, with the brakes cycling
#   ./view.sh --zoom 12    camera pulled back further
#   ./view.sh --slow       live at 1/20 speed (also --speed 0.02)
#   ./view.sh --freeze     hold the design pose, nothing moves
#   ./view.sh --trim       settle to hanging trim, then hold it
#   ./view.sh --raw        stock MuJoCo viewer -- WARNING: NO AERODYNAMICS.
#                          It loads the XML only, and aero lives in Python, so
#                          the vehicle just falls. Use it for geometry, not flight.
#   ./view.sh --bare       aircraft only, no ground or mountains
#   ./view.sh --aero lumped  the paper's single-force model instead of strip
#   ./view.sh --aero off     no aerodynamics at all (it just falls)
#   ./view.sh --thrust 0.8   thrust in N; the XML caps it at 1.0 N (hardware clamp)
#   ./view.sh --inertia    show the equivalent inertia boxes (mass, NOT drag)
#   ./view.sh --build      regenerate paramotor.xml + scene.xml first
#
# This does NOT rebuild by default: it views the XML exactly as it sits on
# disk, so hand edits survive and you can debug the file you are looking at.
# Pass --build after changing build_paramotor.py.
#
# Actuator sliders: right-hand panel -> Control (thrust, servo_pos_L,
# servo_pos_R). Drag bodies with Ctrl + right/left-drag.
#
# AERODYNAMICS lives in paramotor_aero.py, NOT in the XML: the MJCF carries
# density="0" viscosity="0" on purpose and MuJoCo's own fluid model is off.
# The viewer installs the callback itself. Editing coefficients means editing
# paramotor_params.py -- no rebuild needed, it is read at startup.
#
# PLATFORMS
#   macOS requires mjpython rather than python: the viewer must own the main
#   thread. mjpython is macOS-only and ships with the mujoco wheel there.
#   Windows (Git Bash / MSYS2) and Linux use the plain interpreter, and the
#   venv puts it in Scripts/ rather than bin/ on Windows.
#   Override either with env vars if your layout differs:
#       VENV=/path/to/venv ./view.sh
#       PY=/path/to/python MJ=/path/to/mjpython ./view.sh
set -euo pipefail
cd "$(dirname "$0")"

VENV="${VENV:-../.venv}"

case "$(uname -s)" in
  Darwin)                 OS=mac     ;;
  MINGW*|MSYS*|CYGWIN*)   OS=windows ;;
  *)                      OS=linux   ;;
esac

# Windows venvs use Scripts/ and .exe; everything else uses bin/.
if [[ "$OS" == "windows" ]]; then
  BIN="$VENV/Scripts"; EXE=".exe"
else
  BIN="$VENV/bin";     EXE=""
fi

PY="${PY:-$BIN/python$EXE}"

# Fall back to whatever python is on PATH if the venv is missing or elsewhere.
if [[ ! -x "$PY" ]]; then
  if command -v python3 >/dev/null 2>&1; then PY=python3
  elif command -v python >/dev/null 2>&1; then PY=python
  else
    echo "view.sh: no python found (looked for $BIN/python$EXE). Set PY= or VENV=." >&2
    exit 1
  fi
  echo "view.sh: no venv at $VENV, using $(command -v "$PY")" >&2
fi

# The viewer launcher: mjpython on macOS, the same python everywhere else.
if [[ "$OS" == "mac" ]]; then
  MJ="${MJ:-$BIN/mjpython}"
  if [[ ! -x "$MJ" ]]; then
    if command -v mjpython >/dev/null 2>&1; then MJ=mjpython
    else
      echo "view.sh: mjpython not found at $MJ. macOS needs it -- the viewer" >&2
      echo "         must own the main thread. Install with: $PY -m pip install mujoco" >&2
      exit 1
    fi
  fi
else
  MJ="${MJ:-$PY}"
fi

# Default: view the XML as it sits on disk so hand edits survive.
# --build regenerates it from build_paramotor.py first.
ARGS=()
BUILD=0
for a in "$@"; do
  if [[ "$a" == "--build" ]]; then BUILD=1; else ARGS+=("$a"); fi
done

if [[ $BUILD -eq 1 ]]; then
  echo "--build: regenerating paramotor.xml + scene.xml"
  "$PY" build_paramotor.py
fi

if [[ "${ARGS[0]:-}" == "--raw" ]]; then
  exec "$MJ" -m mujoco.viewer --mjcf=scene.xml
else
  # ${ARGS[@]+...} so an empty array does not trip set -u on bash 3.2 (macOS)
  # and does not pass a bogus empty argument through to argparse.
  exec "$MJ" view_paramotor.py ${ARGS[@]+"${ARGS[@]}"}
fi
