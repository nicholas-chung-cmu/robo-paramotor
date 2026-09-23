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
#   ./view.sh --thrust 0.8   thrust in N; validated envelope is 0 to ~1.0 N
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
# macOS requires mjpython rather than python: the viewer must own the main thread.
set -euo pipefail
cd "$(dirname "$0")"
PY=../.venv/bin/python
MJ=../.venv/bin/mjpython

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
  exec "$MJ" view_paramotor.py "${ARGS[@]:-}"
fi
