#!/bin/bash
# Train, evaluate and analyze PPO policies inside the Docker image.
#
#   docker/train.sh                                  # one run, named ppo-<timestamp>
#   docker/train.sh --name first --updates 2000 --num-envs 2048
#   docker/train.sh --name first --resume --updates 1000   # continue runs/first
#   docker/train.sh --name baseline --seeds 5 --updates 500  # runs/baseline/seed0..4
#   docker/train.sh --compare baseline high_lr       # rliable comparison of configs
#   docker/train.sh --smoke                          # minute-long end-to-end check
#   docker/train.sh --name first --verbose           # + docker build output, PPO losses/KL per
#                                                    #   update, watcher messages (-v also works)
#   docker/train.sh --analyze first                  # re-plot runs/first (works mid-training)
#   docker/train.sh --cpu --smoke                    # no GPU: CPU image, no --gpus
#   docker/train.sh --name first --episodes 100      # 100 evaluation flights per route
#   docker/train.sh --test                           # run the test suite in the image
#   docker/train.sh --test tests/test_rl.py -k gps   # any pytest arguments after --test
#   docker/train.sh --name first --retries 5         # retry a crashed run up to 5 times (default 3)
#   docker/train.sh --finish first                   # finish an interrupted runs/first to its original
#                                                    #   target (same retries), then evaluate + analyze
#   docker/train.sh --name base --seeds 4 --gpus 0,1 # seeds in parallel, one worker per GPU
#                                                    #   (default GPU list: machine.toml `gpus`)
#   docker/train.sh --name first --headed            # train + a window replaying, for every
#                                                    # new checkpoint, its best of 16 whole flights
#   docker/train.sh --watch first                    # that window for a run already training
#   docker/train.sh --stop                           # stop every training/evaluation job (no retry)
#   docker/train.sh --viewer                         # interactive MuJoCo viewer (free flight)
#   docker/train.sh --viewer --sweep --zoom 12       # viewer options after --viewer
#   docker/train.sh --viewer python -m rl.evaluate runs/first/checkpoint.pkl \
#       --path left --episodes 1 --view              # replay one evaluation flight
#
# Ctrl-C (or `kill <train.sh pid>`) stops the job's containers and does not
# retry; only crashes are retried. --stop does the same from another terminal.
# The watcher window stays open; close it to stop it.
#
# Any option it does not recognise goes straight to `python -m rl.train`
# (see `python -m rl.train --help`; --config file.json takes env/ppo overrides).
# Pass --config to give a configuration its hyperparameters, e.g.
#   docker/train.sh --name high_lr --seeds 5 --config configs/high_lr.json
#
# Everything lands in runs/<name>/ on the host (runs/<name>/seed<k>/ with --seeds):
#   config.json, metrics.csv, eval.csv, checkpoint.pkl, train.log   (training)
#   eval/summary.csv, flights.csv, routes.csv, meta.json             (evaluation)
#   analysis/{training,curriculum,evaluation}.png, analysis/report.md
# --compare writes runs/compare/<a>-vs-<b>/ (see python -m rl.compare --help).
#
# The image is rebuilt only when rl/requirements-rl.txt changes (Docker layer
# cache); code is mounted, not copied, so edits apply immediately.
# Set JAX_CUDA=cuda12 for drivers older than CUDA 13.
set -euo pipefail
cd "$(dirname "$0")/.."

ORIGINAL_ARGS="$*"
NAME="ppo-$(date +%Y%m%d-%H%M%S)"
CPU=0 RESUME=0 FINISH=0 SMOKE=0 SEEDS=1 SEED=0 EPISODES=20 MODE=train HEADED=0 VERBOSE=0 RETRIES=3
TARGETS=() TRAIN_ARGS=() GPU_ARG=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --name)      NAME="$2"; shift 2 ;;
    --cpu)       CPU=1; shift ;;
    --resume)    RESUME=1; shift ;;
    --finish)    RESUME=1; FINISH=1; NAME="$2"; shift 2 ;;
    --retries)   RETRIES="$2"; shift 2 ;;
    --gpus)      GPU_ARG="$2"; shift 2 ;;
    --smoke)     SMOKE=1; TRAIN_ARGS+=("$1"); shift ;;
    --seeds)     SEEDS="$2"; shift 2 ;;
    --seed)      SEED="$2"; shift 2 ;;
    --episodes)  EPISODES="$2"; shift 2 ;;
    --analyze)   MODE=analyze; TARGETS+=("$2"); shift 2 ;;
    --test)      MODE=test; shift; TARGETS=("$@"); break ;;
    --headed)    HEADED=1; shift ;;
    --verbose|-v) VERBOSE=1; TRAIN_ARGS+=(--verbose); shift ;;
    --watch)     MODE=watch; TARGETS+=("$2"); shift 2 ;;
    --stop)      MODE=stop; shift ;;
    --viewer)    MODE=viewer; shift; TARGETS=("$@"); break ;;
    --compare)   MODE=compare; shift
                 while [[ $# -gt 0 && "$1" != --* ]]; do TARGETS+=("$1"); shift; done ;;
    -h|--help)   sed -n '2,/^set -euo/p' "$0" | sed '$d'; exit 0 ;;
    *)           TRAIN_ARGS+=("$1"); shift ;;
  esac
done

# Every job container is labelled with the train.sh that started it, so a stop
# can find them all. The watcher window is not, so it outlives the job.
LABEL="paramotor.job"
stop_jobs() {  # stop_jobs [owner pid]: docker kill this job's containers (all if no pid)
  local filter="label=$LABEL${1:+=$1}"
  docker ps -q --filter "$filter" | xargs -r docker kill >/dev/null 2>&1 || true
}
if [[ $MODE == stop ]]; then
  # Signal each train.sh holding a GPU lock (its trap stops its containers and
  # skips the retry), then kill any labelled container left over.
  for lock in runs/.gpu*.lock; do
    [[ -f "$lock" ]] || continue
    pid=$(sed -n 's/^pid \([0-9]*\) .*/\1/p' "$lock" | head -n 1)
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
      echo "train.sh: stopping pid $pid ($lock)"; kill -TERM "$pid" || true
    fi
  done
  sleep 2
  stop_jobs
  echo "train.sh: no training jobs running"
  exit 0
fi

if [[ $CPU -eq 1 ]]; then JAX_CUDA=cpu; else JAX_CUDA="${JAX_CUDA:-cuda13}"; fi
IMAGE="paramotor-rl:$JAX_CUDA"

if [[ $VERBOSE -eq 1 ]]; then
  docker build -f docker/Dockerfile -t "$IMAGE" --build-arg JAX_CUDA="$JAX_CUDA" .
else
  docker build -q -f docker/Dockerfile -t "$IMAGE" --build-arg JAX_CUDA="$JAX_CUDA" . >/dev/null
fi

# GPUs to use: --gpus 0,1, else the `gpus` list in machine.toml (or a
# machine.local.toml override), else GPU 0. Each
# container sees exactly ONE of them (as its device 0), so the training code
# needs no multi-GPU support: --seeds runs one worker per GPU in parallel.
if [[ -z "$GPU_ARG" ]]; then
  GPU_ARG=$(python3 -m rl.machine gpus 2>/dev/null || echo 0)
fi
IFS=, read -r -a GPUS <<< "$GPU_ARG"
GPU_DEVICE="${GPUS[0]}"   # the GPU in_container uses; workers set their own

# Run as the host user so runs/ is not owned by root. HOME and MPLCONFIGDIR
# point at /tmp because that user has no home directory in the image.
in_container() {
  local gpu=() tty=() label=()
  [[ "${WATCHER:-0}" -eq 1 ]] || label=(--label "$LABEL=$$")
  [[ $CPU -eq 0 && "${NO_GPU:-0}" -eq 0 ]] && gpu=(--gpus "device=$GPU_DEVICE")
  [[ -t 1 ]] && tty=(-t)  # live, coloured output when run from a terminal
  docker run --rm ${label[@]+"${label[@]}"} ${gpu[@]+"${gpu[@]}"} ${tty[@]+"${tty[@]}"} ${display[@]+"${display[@]}"} --ipc=host \
    -u "$(id -u):$(id -g)" -e HOME=/tmp -e MPLCONFIGDIR=/tmp \
    -e WARP_CACHE_PATH=/workspace/runs/.warp_cache \
    -v "$PWD:/workspace" -w /workspace "$IMAGE" "$@"
}

# The run directories behind a name: runs/<name>/seed* if present, else runs/<name>.
run_dirs() {
  local seeds=("runs/$1"/seed*/)
  if [[ -d "${seeds[0]}" ]]; then printf '%s\n' "${seeds[@]%/}"; else echo "runs/$1"; fi
}

# Share the host X server: its socket plus this session's auth cookie, so no
# `xhost +` is needed. Linux/X11 (incl. XWayland) only.
use_display() {
  [[ -n "${DISPLAY:-}" && -d /tmp/.X11-unix ]] || {
    echo "train.sh: a window needs an X11 display (DISPLAY is unset)" >&2; exit 1; }
  display=(-e DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix:ro)
  if [[ -n "${XAUTHORITY:-}" && -f "$XAUTHORITY" ]]; then
    display+=(-e XAUTHORITY=/tmp/.Xauthority -v "$XAUTHORITY:/tmp/.Xauthority:ro")
  fi
}

# One GPU job per GPU. Two jobs on one card can drive it out of memory, and on
# the 5080 desktop that has crashed the NVIDIA driver and the displays.
# Training, evaluation and tests hold runs/.gpu<N>.lock for each GPU they use
# until train.sh exits; the watcher window and the interactive viewer are
# light and do not take it.
LOCK_FDS=()
# Close the lock descriptors in a subshell that outlives training (the watcher
# window), so the GPUs are free as soon as train.sh exits.
release_locks() { local fd; for fd in ${LOCK_FDS[@]+"${LOCK_FDS[@]}"}; do exec {fd}>&-; done; }
gpu_lock() {  # gpu_lock N...
  [[ $CPU -eq 1 ]] && return 0
  mkdir -p runs
  local n fd
  for n in "$@"; do
    exec {fd}>>"runs/.gpu$n.lock"
    LOCK_FDS+=("$fd")
    if ! flock -n "$fd"; then
      echo "train.sh: GPU $n is busy (runs/.gpu$n.lock): $(cat "runs/.gpu$n.lock")" >&2
      echo "train.sh: wait for that job, stop it, or pick other GPUs with --gpus." >&2
      exit 1
    fi
    : > "runs/.gpu$n.lock"
    echo "pid $$ since $(date '+%F %T'): docker/train.sh $ORIGINAL_ARGS" >> "runs/.gpu$n.lock"
  done
}

case "$MODE" in
  viewer)
    use_display
    # Bare --viewer or viewer options: the interactive viewer. Else: any command.
    if [[ ${#TARGETS[@]} -eq 0 || "${TARGETS[0]}" == --* ]]; then
      TARGETS=(python -m viewer.view_paramotor ${TARGETS[@]+"${TARGETS[@]}"})
    fi
    in_container "${TARGETS[@]}"
    exit $? ;;
  watch)
    use_display
    WATCHER=1 in_container python -m viewer.watch_training "runs/${TARGETS[0]}"
    exit $? ;;
  test)
    gpu_lock "$GPU_DEVICE"
    in_container python -m pytest -v ${TARGETS[@]+"${TARGETS[@]}"}
    exit $? ;;
  analyze)
    for dir in $(run_dirs "${TARGETS[0]}"); do NO_GPU=1 in_container python -m rl.analyze "$dir"; done
    exit 0 ;;
  compare)
    [[ ${#TARGETS[@]} -ge 1 ]] || { echo "train.sh: --compare needs run names" >&2; exit 1; }
    NO_GPU=1 in_container python -m rl.compare "${TARGETS[@]/#/runs/}"
    exit 0 ;;
esac

[[ $SMOKE -eq 1 ]] && EPISODES=1

# Train one seed into $1. If training crashes (out of GPU memory, a killed
# container, NaNs), retry up to $RETRIES attempts in all: resume from the last
# checkpoint (saved every 5 updates by default) and --finish the original
# target, or start over if no checkpoint was written yet.
train_one() {
  local run="$1" seed="$2" extra=() attempt=1 status
  mkdir -p "$run"
  if [[ $RESUME -eq 1 ]]; then
    [[ -f "$run/checkpoint.pkl" ]] || { echo "train.sh: no $run/checkpoint.pkl to resume" >&2; exit 1; }
    extra=(--resume "$run/checkpoint.pkl")
    [[ $FINISH -eq 1 ]] && extra+=(--finish)
  else
    extra=(--seed "$seed")
  fi
  echo "== [1/3] training seed $seed -> $run   (watch: tail -f $run/train.log)"
  while true; do
    status=0
    in_container python -m rl.train --output "$run" "${extra[@]}" \
      ${TRAIN_ARGS[@]+"${TRAIN_ARGS[@]}"} 2>&1 | tee -a "$run/train.log" || status=$?
    [[ $status -eq 0 ]] && return 0
    # Stopped on purpose (Ctrl-C / SIGTERM, or --stop): no retry.
    if [[ $status -eq 130 || $status -eq 143 || -f "$STOP_FILE" ]]; then
      echo "== training stopped (exit $status); not retrying. Resume with: docker/train.sh --finish ${run#runs/}" | tee -a "$run/train.log"
      exit "$status"
    fi
    if (( attempt >= RETRIES )); then
      echo "== training failed (exit $status) on attempt $attempt of $RETRIES; giving up" | tee -a "$run/train.log"
      exit "$status"
    fi
    attempt=$((attempt + 1))
    if [[ -f "$run/checkpoint.pkl" ]]; then
      extra=(--resume "$run/checkpoint.pkl" --finish)
      echo "== training failed (exit $status); retry $attempt of $RETRIES from the last checkpoint in 30 s" | tee -a "$run/train.log"
    else
      echo "== training failed (exit $status) before a checkpoint; retry $attempt of $RETRIES from scratch in 30 s" | tee -a "$run/train.log"
    fi
    sleep 30
    [[ -f "$STOP_FILE" ]] && exit 130  # stopped during the pause
  done
}

# Background workers ignore Ctrl-C (bash does that to `&` jobs), so this shell
# catches it: mark the stop, end the workers and kill this job's containers.
STOP_FILE=$(mktemp -u "${TMPDIR:-/tmp}/paramotor-stop.$$.XXXX")
PIDS=()
on_stop() {
  trap - INT TERM
  echo; echo "== stopping: killing this job's containers (no retry)"
  touch "$STOP_FILE"
  stop_jobs "$$"
  ((${#PIDS[@]})) && kill -TERM "${PIDS[@]}" 2>/dev/null || true
  ((${#PIDS[@]})) && wait "${PIDS[@]}" 2>/dev/null || true
  stop_jobs "$$"
  rm -f "$STOP_FILE"
  exit 130
}
trap on_stop INT TERM

# One worker per GPU, at most one per seed; each trains its seeds in turn.
WORKERS=$(( SEEDS < ${#GPUS[@]} ? SEEDS : ${#GPUS[@]} ))
(( WORKERS >= 1 )) || WORKERS=1
gpu_lock "${GPUS[@]:0:$WORKERS}"  # before the watcher, so a refused run leaves no window behind
if [[ $HEADED -eq 1 ]]; then
  # The watcher window runs beside training in its own container and stays
  # open after training ends; close the window to stop it.
  mkdir -p "runs/$NAME"
  if [[ $VERBOSE -eq 1 ]]; then  # watcher messages in this terminal too, prefixed
    ( release_locks; use_display; WATCHER=1 in_container python -m viewer.watch_training "runs/$NAME" 2>&1 \
        | tee "runs/$NAME/watch.log" | sed -u 's/^/[watch] /' ) &
  else
    ( release_locks; use_display; WATCHER=1 in_container python -m viewer.watch_training "runs/$NAME" \
        > "runs/$NAME/watch.log" 2>&1 ) &
  fi
  echo "== watcher window started (log: runs/$NAME/watch.log); it opens after the first checkpoint"
fi

RUNS=()
if [[ $SEEDS -le 1 ]]; then
  RUNS=("runs/$NAME")
else
  for ((k = 0; k < SEEDS; k++)); do RUNS+=("runs/$NAME/seed$k"); done
fi
# Worker w trains seeds w, w+WORKERS, ... on GPU ${GPUS[w]}, each with its own
# retries. With several workers, terminal lines are prefixed [gpuN].
for ((w = 0; w < WORKERS; w++)); do
  (
    GPU_DEVICE="${GPUS[w]}"
    for ((k = w; k < ${#RUNS[@]}; k += WORKERS)); do
      if [[ $WORKERS -gt 1 ]]; then
        train_one "${RUNS[k]}" "$((SEED + k))" 2>&1 | sed -u "s/^/[gpu$GPU_DEVICE] /"
        [[ ${PIPESTATUS[0]} -eq 0 ]] || exit 1
      else
        train_one "${RUNS[k]}" "$((SEED + k))"
      fi
    done
  ) &
  PIDS+=($!)
done
FAILED=0
for pid in "${PIDS[@]}"; do wait "$pid" || FAILED=1; done
# Evaluate whatever finished; a seed that gave up has no new checkpoint.
DONE=()
for run in "${RUNS[@]}"; do [[ -f "$run/checkpoint.pkl" ]] && DONE+=("$run"); done
[[ ${#DONE[@]} -gt 0 ]] || { echo "== no run produced a checkpoint" >&2; exit 1; }
[[ $FAILED -eq 1 ]] && echo "== some seeds failed after retries; evaluating the ${#DONE[@]} with checkpoints"
RUNS=("${DONE[@]}")

# All seeds fly the same flights in one batch: about the cost of evaluating one.
echo "== [2/3] evaluating ${#RUNS[@]} checkpoint(s), $EPISODES flights per route each -> <run>/eval"
in_container python -m rl.evaluate "${RUNS[@]/%//checkpoint.pkl}" --episodes "$EPISODES" \
  2>&1 | tee "runs/$NAME/eval.log"
for run in "${RUNS[@]}"; do
  echo "== [3/3] analysis -> $run/analysis"
  NO_GPU=1 in_container python -m rl.analyze "$run" | tail -n 1
done
[[ $SEEDS -gt 1 && $FAILED -eq 0 ]] && echo "== all $SEEDS seeds done. Compare configs with: docker/train.sh --compare $NAME <other names>"
exit $FAILED
