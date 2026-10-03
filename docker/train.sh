#!/bin/bash
# Train, evaluate and analyze a PPO policy inside the Docker image.
#
#   docker/train.sh                                  # default run, named ppo-<timestamp>
#   docker/train.sh --name first --updates 2000 --num-envs 256
#   docker/train.sh --name first --resume --updates 1000   # continue runs/first
#   docker/train.sh --smoke                          # minute-long end-to-end check
#   docker/train.sh --analyze first                  # re-plot runs/first (works mid-training)
#   docker/train.sh --cpu --smoke                    # no GPU: CPU image, no --gpus
#
# Any option it does not recognise goes straight to `python -m rl.train`
# (see `python -m rl.train --help`; --config file.json takes env/ppo overrides).
#
# Everything lands in runs/<name>/ on the host:
#   config.json, metrics.csv, eval.csv, checkpoint.pkl, train.log   (training)
#   eval/summary.csv, eval/flights.csv                               (evaluation)
#   analysis/{training,curriculum,evaluation}.png, analysis/report.md
#
# The image is rebuilt only when rl/requirements-rl.txt changes (Docker layer
# cache); code is mounted, not copied, so edits apply immediately.
# Set JAX_CUDA=cuda12 for drivers older than CUDA 13.
set -euo pipefail
cd "$(dirname "$0")/.."

NAME="ppo-$(date +%Y%m%d-%H%M%S)"
CPU=0 RESUME=0 SMOKE=0 ANALYZE_ONLY="" EPISODES=3
TRAIN_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --name)      NAME="$2"; shift 2 ;;
    --cpu)       CPU=1; shift ;;
    --resume)    RESUME=1; shift ;;
    --smoke)     SMOKE=1; TRAIN_ARGS+=("$1"); shift ;;
    --analyze)   ANALYZE_ONLY="$2"; shift 2 ;;
    --episodes)  EPISODES="$2"; shift 2 ;;
    -h|--help)   sed -n '2,24p' "$0"; exit 0 ;;
    *)           TRAIN_ARGS+=("$1"); shift ;;
  esac
done

if [[ $CPU -eq 1 ]]; then JAX_CUDA=cpu; else JAX_CUDA="${JAX_CUDA:-cuda13}"; fi
IMAGE="paramotor-rl:$JAX_CUDA"

docker build -q -f docker/Dockerfile -t "$IMAGE" --build-arg JAX_CUDA="$JAX_CUDA" . >/dev/null

# Run as the host user so runs/ is not owned by root. HOME and MPLCONFIGDIR
# point at /tmp because that user has no home directory in the image.
in_container() {
  local gpu=()
  [[ $CPU -eq 0 && "${NO_GPU:-0}" -eq 0 ]] && gpu=(--gpus all)
  docker run --rm ${gpu[@]+"${gpu[@]}"} --ipc=host \
    -u "$(id -u):$(id -g)" -e HOME=/tmp -e MPLCONFIGDIR=/tmp \
    -v "$PWD:/workspace" -w /workspace "$IMAGE" "$@"
}

if [[ -n "$ANALYZE_ONLY" ]]; then
  NO_GPU=1 in_container python -m rl.analyze "runs/$ANALYZE_ONLY"
  exit 0
fi

RUN="runs/$NAME"
mkdir -p "$RUN"
if [[ $RESUME -eq 1 ]]; then
  [[ -f "$RUN/checkpoint.pkl" ]] || { echo "train.sh: no $RUN/checkpoint.pkl to resume" >&2; exit 1; }
  TRAIN_ARGS+=(--resume "$RUN/checkpoint.pkl")
fi
[[ $SMOKE -eq 1 ]] && EPISODES=1

echo "== [1/3] training -> $RUN   (watch: tail -f $RUN/train.log; plots: docker/train.sh --analyze $NAME)"
in_container python -m rl.train --output "$RUN" ${TRAIN_ARGS[@]+"${TRAIN_ARGS[@]}"} 2>&1 | tee -a "$RUN/train.log"

echo "== [2/3] evaluating on fixed routes -> $RUN/eval"
in_container python -m rl.evaluate "$RUN/checkpoint.pkl" --episodes "$EPISODES" --output "$RUN/eval" 2>&1 | tee "$RUN/eval.log"

echo "== [3/3] analysis -> $RUN/analysis"
NO_GPU=1 in_container python -m rl.analyze "$RUN"
