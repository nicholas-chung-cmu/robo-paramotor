#!/bin/bash
# Train, evaluate and analyze PPO policies inside the Docker image.
#
#   docker/train.sh                                  # one run, named ppo-<timestamp>
#   docker/train.sh --name first --updates 2000 --num-envs 256
#   docker/train.sh --name first --resume --updates 1000   # continue runs/first
#   docker/train.sh --name baseline --seeds 5 --updates 500  # runs/baseline/seed0..4
#   docker/train.sh --compare baseline high_lr       # rliable comparison of configs
#   docker/train.sh --smoke                          # minute-long end-to-end check
#   docker/train.sh --analyze first                  # re-plot runs/first (works mid-training)
#   docker/train.sh --cpu --smoke                    # no GPU: CPU image, no --gpus
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

NAME="ppo-$(date +%Y%m%d-%H%M%S)"
CPU=0 RESUME=0 SMOKE=0 SEEDS=1 SEED=0 EPISODES=3 MODE=train
TARGETS=() TRAIN_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --name)      NAME="$2"; shift 2 ;;
    --cpu)       CPU=1; shift ;;
    --resume)    RESUME=1; shift ;;
    --smoke)     SMOKE=1; TRAIN_ARGS+=("$1"); shift ;;
    --seeds)     SEEDS="$2"; shift 2 ;;
    --seed)      SEED="$2"; shift 2 ;;
    --episodes)  EPISODES="$2"; shift 2 ;;
    --analyze)   MODE=analyze; TARGETS+=("$2"); shift 2 ;;
    --compare)   MODE=compare; shift
                 while [[ $# -gt 0 && "$1" != --* ]]; do TARGETS+=("$1"); shift; done ;;
    -h|--help)   sed -n '2,28p' "$0"; exit 0 ;;
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

# The run directories behind a name: runs/<name>/seed* if present, else runs/<name>.
run_dirs() {
  local seeds=("runs/$1"/seed*/)
  if [[ -d "${seeds[0]}" ]]; then printf '%s\n' "${seeds[@]%/}"; else echo "runs/$1"; fi
}

case "$MODE" in
  analyze)
    for dir in $(run_dirs "${TARGETS[0]}"); do NO_GPU=1 in_container python -m rl.analyze "$dir"; done
    exit 0 ;;
  compare)
    [[ ${#TARGETS[@]} -ge 1 ]] || { echo "train.sh: --compare needs run names" >&2; exit 1; }
    NO_GPU=1 in_container python -m rl.compare "${TARGETS[@]/#/runs/}"
    exit 0 ;;
esac

[[ $SMOKE -eq 1 ]] && EPISODES=1

# Train -> evaluate -> analyze one seed into $1.
pipeline() {
  local run="$1" seed="$2" extra=()
  mkdir -p "$run"
  if [[ $RESUME -eq 1 ]]; then
    [[ -f "$run/checkpoint.pkl" ]] || { echo "train.sh: no $run/checkpoint.pkl to resume" >&2; exit 1; }
    extra=(--resume "$run/checkpoint.pkl")
  else
    extra=(--seed "$seed")
  fi
  echo "== [1/3] training seed $seed -> $run   (watch: tail -f $run/train.log)"
  in_container python -m rl.train --output "$run" "${extra[@]}" \
    ${TRAIN_ARGS[@]+"${TRAIN_ARGS[@]}"} 2>&1 | tee -a "$run/train.log"
  echo "== [2/3] evaluating on fixed routes -> $run/eval"
  in_container python -m rl.evaluate "$run/checkpoint.pkl" --episodes "$EPISODES" \
    --output "$run/eval" 2>&1 | tee "$run/eval.log"
  echo "== [3/3] analysis -> $run/analysis"
  NO_GPU=1 in_container python -m rl.analyze "$run" | tail -n 1
}

if [[ $SEEDS -le 1 ]]; then
  pipeline "runs/$NAME" "$SEED"
else
  for ((k = 0; k < SEEDS; k++)); do pipeline "runs/$NAME/seed$k" "$((SEED + k))"; done
  echo "== all $SEEDS seeds done. Compare configs with: docker/train.sh --compare $NAME <other names>"
fi
