#!/usr/bin/env bash
#SBATCH --job-name=fortbo-landreman
#SBATCH --nodes=1
#SBATCH --ntasks=5
#SBATCH --cpus-per-task=13
#SBATCH --gpus-per-node=4
#SBATCH --time=03:00:00

# This job is the FortBO side of the archived Landreman comparison.  It is
# intentionally separate from the archived-control wrapper: both use the same
# historical allocation, but only this job runs the FortBO MPI bridge.

set -euo pipefail

FORTBO_ROOT=${FORTBO_ROOT:?set FORTBO_ROOT to the clean FortBO checkout}
PYTHON=${LANDREMAN_PYTHON:-${FORTBO_ROOT}/.venv/bin/python}
SOURCE_ROOT=${FORTBO_REPRODUCTION_SOURCE_ROOT:-/var/tmp/ert/fortbo-reproduction-sources}
LANDREMAN_ARCHIVE=${LANDREMAN_ARCHIVE:-${SOURCE_ROOT}/landreman/20260514-01-zenodo_for_data_informed_spaces_paper.20260617.tar}
PREP_ROOT=${LANDREMAN_PREP_ROOT:-/var/tmp/ert/landreman-reproduction/${SLURM_JOB_ID}-fortbo}
SCRATCH=${LANDREMAN_FORTBO_SCRATCH:-${PREP_ROOT}/fortbo-scratch}
OUTPUT=${LANDREMAN_FORTBO_OUTPUT:-${PREP_ROOT}/fortbo.json}
METADATA_ROOT=${LANDREMAN_FORTBO_METADATA_ROOT:-${PREP_ROOT}.metadata}
FORTBO_COMMAND=${FORTBO_COMMAND:-${HOME}/.local/bin/fo}
FO_CACHE_DIR=${FORTBO_FO_CACHE:-${PREP_ROOT}/fo-cache}

test -d "$FORTBO_ROOT"
test -x "$PYTHON"
test -n "${SLURM_JOB_ID:-}"
test "$(git -C "$FORTBO_ROOT" status --porcelain)" = ""
test "$(git -C "$FORTBO_ROOT" rev-parse --is-inside-work-tree)" = true
test ! -e "$OUTPUT"

export LANDREMAN_ARCHIVE FO_CACHE_DIR
mkdir -p "$SOURCE_ROOT" "$METADATA_ROOT"

scontrol show job -dd "$SLURM_JOB_ID" > "$METADATA_ROOT/slurm-job.txt"
lscpu > "$METADATA_ROOT/lscpu.txt"
nvidia-smi -L > "$METADATA_ROOT/nvidia-smi.txt"
git -C "$FORTBO_ROOT" rev-parse HEAD > "$METADATA_ROOT/fortbo-commit.txt"
"$PYTHON" -VV > "$METADATA_ROOT/python.txt" 2>&1

"$PYTHON" "$FORTBO_ROOT/scripts/fortbo_environment.py" \
    --fortbo-root "$FORTBO_ROOT" \
    --fortbo-command "$FORTBO_COMMAND" \
    --test > "$METADATA_ROOT/fortbo-environment.json"

"$PYTHON" "$FORTBO_ROOT/scripts/fetch_reproduction_sources.py" \
    --manifest "$FORTBO_ROOT/configs/reproduction/source-downloads.json" \
    --root "$SOURCE_ROOT" --artifact landreman_archive \
    --min-free-gib 8 --check-only

"$PYTHON" "$FORTBO_ROOT/scripts/run_landreman_original.py" \
    --config "$FORTBO_ROOT/configs/reproduction/landreman-data-informed-turbo.json" \
    --archive "$LANDREMAN_ARCHIVE" --run-root "$PREP_ROOT" \
    --python "$PYTHON"

srun --ntasks=5 --cpus-per-task=13 --cpu-bind=cores \
    --gpu-bind=map_gpu:0,1,2,3 "$PYTHON" \
    "$FORTBO_ROOT/scripts/run_landreman_fortbo.py" \
    --alpha-root "$PREP_ROOT/software/alpha_opt" \
    --archive "$LANDREMAN_ARCHIVE" \
    --scratch "$SCRATCH" --output "$OUTPUT" \
    --fortbo-command "$FORTBO_COMMAND" --budget 10000 --initial 40 --seed 0 \
    --workers 4
