#!/usr/bin/env bash
#SBATCH --job-name=fortbo-landreman
#SBATCH --nodes=1
#SBATCH --ntasks=5
#SBATCH --cpus-per-task=13
#SBATCH --gpus-per-node=4
#SBATCH --time=03:00:00

# This job is the archived Landreman control only.  It must be submitted on a
# Tu Graz Slurm cluster, never executed on the FortBO workstation.  Override
# partition/account/GRES options at submission time when the target cluster
# requires them, for example: sbatch --partition=<p> --account=<a> <this-file>.

set -euo pipefail

FORTBO_ROOT=${FORTBO_ROOT:?set FORTBO_ROOT to the clean FortBO checkout}
PYTHON=${LANDREMAN_PYTHON:-${FORTBO_ROOT}/.venv/bin/python}
SOURCE_ROOT=${FORTBO_REPRODUCTION_SOURCE_ROOT:-/var/tmp/ert/fortbo-reproduction-sources}
LANDREMAN_ARCHIVE=${LANDREMAN_ARCHIVE:-${SOURCE_ROOT}/landreman/20260514-01-zenodo_for_data_informed_spaces_paper.20260617.tar}
RUN_ROOT=${LANDREMAN_RUN_ROOT:-/var/tmp/ert/landreman-reproduction/${SLURM_JOB_ID}}
METADATA_ROOT=${LANDREMAN_METADATA_ROOT:-${RUN_ROOT}.metadata}

test -d "$FORTBO_ROOT"
test -x "$PYTHON"
test -n "${SLURM_JOB_ID:-}"
test "$(git -C "$FORTBO_ROOT" status --porcelain)" = ""
test "$(git -C "$FORTBO_ROOT" rev-parse --is-inside-work-tree)" = true

export LANDREMAN_ARCHIVE
mkdir -p "$SOURCE_ROOT" "$METADATA_ROOT"

scontrol show job -dd "$SLURM_JOB_ID" > "$METADATA_ROOT/slurm-job.txt"
lscpu > "$METADATA_ROOT/lscpu.txt"
nvidia-smi -L > "$METADATA_ROOT/nvidia-smi.txt"
git -C "$FORTBO_ROOT" rev-parse HEAD > "$METADATA_ROOT/fortbo-commit.txt"
"$PYTHON" -VV > "$METADATA_ROOT/python.txt" 2>&1

"$PYTHON" "$FORTBO_ROOT/scripts/fetch_reproduction_sources.py" \
    --manifest "$FORTBO_ROOT/configs/reproduction/source-downloads.json" \
    --root "$SOURCE_ROOT" --artifact landreman_archive \
    --min-free-gib 8

"$PYTHON" "$FORTBO_ROOT/scripts/run_landreman_original.py" \
    --config "$FORTBO_ROOT/configs/reproduction/landreman-data-informed-turbo.json" \
    --archive "$LANDREMAN_ARCHIVE" --run-root "$RUN_ROOT" \
    --python "$PYTHON" --execute
