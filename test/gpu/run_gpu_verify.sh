#!/usr/bin/env bash
# Build and run the OpenACC device kernels against a real GPU.
#
# This is not part of `fo test`, and the reason is a compiler defect rather
# than a choice. A full nvfortran build of the stack fails with internal
# compiler errors in FortAD -- "Deferred-length character symbol must have
# descriptor", and a segfault in fort1 -- so the whole library cannot be built
# with OpenACC today. `fortbo_device` depends only on FortNum's kinds and
# status, so it compiles standalone and the kernels can still be verified.
#
# Under gfortran, `_OPENACC` is undefined and the device branch compiles out:
# the in-tree `test_device` then exercises only the refusal path. It says so in
# its output, but it cannot prove the kernels work because it never builds
# them. This does.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root="$(cd "$here/../.." && pwd)"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

if ! command -v nvfortran >/dev/null; then
    echo "nvfortran not found; GPU verification needs the NVIDIA HPC SDK" >&2
    exit 2
fi

find "$root/../fortnum/src" -name fortnum_kinds.f90 -exec cp {} "$work" \;
find "$root/../fortnum/src" -name fortnum_status.f90 -exec cp {} "$work" \;
cp "$root/src/fortbo_device.F90" "$work"
cp "$here/gpu_verify.f90" "$work"

cd "$work"
flags="-acc -gpu=ccnative -O2"
nvfortran $flags -c fortnum_kinds.f90 fortnum_status.f90
nvfortran $flags -c fortbo_device.F90
nvfortran $flags -o gpu_verify gpu_verify.f90 fortnum_kinds.o fortnum_status.o fortbo_device.o
./gpu_verify
