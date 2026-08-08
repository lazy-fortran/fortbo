#!/usr/bin/env python3
"""Build the published Glas/Bindel periodic Fourier perturbation covariance.

The coefficient convention is the one in equations (3.3)--(3.5) of Glas et
al.: ``g(t) = c0/2 + sum(c_k cos(k t) + s_k sin(k t))`` and each Cartesian
coordinate is an independent periodic GP. The returned matrix repeats that
Cartesian block for each independent coil in one field period.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable, Optional

import numpy as np


def _modified_bessel_i(order: int, argument: float) -> float:
    """Evaluate I_order(argument) using its positive convergent series."""
    term = (argument / 2.0) ** order / math.factorial(order)
    total = term
    for index in range(1, 1000):
        term *= (argument * argument / 4.0) / (index * (order + index))
        total += term
        if abs(term) <= abs(total) * 2.0e-16:
            break
    else:
        raise RuntimeError("modified Bessel series did not converge")
    return total


def _coefficient_variances(modes: int, amplitude: float,
                           lengthscale: float) -> np.ndarray:
    h = amplitude * amplitude / 3.0
    argument = 1.0 / (lengthscale * lengthscale)
    scale = h * math.exp(-argument)
    cosine = [4.0 * scale * _modified_bessel_i(0, argument)]
    cosine.extend(
        2.0 * scale * _modified_bessel_i(order, argument)
        for order in range(1, modes + 1)
    )
    sine = cosine[1:]
    return np.asarray(cosine + sine, dtype=np.float64)


def periodic_fourier_covariance(
    modes: int,
    amplitude: float,
    lengthscale: float,
    coils: int = 1,
) -> np.ndarray:
    """Return the block-diagonal covariance for one-period coil coefficients.

    ``amplitude`` is the desired RMS norm of the three-coordinate perturbation
    in metres. ``coils`` repeats the independent x/y/z block; the resulting
    width is ``coils * 3 * (2*modes + 1)``.
    """
    if modes < 0 or coils < 1 or amplitude <= 0.0 or lengthscale <= 0.0:
        raise ValueError("modes/coils must be positive and scales must be nonzero")
    coordinate = np.diag(_coefficient_variances(modes, amplitude, lengthscale))
    coil_block = np.kron(np.eye(3, dtype=np.float64), coordinate)
    return np.kron(np.eye(coils, dtype=np.float64), coil_block)


def _write_json(path: Path, covariance: np.ndarray, amplitude: float,
                lengthscale: float, modes: int, coils: int) -> None:
    document = {
        "schema_version": 1,
        "source": "Glas et al. equations 3.3--3.5",
        "modes": modes,
        "coils": coils,
        "amplitude_m": amplitude,
        "lengthscale": lengthscale,
        "dimension": int(covariance.shape[0]),
        "covariance": covariance.tolist(),
    }
    path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modes", type=int, default=6)
    parser.add_argument("--coils", type=int, default=5)
    parser.add_argument("--amplitude-mm", type=float, required=True)
    parser.add_argument("--lengthscale", type=float, default=0.5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    covariance = periodic_fourier_covariance(
        args.modes, args.amplitude_mm * 1.0e-3, args.lengthscale, args.coils)
    _write_json(path=args.output, covariance=covariance,
                amplitude=args.amplitude_mm * 1.0e-3,
                lengthscale=args.lengthscale, modes=args.modes, coils=args.coils)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
