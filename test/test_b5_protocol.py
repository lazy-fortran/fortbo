"""Exercise the external-evaluator protocol with an independent oracle."""

from __future__ import annotations

import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_b5_protocol_round_trips_points_and_failures() -> None:
    process = subprocess.Popen(
        ["fo", "exec", "fortbo_b5_protocol", "2", "4", "4", "7", "1", "2"],
        cwd=ROOT,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    assert process.stdin is not None
    assert process.stdout is not None
    values: list[float] = []
    batch_number = 0
    try:
        while True:
            line = _next_protocol_line(process.stdout)
            if line.startswith("DONE "):
                break
            _, count_text = line.split()
            count = int(count_text)
            points = []
            for _ in range(count):
                parts = process.stdout.readline().split()
                assert parts[0] == "POINT"
                point = [float(value) for value in parts[3:]]
                assert len(point) == 2
                assert all(0.0 <= value <= 1.0 for value in point)
                points.append(point)
            process.stdin.write(f"TELL {count}\n")
            for index, point in enumerate(points, 1):
                if batch_number == 0 and index == 1:
                    process.stdin.write(f"FAIL {index}\n")
                else:
                    value = sum((coordinate - 0.25) ** 2 for coordinate in point)
                    values.append(value)
                    process.stdin.write(f"VALUE {index} {value:.17g}\n")
            process.stdin.flush()
            batch_number += 1
    finally:
        process.stdin.close()
    stderr = process.stderr.read() if process.stderr is not None else ""
    assert process.wait(timeout=60) == 0, stderr
    assert len(values) == 3


def _next_protocol_line(stream) -> str:
    while True:
        line = stream.readline()
        assert line, "FortBO protocol ended unexpectedly"
        stripped = line.strip()
        if stripped.startswith(("ASK ", "DONE ")):
            return stripped
