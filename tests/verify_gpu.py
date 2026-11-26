"""Check the GPU executable against an independent CPU reference."""

import argparse
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np


def cpu_reference(grid, passes, wrap_x):
    """Scatter a donor's four shares into the next buffer on the CPU."""
    height, width = grid.shape
    for _ in range(passes):
        result = np.minimum(grid, 1)
        for y, x in zip(*np.nonzero(grid > 1)):
            share = float(grid[y, x] - 1) / 4
            for dy, dx in ((0, -1), (0, 1), (-1, 0), (1, 0)):
                ny, nx = y + dy, x + dx
                if wrap_x:
                    nx %= width
                if 0 <= ny < height and 0 <= nx < width and grid[ny, nx] >= 0:
                    result[ny, nx] += share
                else:
                    result[y, x] += share
        grid = result
    return grid


def run_gpu(binary, grid, passes, wrap_x, directory):
    height, width = grid.shape
    source = directory / "input.f32"
    target = directory / "output.f32"
    grid.astype("<f4", copy=False).tofile(source)
    command = [binary, str(width), str(height), source, target, str(passes)]
    if wrap_x:
        command.append("--wrap-x")
    start = perf_counter()
    subprocess.run(command, check=True)
    elapsed = perf_counter() - start
    return np.fromfile(target, dtype="<f4").reshape((height, width)), elapsed


def mass(grid):
    return np.maximum(grid, 0).sum(dtype=np.float64)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("binary", type=lambda path: Path(path).resolve())
    parser.add_argument("--benchmark", action="store_true")
    args = parser.parse_args()

    coast = np.array([[5, -1, 0], [0, 0, -1], [-1, -1, 3]], dtype=np.float32)
    many_blocks = np.random.default_rng(73).uniform(0, 5, (35, 67)).astype(np.float32)
    many_blocks[::4, ::3] = -1
    cases = [
        (coast, 0, False),
        (coast, 1, False),
        (coast, 5, True),
        (coast[:, :1], 5, True),
        (coast[:1, :], 5, False),
        (many_blocks, 7, False),
        (many_blocks, 8, True),
    ]
    with TemporaryDirectory() as tmp:
        directory = Path(tmp)
        for grid, passes, wrap in cases:
            result, _ = run_gpu(args.binary, grid, passes, wrap, directory)
            np.testing.assert_array_equal(result < 0, grid < 0)
            expected = cpu_reference(grid, passes, wrap)
            np.testing.assert_allclose(result, expected, atol=2e-6, rtol=2e-6)
            np.testing.assert_allclose(mass(result), mass(grid), atol=1e-4, rtol=1e-6)
        print(f"All {len(cases)} CPU and GPU comparisons passed.")

        spike = np.zeros((95, 97), dtype=np.float32)
        spike[47, 48] = 1000
        result, _ = run_gpu(args.binary, spike, 4096, False, directory)
        drift = mass(result) - mass(spike)
        np.testing.assert_allclose(drift, 0, atol=1e-5)
        print(f"Mass drift after 4096 passes: {drift:+.3g} marks")

        source = directory / "overflow.f32"
        np.array([2.2e9, 2.2e9], dtype="<f4").tofile(source)
        overflow_output = directory / "overflow-out.f32"
        failed = subprocess.run(
            [args.binary, "2", "1", source, overflow_output, "1"],
            capture_output=True,
            text=True,
        )
        assert failed.returncode and not overflow_output.exists(), failed

        if args.benchmark:
            grid = np.zeros((4333, 10000), dtype=np.float32)
            centre_y, centre_x = (size // 2 for size in grid.shape)
            grid[centre_y, centre_x] = 1000
            result, elapsed = run_gpu(args.binary, grid, 32, False, directory)
            window = np.s_[centre_y - 32 : centre_y + 33, centre_x - 32 : centre_x + 33]
            expected = cpu_reference(grid[window], 32, False)
            np.testing.assert_allclose(result[window], expected, atol=2e-6, rtol=2e-6)
            np.testing.assert_allclose(mass(result), mass(grid), atol=1e-4, rtol=1e-6)
            print(f"Full process and file I/O: {elapsed * 1000:.1f} ms")


if __name__ == "__main__":
    main()
