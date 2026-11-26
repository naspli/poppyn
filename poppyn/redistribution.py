"""Redistribute normalised counts, audit the mass, and allocate whole marks."""

from dataclasses import dataclass
import math
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
from time import monotonic

import numpy as np


MASKED = -1.0
STALL_PASSES = 4_096
MIN_OVERFLOW_IMPROVEMENT = 0.0001  # In bright pixels' worth of people.


def gather(grid, wrap_x=False):
    """One synchronous pass using the same stencil as the Mojo kernel."""
    land = grid >= 0
    share = np.maximum(grid - 1, 0) * 0.25
    north = np.empty_like(grid)
    north[0] = share[0]
    north[1:] = np.where(land[:-1], share[:-1], share[1:])
    south = np.empty_like(grid)
    south[-1] = share[-1]
    south[:-1] = np.where(land[1:], share[1:], share[:-1])
    west = np.empty_like(grid)
    west[:, 0] = np.where(land[:, -1], share[:, -1], share[:, 0]) if wrap_x else share[:, 0]
    west[:, 1:] = np.where(land[:, :-1], share[:, :-1], share[:, 1:])
    east = np.empty_like(grid)
    east[:, -1] = np.where(land[:, 0], share[:, 0], share[:, -1]) if wrap_x else share[:, -1]
    east[:, :-1] = np.where(land[:, 1:], share[:, 1:], share[:, :-1])
    incoming = ((west + east) + north) + south
    return np.where(land, np.minimum(grid, 1) + incoming, MASKED)


@dataclass
class Settlement:
    grid: np.ndarray
    passes: int
    seconds: float
    overflow: float
    mass_drift: float
    conversion_drift: float


def measure(grid, land):
    mass = np.sum(grid, dtype=np.float64, where=land)
    overflow = np.maximum(grid - 1, 0).sum(dtype=np.float64)
    return float(mass), float(overflow)


def gpu_passes(grid, gpu, passes, wrap_x, temporary_dir):
    source = temporary_dir / "source.f32"
    output = temporary_dir / "settled.f32"
    grid.tofile(source)
    output.unlink(missing_ok=True)
    height, width = grid.shape
    command = [str(gpu.resolve()), str(width), str(height), str(source), str(output), str(passes)]
    if wrap_x:
        command.append("--wrap-x")
    subprocess.run(command, check=True, stdout=subprocess.DEVNULL)
    if output.stat().st_size != grid.nbytes:
        raise ValueError("GPU returned a file with the wrong number of cells")
    return np.fromfile(output, dtype="<f4").reshape(grid.shape)


def settle(counts, land, people_per_mark, *, gpu=None, wrap_x=False,
           batch_passes=256, max_passes=None, overflow_tolerance=0.20,
           drift_tolerance=0.20, progress=None):
    """Spread until the excess is small enough, or until measurable progress stalls."""
    if (counts.ndim != 2 or counts.shape != land.shape or counts.size == 0
            or not np.isfinite(counts).all() or np.any(counts < 0)
            or np.any((~land) & (counts != 0)) or not 0 < people_per_mark < math.inf):
        raise ValueError("expected a finite non-negative population grid and a matching land mask")
    if (batch_passes < 1 or (max_passes is not None and max_passes < 0)
            or min(overflow_tolerance, drift_tolerance) < 0
            or not 0 <= overflow_tolerance + drift_tolerance < 0.5):
        raise ValueError("invalid pass limits or combined population tolerance; keep it below half a mark")
    target_mass = float(counts.sum(dtype=np.float64) / people_per_mark)
    if math.floor(target_mass + 0.5) > land.sum():
        raise ValueError("the eligible cells cannot represent this population at the requested scale")
    grid = np.where(land, counts / people_per_mark, MASKED).astype("<f4" if gpu else np.float64)
    initial_mass, overflow = measure(grid, land)
    if abs(initial_mass - target_mass) > drift_tolerance:
        change = (initial_mass - target_mass) * people_per_mark
        raise ValueError(f"preparing the grid changed its population total by {change:+,.2f} people, "
                         "exceeding the allowed rounding error")
    mass = initial_mass
    passes = 0
    progress_overflow = overflow
    last_progress = 0
    start = monotonic()
    with TemporaryDirectory(prefix="poppyn-grid-") as directory:
        while overflow > overflow_tolerance and (max_passes is None or passes < max_passes):
            batch = batch_passes if max_passes is None else min(batch_passes, max_passes - passes)
            if gpu:
                grid = gpu_passes(grid, Path(gpu), batch, wrap_x, Path(directory))
                if not np.isfinite(grid).all() or np.any(np.where(land, grid < 0, grid != MASKED)):
                    raise ValueError("GPU returned non-finite population or changed the land mask")
            else:
                for _ in range(batch):
                    grid = gather(grid, wrap_x)
            passes += batch
            mass, overflow = measure(grid, land)
            if progress is not None:
                progress(passes, overflow, mass - target_mass)
            if abs(mass - target_mass) > drift_tolerance:
                break
            # Keep the reference until reductions add up to a measurable improvement.
            if progress_overflow - overflow >= MIN_OVERFLOW_IMPROVEMENT:
                progress_overflow, last_progress = overflow, passes
            elif passes - last_progress >= STALL_PASSES:
                break
    if abs(mass - target_mass) > drift_tolerance:
        change = (mass - target_mass) * people_per_mark
        raise ValueError(f"the population total changed by {change:+,.2f} people after {passes:,} rounds, "
                         "exceeding the allowed rounding error")
    if overflow > overflow_tolerance:
        remaining = f"about {overflow * people_per_mark:,.2f} people still above pixel capacity"
        if max_passes is not None and passes >= max_passes:
            raise ValueError(f"reached the requested limit of {passes:,} rounds with {remaining}. "
                             "Increase --max-passes or omit it to continue while excess population is falling.")
        raise ValueError(f"excess population stalled for {passes - last_progress:,} rounds, with {remaining}. "
                         "Check for an overfull land component such as an island, or use a finer grid.")
    return Settlement(grid, passes, monotonic() - start, overflow, mass - target_mass,
                      initial_mass - target_mass)


def child_blocks(grid):
    """Group child cells in 2 × 2 blocks, with zero padding along odd edges."""
    height, width = grid.shape
    if height % 2 or width % 2:
        grid = np.pad(grid, ((0, height % 2), (0, width % 2)))
    return grid.reshape((height + 1) // 2, 2, (width + 1) // 2, 2).transpose(0, 2, 1, 3)


def round_marks(settled, land, budget):
    """Allocate an exact budget while rounding every 2 × 2 subtree locally."""
    totals = [np.where(land, np.minimum(settled, 1), 0).astype(np.float64)]
    while totals[-1].shape != (1, 1):
        totals.append(child_blocks(totals[-1]).sum(axis=(2, 3)))
    total = totals[-1].item()
    if budget not in (math.floor(total), math.ceil(total)):
        raise ValueError(f"mark budget {budget:,} cannot represent {total:,.6f} settled marks")
    quotas = np.array([[budget]], dtype=np.int32)
    # Deterministic ties avoid consistently favouring one corner of each block.
    rng = np.random.default_rng(0)
    for counts in reversed(totals[:-1]):
        height, width = counts.shape
        rows, columns = quotas.shape
        children = child_blocks(counts).reshape(rows, columns, 4)
        floors = np.floor(children).astype(np.int32)
        fractions = children - floors
        extras = quotas - floors.sum(axis=-1)
        if np.any(extras < 0) or np.any(extras > (fractions > 0).sum(axis=-1)):
            raise ValueError("a block's mark quota exceeds the available fractional cells")
        order = np.lexsort((rng.random(fractions.shape, dtype=np.float32), -fractions), axis=-1)
        extras_in_order = np.arange(4) < extras[..., None]
        add = np.empty(children.shape, dtype=bool)
        np.put_along_axis(add, order, extras_in_order, axis=-1)
        quotas = (floors + add).reshape(rows, columns, 2, 2).transpose(0, 2, 1, 3)
        quotas = quotas.reshape(rows * 2, columns * 2)[:height, :width]
    return quotas.astype(bool)
