from __future__ import annotations

"""Cross-platform execution helpers for bounded real parallelism.

CPU-heavy work uses ``ProcessPoolExecutor``.  File reads/writes remain in the
parent process or use threads only when the operation is predominantly I/O.
The worker initializer limits native numerical thread pools to one thread per
process so a process pool does not oversubscribe the machine.
"""

import os
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from typing import Any, Callable


def initialise_process_worker() -> None:
    """Limit native BLAS/OpenMP pools inside each worker process."""
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ[name] = "1"
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(limits=1)
    except Exception:
        pass


class LazyProcessPool:
    """Context-managed process pool created only on the first submitted task.

    Entering the context does not create worker processes. This is important
    for workflows that perform a serial streaming/I/O stage before CPU-heavy
    tile work: the serial stage remains a single Python process and worker
    processes are spawned only when the first parallel task is actually ready.
    """

    def __init__(self, max_workers: int):
        self.max_workers = max(1, int(max_workers))
        self._executor: ProcessPoolExecutor | None = None

    @property
    def started(self) -> bool:
        return self._executor is not None

    def _ensure_executor(self) -> ProcessPoolExecutor:
        if self._executor is None:
            ctx = get_context("spawn")
            self._executor = ProcessPoolExecutor(
                max_workers=self.max_workers,
                mp_context=ctx,
                initializer=initialise_process_worker,
            )
        return self._executor

    def submit(self, fn: Callable[..., Any], /, *args: Any, **kwargs: Any):
        return self._ensure_executor().submit(fn, *args, **kwargs)

    def shutdown(self, wait: bool = True, *, cancel_futures: bool = False) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=wait, cancel_futures=cancel_futures)
            self._executor = None

    def __enter__(self) -> "LazyProcessPool":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.shutdown(wait=True, cancel_futures=exc_type is not None)


def process_pool(max_workers: int) -> LazyProcessPool:
    """Return a lazy, cross-platform process pool.

    ``spawn`` is used when the first task is submitted. No child process exists
    merely because a parallel stage was prepared or a context was entered.
    """
    return LazyProcessPool(max_workers)


def window_to_tuple(window: Any) -> tuple[int, int, int, int]:
    """Convert a rasterio Window-like object to a simple pickle-safe tuple."""
    return (
        int(window.col_off),
        int(window.row_off),
        int(window.width),
        int(window.height),
    )
