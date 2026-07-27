"""Tests for shared synthetic-data generation utilities."""

import pytest
import torch
from sampleworks.synthetic.synthetic_utils import resolve_parallel_jobs


@pytest.mark.parametrize("n_jobs", [-2, -1, 2, 8])
def test_resolve_parallel_jobs_serializes_cuda_work(n_jobs: int) -> None:
    """CUDA requests that imply multiple processes are clamped to one worker."""
    assert resolve_parallel_jobs(torch.device("cuda:3"), n_jobs) == 1


@pytest.mark.parametrize("n_jobs", [-2, -1, 1, 2, 8])
def test_resolve_parallel_jobs_preserves_cpu_parallelism(n_jobs: int) -> None:
    """CPU calculations retain the requested joblib parallelism."""
    assert resolve_parallel_jobs(torch.device("cpu"), n_jobs) == n_jobs


def test_resolve_parallel_jobs_preserves_single_cuda_worker() -> None:
    """An explicitly serial CUDA request does not need adjustment."""
    assert resolve_parallel_jobs("cuda", 1) == 1
