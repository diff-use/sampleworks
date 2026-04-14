"""Tests for RF3Wrapper device selection (issue #37).

Fabric's ``devices`` argument treats an ``int`` as "take this many GPUs, starting
from index 0", which pins every worker to ``cuda:0`` and serialises nominally
parallel jobs. The wrapper must accept an explicit device and bind Fabric to
that specific GPU index via ``devices_per_node=[idx]``.
"""

import pytest
import torch
from sampleworks.utils.imports import require_rf3, RF3_AVAILABLE


pytestmark = pytest.mark.skipif(not RF3_AVAILABLE, reason="RF3 dependencies not installed")

if RF3_AVAILABLE:
    from sampleworks.models.rf3.wrapper import _cuda_index, RF3Wrapper


class TestCudaIndex:
    """Validate CUDA index extraction used to drive Fabric device selection."""

    @pytest.mark.parametrize(
        ("device", "expected"),
        [
            ("cuda:0", 0),
            ("cuda:3", 3),
            (torch.device("cuda", 5), 5),
            ("cuda", 0),
        ],
    )
    def test_returns_index(self, device, expected):
        assert _cuda_index(device) == expected

    @pytest.mark.parametrize("device", ["cpu", torch.device("cpu")])
    def test_rejects_non_cuda(self, device):
        with pytest.raises(ValueError, match="CUDA device"):
            _cuda_index(device)


@pytest.mark.gpu
@pytest.mark.slow
class TestRF3WrapperDeviceBinding:
    """Regression for issue #37: device must propagate to Fabric.

    Prior behaviour: RF3Wrapper ignored caller-specified device and always
    landed on ``cuda:0`` because Fabric defaults to ``devices=1``.
    """

    @require_rf3()
    def test_wrapper_honors_requested_cuda_index(self, rf3_checkpoint_path):
        if torch.cuda.device_count() < 2:
            pytest.skip("multi-GPU regression test needs >= 2 CUDA devices")

        wrapper = RF3Wrapper(checkpoint_path=rf3_checkpoint_path, device="cuda:1")
        assert wrapper.device == torch.device("cuda:1")
