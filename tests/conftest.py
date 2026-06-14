from sampleworks.gpu_memory_optimizer import optimize_gpu_memory

def test_gpu_memory_optimization():
    """Test GPU memory optimization during CI validation."""
    result = optimize_gpu_memory(device_id=0)
    assert result["status"] == "optimized"
    assert result["memory_freed_mb"] >= 0.0
