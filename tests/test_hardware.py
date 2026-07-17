"""CPU-safe checks for hardware capability detection."""

import torch

from picotron.utils.hardware import (
    AttentionBackend,
    detect_attention_backend,
    get_gpu_compute_capability,
    select_training_dtype,
)


def test_hardware_detection_is_safe_without_cuda() -> None:
    capability = get_gpu_compute_capability()
    dtype = select_training_dtype()
    backend_report = detect_attention_backend()

    if not torch.cuda.is_available():
        assert capability is None
        assert dtype is torch.float32
        assert not backend_report.flash_attention_available
        assert not backend_report.xformers_available

    assert backend_report.selected in set(AttentionBackend)

