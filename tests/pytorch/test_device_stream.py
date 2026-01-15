# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch

from lmdeploy.pytorch.devices import current_stream, device_stream_context


class TestDeviceStream:
    """Test device stream utilities."""

    def test_current_stream_cuda(self):
        """Test current_stream function for CUDA."""
        if not torch.cuda.is_available():
            pytest.skip('CUDA not available')
        
        stream = current_stream('cuda')
        assert stream is not None
        assert isinstance(stream, torch.cuda.Stream)

    def test_current_stream_ascend(self):
        """Test current_stream function for Ascend/NPU."""
        try:
            import torch_npu
            stream = current_stream('ascend')
            # If torch_npu is available, stream should not be None
            assert stream is not None
        except ImportError:
            # If torch_npu is not available, stream will be None
            stream = current_stream('ascend')
            assert stream is None

    def test_current_stream_unknown(self):
        """Test current_stream function for unknown device."""
        stream = current_stream('unknown_device')
        assert stream is None

    def test_device_stream_context_cuda(self):
        """Test device_stream_context for CUDA."""
        if not torch.cuda.is_available():
            pytest.skip('CUDA not available')
        
        stream = torch.cuda.Stream()
        with device_stream_context(stream, 'cuda'):
            # Context manager should work without errors
            pass

    def test_device_stream_context_none_stream(self):
        """Test device_stream_context with None stream."""
        # Should work without errors even with None stream
        with device_stream_context(None, 'cuda'):
            pass
        
        with device_stream_context(None, 'ascend'):
            pass

    def test_device_stream_context_unknown_device(self):
        """Test device_stream_context for unknown device."""
        # Should work without errors for unknown device
        with device_stream_context(None, 'unknown_device'):
            pass
