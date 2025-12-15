# Copyright (c) OpenMMLab. All rights reserved.
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable, Optional

import torch

from lmdeploy.pytorch.utils import CtxMgrBase, singleton


@dataclass
class DeviceContext:
    device_type: str = 'cuda'


DefaultContext = DeviceContext()


@singleton
class DeviceManager(CtxMgrBase[DeviceContext]):

    def __init__(self):
        super().__init__(DefaultContext)
        self._context_callback: dict[int, Callable] = dict()
        self._next_cb_handle = 0

    def register_context_callback(self, callback: Callable):
        """Register callback."""
        handle = self._next_cb_handle
        self._context_callback[handle] = callback
        self._next_cb_handle += 1
        return handle

    def unregister_context_callback(self, handle: int):
        """Unregister callback."""
        self._context_callback.pop(handle, None)


def get_device_manager():
    """Get device manager."""
    return DeviceManager()


def current_stream(device_type: str = 'cuda'):
    """Get current stream for the specified device type.
    
    Args:
        device_type: Device type ('cuda', 'ascend', 'npu', etc.)
        
    Returns:
        Current stream object for the device type
    """
    if device_type == 'cuda':
        return torch.cuda.current_stream()
    elif device_type in ['ascend', 'npu']:
        try:
            import torch_npu
            return torch_npu.npu.current_stream()
        except ImportError:
            # Fallback to None if torch_npu is not available
            return None
    else:
        # For other device types, return None
        return None


@contextmanager
def device_stream_context(stream, device_type: str = 'cuda'):
    """Context manager for device streams that works across different device types.
    
    Args:
        stream: Stream object (can be None for non-streaming devices)
        device_type: Device type ('cuda', 'ascend', 'npu', etc.)
        
    Yields:
        None
    """
    if device_type == 'cuda' and stream is not None:
        with torch.cuda.stream(stream):
            yield
    elif device_type in ['ascend', 'npu'] and stream is not None:
        try:
            import torch_npu
            with torch_npu.npu.stream(stream):
                yield
        except ImportError:
            # If torch_npu is not available, just yield without stream context
            yield
    else:
        # For other device types or None stream, just yield without stream context
        yield
