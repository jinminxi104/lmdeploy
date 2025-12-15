# Copyright (c) OpenMMLab. All rights reserved.
from .device_manager import (DefaultContext, DeviceContext, current_stream, device_stream_context, get_device_manager)

__all__ = ['DeviceContext', 'DefaultContext', 'get_device_manager', 'current_stream', 'device_stream_context']
