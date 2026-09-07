# SPDX-License-Identifier: Apache-2.0
"""Keep Qwen3-VL vision-forward timing separate from MM input overhead."""

from functools import wraps
from typing import Any

from vllm.model_executor.models.qwen3_vl import Qwen3_VisionTransformer
from vllm.utils.timing_trace import timing_span


_original_forward = Qwen3_VisionTransformer.forward

if not getattr(_original_forward, "_vllm_ascend_vit_timing", False):

    @wraps(_original_forward)
    def _timed_vision_forward(self: Any, *args: Any, **kwargs: Any) -> Any:
        # Excludes MM batching/H2D and the caller-side pixel_values dtype cast.
        with timing_span(
            "worker.vit_forward",
            sync_device=True,
            phase="prefill",
        ):
            return _original_forward(self, *args, **kwargs)

    _timed_vision_forward._vllm_ascend_vit_timing = True  # type: ignore[attr-defined]
    Qwen3_VisionTransformer.forward = _timed_vision_forward
