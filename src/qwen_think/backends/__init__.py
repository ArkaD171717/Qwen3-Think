"""
qwen-think backends: Backend auto-detection and registry.
"""

from __future__ import annotations

from typing import Dict, List, Type

from ..types import Backend
from .base import BaseBackend
from .dashscope import DashScopeBackend
from .llamacpp import LlamaCppBackend
from .vllm import SGLangBackend, VLLMBackend

# Registry of all built-in backends
_BACKEND_REGISTRY: Dict[Backend, Type[BaseBackend]] = {
    Backend.VLLM: VLLMBackend,
    Backend.SGLANG: SGLangBackend,
    Backend.DASHSCOPE: DashScopeBackend,
    Backend.LLAMACPP: LlamaCppBackend,
    Backend.OPENAI: VLLMBackend,  # OpenAI-compatible treated like vLLM
}


def get_backend(backend: Backend, **kwargs) -> BaseBackend:
    """Instantiate a backend by enum value.

    Args:
        backend: The backend type to create.
        **kwargs: Additional arguments passed to the backend constructor.

    Returns:
        An instance of the requested backend.
    """
    cls = _BACKEND_REGISTRY.get(backend)
    if cls is None:
        raise ValueError(
            f"Unknown backend: {backend}. Supported: {list(_BACKEND_REGISTRY.keys())}"
        )
    return cls(**kwargs)


def detect_backend(base_url: str) -> BaseBackend:
    """Auto-detect the backend from a base URL.

    Runs all registered backends' detection heuristics and returns
    the one with the highest confidence score.

    Args:
        base_url: The API base URL to detect.

    Returns:
        The backend with the highest confidence score.

    Raises:
        ValueError: If no backend could be detected.
    """
    candidates: List[tuple[float, BaseBackend]] = []

    for backend_type, cls in _BACKEND_REGISTRY.items():
        instance = cls()
        score = instance.detect(base_url)
        if score > 0:
            candidates.append((score, instance))

    if not candidates:
        # Default to vLLM for unknown URLs with /v1 endpoint
        return VLLMBackend()

    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


__all__ = [
    "BaseBackend",
    "VLLMBackend",
    "SGLangBackend",
    "DashScopeBackend",
    "LlamaCppBackend",
    "get_backend",
    "detect_backend",
]
