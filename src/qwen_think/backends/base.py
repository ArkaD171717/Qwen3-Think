"""Abstract base class for backend normalization."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from ..types import Backend, BackendPayload, ThinkingMode


class BaseBackend(ABC):
    """Abstract base class for backend-specific thinking flag normalization."""

    backend: Backend

    @abstractmethod
    def build_payload(
        self,
        mode: ThinkingMode,
        preserve_thinking: bool = True,
        sampling: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> BackendPayload: ...

    @abstractmethod
    def detect(self, base_url: Optional[str] = None) -> float: ...

    def _common_sampling(
        self,
        mode: ThinkingMode,
        sampling: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if sampling is None:
            from ..sampling import SamplingManager

            sm = SamplingManager()
            return sm.get_params(mode)
        return sampling
