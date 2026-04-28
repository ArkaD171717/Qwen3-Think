"""Thinking session manager for Qwen3.6: backend normalization,
atomic sampling swap, and 128K context budget guard."""

from .backends import (
    BaseBackend,
    DashScopeBackend,
    LlamaCppBackend,
    SGLangBackend,
    VLLMBackend,
    detect_backend,
    get_backend,
)
from .budget import BudgetManager, BudgetStatus, estimate_tokens
from .router import ComplexityRouter, LLMClassifier, RuleBasedClassifier
from .sampling import SamplingManager
from .session import ThinkingSession
from .types import (
    NON_THINKING_SAMPLING,
    THINKING_SAMPLING,
    Backend,
    BudgetAction,
    Complexity,
    Message,
    RouterDecision,
    SamplingConfig,
    ThinkingMode,
    ThinkingState,
)

__version__ = "0.1.0"

__all__ = [
    # Session (main entry point)
    "ThinkingSession",
    # Backends
    "BaseBackend",
    "VLLMBackend",
    "SGLangBackend",
    "DashScopeBackend",
    "LlamaCppBackend",
    "get_backend",
    "detect_backend",
    # Budget
    "BudgetManager",
    "BudgetStatus",
    "estimate_tokens",
    # Router
    "ComplexityRouter",
    "RuleBasedClassifier",
    "LLMClassifier",
    # Sampling
    "SamplingManager",
    "THINKING_SAMPLING",
    "NON_THINKING_SAMPLING",
    # Types
    "Backend",
    "BudgetAction",
    "Complexity",
    "Message",
    "RouterDecision",
    "SamplingConfig",
    "ThinkingMode",
    "ThinkingState",
]
