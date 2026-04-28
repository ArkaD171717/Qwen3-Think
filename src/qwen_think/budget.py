"""Context token budget management with 128K minimum guard for Qwen3.6."""

from __future__ import annotations

import logging
from typing import Callable, List, Optional

from .types import BudgetAction, BudgetStatus, Message

logger = logging.getLogger("qwen-think.budget")

DEFAULT_MIN_CONTEXT = 128_000
WARN_RATIO = 0.3
COMPRESS_RATIO = 0.15
AVG_TOKENS_PER_CHAR = 0.5
COMPRESSED_MESSAGE_MAX_TOKENS = 200


def estimate_tokens(text: str) -> int:
    """Heuristic: ~0.5 tokens/char. Use a real tokenizer for exact counts."""
    if not text:
        return 0
    return max(1, int(len(text) * AVG_TOKENS_PER_CHAR))


def summarize_text(text: str, max_tokens: int = COMPRESSED_MESSAGE_MAX_TOKENS) -> str:
    if not text:
        return ""
    max_chars = int(max_tokens / AVG_TOKENS_PER_CHAR)
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


def compress_messages(
    messages: List[Message],
    keep_recent: int = 4,
    max_tokens_per_message: int = COMPRESSED_MESSAGE_MAX_TOKENS,
) -> List[Message]:
    if len(messages) <= keep_recent:
        return messages

    compressed: List[Message] = []

    for i, msg in enumerate(messages):
        if i >= len(messages) - keep_recent or msg.role == "system":
            compressed.append(msg)
        else:
            new_content = summarize_text(msg.content, max_tokens_per_message)
            new_thinking = (
                summarize_text(msg.thinking_content, max_tokens_per_message)
                if msg.thinking_content
                else None
            )
            compressed_msg = Message(
                role=msg.role,
                content=new_content,
                thinking_content=new_thinking,
                token_count=estimate_tokens(new_content),
                preserved=msg.preserved,
            )
            compressed.append(compressed_msg)

    return compressed


class BudgetManager:
    """Tracks token usage and guards the 128K minimum context threshold."""

    def __init__(
        self,
        total_budget: int = 200_000,
        min_context: int = DEFAULT_MIN_CONTEXT,
        token_counter: Optional[Callable[[str], int]] = None,
        warn_ratio: float = WARN_RATIO,
        compress_ratio: float = COMPRESS_RATIO,
        auto_compress: bool = True,
    ) -> None:
        self.total_budget = total_budget
        self.min_context = min_context
        self.token_counter = token_counter or estimate_tokens
        self.warn_ratio = warn_ratio
        self.compress_ratio = compress_ratio
        self.auto_compress = auto_compress

    def count_tokens(self, text: str) -> int:
        return self.token_counter(text)

    def count_message_tokens(self, message: Message) -> int:
        tokens = self.count_tokens(message.content)
        if message.thinking_content:
            tokens += self.count_tokens(message.thinking_content)
        return tokens

    def count_messages_tokens(self, messages: List[Message]) -> int:
        return sum(self.count_message_tokens(msg) for msg in messages)

    def check_budget(self, messages: List[Message]) -> BudgetStatus:
        used = self.count_messages_tokens(messages)
        available = self.total_budget - used

        # Thresholds relative to min_context so the cascade
        # (REFUSE < COMPRESS < WARN < OK) holds for any budget size
        warn_threshold = self.min_context * (1.0 + self.warn_ratio)
        compress_threshold = self.min_context * (1.0 + self.compress_ratio)

        if available < self.min_context:
            action = BudgetAction.REFUSE
            message = (
                f"CRITICAL: Available context ({available:,} tokens) is below "
                f"the minimum ({self.min_context:,} tokens) required to "
                f"preserve Qwen3.6 thinking capabilities. Reasoning quality "
                f"will be silently degraded. Compress or reduce conversation "
                f"history immediately."
            )
        elif available < compress_threshold:
            action = BudgetAction.COMPRESS
            message = (
                f"WARNING: Available context ({available:,} tokens) is very "
                f"low (just above the {self.min_context:,} minimum). "
                f"Auto-compression recommended to preserve thinking quality."
            )
        elif available < warn_threshold:
            action = BudgetAction.WARN
            message = (
                f"Context usage at {used / self.total_budget:.1%}. "
                f"Available: {available:,} tokens. "
                f"Approaching the threshold for thinking quality degradation."
            )
        else:
            action = BudgetAction.OK
            message = (
                f"Context usage: {used / self.total_budget:.1%}. "
                f"Available: {available:,} tokens."
            )

        return BudgetStatus(
            total_tokens=self.total_budget,
            used_tokens=used,
            available_tokens=available,
            min_context=self.min_context,
            action=action,
            message=message,
        )

    def compress(
        self,
        messages: List[Message],
        keep_recent: int = 4,
    ) -> List[Message]:
        original_tokens = self.count_messages_tokens(messages)
        compressed = compress_messages(messages, keep_recent=keep_recent)
        new_tokens = self.count_messages_tokens(compressed)
        freed = original_tokens - new_tokens

        logger.info(
            "Compressed conversation: %d → %d tokens (freed %d)",
            original_tokens,
            new_tokens,
            freed,
        )

        return compressed

    def update_message_counts(self, messages: List[Message]) -> List[Message]:
        for msg in messages:
            msg.token_count = self.count_message_tokens(msg)
        return messages
