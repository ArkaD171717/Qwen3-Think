from qwen_think.budget import BudgetManager, estimate_tokens
from qwen_think.types import BudgetAction, Message


def test_estimate_tokens_empty():
    assert estimate_tokens("") == 0


def test_estimate_tokens_positive():
    assert estimate_tokens("hello world") > 0


def _make_msg(chars: int) -> Message:
    return Message(role="user", content="x" * chars)


class TestBudgetManager:
    def setup_method(self):
        self.bm = BudgetManager(total_budget=200_000, min_context=128_000)

    def test_empty_is_ok(self):
        status = self.bm.check_budget([])
        assert status.action == BudgetAction.OK
        assert status.available_tokens == 200_000

    def test_warn_triggers(self):
        # With min_context=128K: warn_threshold = 128K * 1.30 = 166,400
        # Need available < 166,400 but >= 128K * 1.15 = 147,200
        # Use ~31K tokens → available = 169K... need more
        # available = 155K → used = 45K → 90,000 chars
        msg = _make_msg(100_000)  # ~50K tokens → 150K available
        status = self.bm.check_budget([msg])
        assert status.action == BudgetAction.WARN

    def test_compress_triggers(self):
        # compress_threshold = 128K * 1.15 = 147,200
        # Need available < 147,200 but >= 128K
        # available = 135K → used = 65K → 130,000 chars
        msg = _make_msg(130_000)
        status = self.bm.check_budget([msg])
        assert status.action == BudgetAction.COMPRESS

    def test_refuse_triggers(self):
        # Need available < 128K → used > 72K → 144,001+ chars
        msg = _make_msg(160_000)
        status = self.bm.check_budget([msg])
        assert status.action == BudgetAction.REFUSE

    def test_is_below_minimum(self):
        msg = _make_msg(160_000)
        status = self.bm.check_budget([msg])
        assert status.is_below_minimum is True

        ok = self.bm.check_budget([])
        assert ok.is_below_minimum is False

    def test_compression_reduces_tokens(self):
        msgs = [Message(role="user", content="x" * 10_000) for _ in range(20)]
        original = self.bm.count_messages_tokens(msgs)
        compressed = self.bm.compress(msgs, keep_recent=4)
        assert self.bm.count_messages_tokens(compressed) < original

    def test_compression_keeps_recent(self):
        msgs = [Message(role="user", content=f"msg{i}" * 500) for i in range(10)]
        compressed = self.bm.compress(msgs, keep_recent=3)
        assert len(compressed) == 10
        # Last 3 should be unchanged
        for i in range(7, 10):
            assert compressed[i].content == msgs[i].content

    def test_usage_ratio(self):
        status = self.bm.check_budget([])
        assert status.usage_ratio == 0.0

        msg = _make_msg(200_000)  # 100K tokens
        status = self.bm.check_budget([msg])
        assert status.usage_ratio == 0.5
