from unittest.mock import MagicMock

from qwen_think.session import ThinkingSession
from qwen_think.types import Backend, ThinkingMode


def _mock_client(content="response", thinking="reasoning"):
    client = MagicMock()
    client.base_url = "http://localhost:8000/v1"
    choice = MagicMock()
    choice.message.content = content
    choice.message.reasoning_content = thinking
    resp = MagicMock()
    resp.choices = [choice]
    client.chat.completions.create.return_value = resp
    return client


class TestThinkingSession:
    def test_init_explicit_backend(self):
        s = ThinkingSession(_mock_client(), backend="vllm")
        assert s.backend == Backend.VLLM

    def test_init_auto_detect(self):
        s = ThinkingSession(_mock_client())
        assert s.backend == Backend.VLLM

    def test_chat_stores_messages(self):
        s = ThinkingSession(_mock_client(), backend="vllm")
        s.chat("hello")
        assert len(s.messages) == 2  # user + assistant

    def test_chat_preserves_thinking(self):
        s = ThinkingSession(_mock_client(), backend="vllm")
        s.chat("hello")
        assert s.messages[1].thinking_content == "reasoning"

    def test_extra_body_has_nested_format(self):
        client = _mock_client()
        s = ThinkingSession(client, backend="vllm")
        s.chat(
            "Refactor this module to use dependency injection", mode=ThinkingMode.THINK
        )
        kwargs = client.chat.completions.create.call_args.kwargs
        eb = kwargs["extra_body"]
        assert "chat_template_kwargs" in eb
        assert eb["chat_template_kwargs"]["enable_thinking"] is True

    def test_non_standard_params_in_extra_body(self):
        """top_k, min_p, repetition_penalty go in extra_body, not top-level."""
        client = _mock_client()
        s = ThinkingSession(client, backend="vllm")
        s.chat("implement something")
        kwargs = client.chat.completions.create.call_args.kwargs
        assert "top_k" not in kwargs
        assert "min_p" not in kwargs
        assert "repetition_penalty" not in kwargs
        assert kwargs["extra_body"]["top_k"] == 20

    def test_standard_params_top_level(self):
        client = _mock_client()
        s = ThinkingSession(client, backend="vllm")
        s.chat("implement something")
        kwargs = client.chat.completions.create.call_args.kwargs
        assert "temperature" in kwargs
        assert "top_p" in kwargs

    def test_budget_refuse_raises(self):
        client = _mock_client()
        s = ThinkingSession(client, backend="vllm", budget=1000, min_context=900)
        # Fill up context
        for _ in range(50):
            s.add_message("user", "x" * 500)
        try:
            s.chat("one more")
            assert False, "Should have raised RuntimeError"
        except RuntimeError as e:
            assert "budget" in str(e).lower()

    def test_clear_history(self):
        s = ThinkingSession(_mock_client(), backend="vllm")
        s.chat("hello")
        assert len(s.messages) > 0
        s.clear_history(keep_system=False)
        assert len(s.messages) == 0

    def test_set_thinking_mode(self):
        s = ThinkingSession(_mock_client(), backend="vllm")
        s.set_thinking_mode(ThinkingMode.NO_THINK)
        assert s.thinking_mode == ThinkingMode.NO_THINK

    def test_repr(self):
        s = ThinkingSession(_mock_client(), backend="vllm")
        r = repr(s)
        assert "vllm" in r
        assert "think" in r
