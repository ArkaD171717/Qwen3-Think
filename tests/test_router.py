from qwen_think.router import ComplexityRouter, RuleBasedClassifier
from qwen_think.types import Complexity, ThinkingMode


class TestRuleBasedClassifier:
    def setup_method(self):
        self.clf = RuleBasedClassifier()

    def test_simple_queries(self):
        for q in ["yes", "What is Python?", "translate hello"]:
            c = self.clf.classify(q)
            assert c in (Complexity.SIMPLE, Complexity.MODERATE), f"{q} → {c}"

    def test_complex_queries(self):
        for q in [
            "Refactor this module to use dependency injection",
            "Debug and rewrite the authentication middleware that is leaking sessions step by step",
        ]:
            c = self.clf.classify(q)
            assert c in (Complexity.COMPLEX, Complexity.AGENTIC), f"{q} → {c}"

    def test_code_boosts_score(self):
        q = "```python\ndef foo():\n    return bar\n```\nfix this function"
        c = self.clf.classify(q)
        assert c in (Complexity.COMPLEX, Complexity.AGENTIC)


class TestComplexityRouter:
    def setup_method(self):
        self.router = ComplexityRouter()

    def test_simple_routes_to_no_think(self):
        d = self.router.route("What is 2+2?")
        if d.complexity == Complexity.SIMPLE:
            assert d.mode == ThinkingMode.NO_THINK
            assert d.preserve_thinking is False

    def test_complex_routes_to_think(self):
        d = self.router.route("Refactor this entire module for async")
        assert d.mode == ThinkingMode.THINK
        assert d.sampling.temperature == 1.0

    def test_force_thinking_overrides(self):
        router = ComplexityRouter(force_thinking=True)
        d = router.route("yes")
        assert d.mode == ThinkingMode.THINK
        assert d.preserve_thinking is True

    def test_override_mode(self):
        d = self.router.route("Refactor this", override_mode=ThinkingMode.NO_THINK)
        assert d.mode == ThinkingMode.NO_THINK
        assert d.preserve_thinking is False

    def test_sampling_matches_mode(self):
        think = self.router.route("Implement a REST API with auth")
        assert think.sampling.temperature == 1.0

    def test_confidence_is_bounded(self):
        d = self.router.route("Implement something complex step by step")
        assert 0.0 <= d.confidence <= 1.0
