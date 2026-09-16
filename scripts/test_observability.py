"""
Task 2 Smoke Test — LLM Observability & Cost Tracking
=======================================================
Verifies:
    TEST 1: CostCalculator — token → USD pricing
    TEST 2: PIIMasker      — regex masking of names/phones/emails
    TEST 3: MockTracer     — graceful fallback (zero-dependency no-op)
    TEST 4: LangfuseTracer — real Cloud connection (skipped if keys missing)

Usage:
    .venv\\Scripts\\python.exe scripts/test_observability.py
    .venv\\Scripts\\python.exe scripts/test_observability.py --force-mock
"""

from __future__ import annotations

import io
import os
import sys
import argparse
import logging
from pathlib import Path

# Ensure project root is on sys.path so `src.*` imports work
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Force UTF-8 for Windows console
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

logging.basicConfig(level=logging.WARNING)

# ── Helpers ───────────────────────────────────────────────────────────────────
PASS = "[PASS]"
FAIL = "[FAIL]"
SKIP = "[SKIP]"


def header(title: str) -> None:
    print("\n" + "-" * 60)
    print(f"  {title}")
    print("-" * 60)


def ok(msg: str) -> None:
    print(f"  {PASS} {msg}")


def fail(msg: str) -> None:
    print(f"  {FAIL} {msg}")


# ── TEST 1: CostCalculator ────────────────────────────────────────────────────
def test_cost_calculator() -> bool:
    header("TEST 1: CostCalculator — Token -> USD Pricing")
    try:
        # pyrefly: ignore [missing-import]
        from src.observability.cost_calculator import CostCalculator, TokenUsage

        calc = CostCalculator()

        # gpt-4o-mini: $0.15/1M input + $0.60/1M output
        # 1000 input + 300 output = $0.000150 + $0.000180 = $0.000330
        usage = TokenUsage(prompt_tokens=1000, completion_tokens=300)
        cost = calc.calculate(usage, model="gpt-4o-mini")
        expected = round((1000 / 1_000_000 * 0.15) + (300 / 1_000_000 * 0.60), 8)

        assert abs(cost - expected) < 1e-9, f"Cost mismatch: {cost} != {expected}"
        ok(f"gpt-4o-mini: 1000+300 tokens = ${cost:.8f} (expected ${expected:.8f})")

        # Zero tokens = zero cost
        zero_usage = TokenUsage(prompt_tokens=0, completion_tokens=0)
        assert calc.calculate(zero_usage) == 0.0
        ok("Zero tokens = $0.00")

        # Unknown model falls back gracefully
        unknown_cost = calc.calculate(usage, model="gpt-99-unknown")
        assert unknown_cost == 0.0
        ok("Unknown model = $0.00 (graceful fallback)")

        # CostRecord accumulation
        rec1 = calc.accumulate(TokenUsage(500, 100), model="gpt-4o-mini")
        rec2 = calc.accumulate(TokenUsage(500, 200), model="gpt-4o-mini")
        combined = rec1 + rec2
        assert combined.prompt_tokens == 1000
        assert combined.completion_tokens == 300
        assert combined.num_calls == 2
        ok(f"Accumulation: 2 calls combined = ${combined.cost_usd:.8f}")

        # Format helper
        fmt = calc.format_cost(0.000033)
        ok(f"format_cost(0.000033) = '{fmt}'")

        return True

    except Exception as exc:
        fail(str(exc))
        import traceback; traceback.print_exc()
        return False


# ── TEST 2: PIIMasker ─────────────────────────────────────────────────────────
def test_pii_masker() -> bool:
    header("TEST 2: PIIMasker — Regex PII Scrubbing")
    try:
        # pyrefly: ignore [missing-import]
        from src.observability.pii_masker import PIIMasker

        m = PIIMasker()

        # Phone number (Vietnamese)
        result = m.mask("Please call me at 0912-345-678 for appointment.")
        assert "[PHONE]" in result, f"Phone not masked: {result}"
        ok(f"Vietnamese phone: '{result}'")

        # Email
        result = m.mask("Contact john.doe@hospital.com for records.")
        assert "[EMAIL]" in result, f"Email not masked: {result}"
        ok(f"Email: '{result}'")

        # Name with honorific
        result = m.mask("Dr. John Smith reviewed the file.")
        assert "[NAME]" in result, f"Name not masked: {result}"
        ok(f"Name with honorific: '{result}'")

        # Empty string safety
        assert m.mask("") == ""
        assert m.mask("No PII here — just medical terms.") == "No PII here — just medical terms."
        ok("Empty string and PII-free string pass through unchanged")

        # Dictionary masking
        d = {
            "query": "Patient 0912-345-678 needs help",
            "session": "abc123",
            "num_results": 5,
            "nested": {"email": "x@y.com"},
        }
        masked = m.mask_dict(d)
        assert "[PHONE]" in masked["query"]
        assert masked["session"] == "abc123"           # unchanged
        assert masked["num_results"] == 5              # non-string unchanged
        assert "[EMAIL]" in masked["nested"]["email"]
        ok("mask_dict: strings masked, non-strings preserved, nested dicts handled")

        return True

    except Exception as exc:
        fail(str(exc))
        import traceback; traceback.print_exc()
        return False


# ── TEST 3: MockTracer (Graceful Fallback) ────────────────────────────────────
def test_mock_tracer() -> bool:
    header("TEST 3: MockTracer — Graceful No-op Fallback")
    try:
        # pyrefly: ignore [missing-import]
        from src.observability.langfuse_tracker import MockTracer, reset_tracer

        reset_tracer()
        tracer = MockTracer()

        # Full trace + span lifecycle — must not raise
        with tracer.trace("test-trace", user_id="test-user", input={"q": "headache"}) as root:
            ok(f"trace() opened: is_mock={root.is_mock}, trace_id={root.trace_id}")

            with root.span("HybridSearch", input={"query": "headache"}) as s:
                s.set_output({"num_results": 5})
                s.set_metadata({"latency_ms": 42.0})
                s.set_level("DEFAULT")
                ok(f"span 'HybridSearch' completed in {s.elapsed_ms:.1f}ms")

            with root.span("LLMGeneration", input={"model": "gpt-4o-mini"}) as s:
                s.set_output({"answer_length": 300})
                s.set_metadata({"cost_usd": 0.0003})
                ok(f"span 'LLMGeneration' completed in {s.elapsed_ms:.1f}ms")

            root.update(output={"answer": "test answer"}, metadata={"cached": False})

        tracer.flush()
        ok("MockTracer: all lifecycle methods completed without errors")

        # Test singleton factory fallback (force_mock=True)
        reset_tracer()
        # pyrefly: ignore [missing-import]
        from src.observability.langfuse_tracker import get_tracer
        t = get_tracer(force_mock=True)
        assert isinstance(t, MockTracer)
        ok("get_tracer(force_mock=True) -> MockTracer singleton")

        return True

    except Exception as exc:
        fail(str(exc))
        import traceback; traceback.print_exc()
        return False


# ── TEST 4: LangfuseTracer (Real Cloud) ───────────────────────────────────────
def test_langfuse_cloud() -> bool:
    header("TEST 4: LangfuseTracer — Langfuse Cloud Connection")

    public_key = os.getenv("LANGFUSE_PUBLIC_KEY", "")
    secret_key  = os.getenv("LANGFUSE_SECRET_KEY", "")
    host        = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")

    if not public_key or not secret_key:
        print(f"  {SKIP} LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY not set in .env")
        print("        Set these keys to enable full Langfuse Cloud observability.")
        return True  # Skip is not a failure

    try:
        # pyrefly: ignore [missing-import]
        from src.observability.langfuse_tracker import LangfuseTracer, reset_tracer
        reset_tracer()

        tracer = LangfuseTracer(
            public_key=public_key,
            secret_key=secret_key,
            host=host,
        )
        ok(f"LangfuseTracer connected to {host}")

        with tracer.trace(
            "smoke-test",
            user_id="test-anon",
            input={"query": "What is asthma?"},
            metadata={"test": True},
        ) as root:
            ok(f"Root trace opened: trace_id={root.trace_id}")

            with root.span("LLMGeneration", input={"model": "gpt-4o-mini"}) as s:
                s.set_output({"answer_length": 150})
                s.set_metadata({"cost_usd": 0.000033, "prompt_tokens": 200, "completion_tokens": 100})
                ok(f"Span 'LLMGeneration' completed in {s.elapsed_ms:.1f}ms")

            root.update(
                output={"answer": "Asthma is a chronic respiratory condition..."},
                metadata={"total_cost_usd": 0.000033},
            )

        tracer.flush()
        ok("Trace flushed to Langfuse Cloud — check your dashboard!")
        print(f"\n  -> Dashboard: {host}/traces")

        return True

    except Exception as exc:
        fail(f"Langfuse Cloud error: {exc}")
        import traceback; traceback.print_exc()
        return False


# ── Main ───────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="Task 2 Observability smoke tests")
    parser.add_argument(
        "--force-mock", action="store_true",
        help="Force MockTracer even if Langfuse keys are present"
    )
    args = parser.parse_args()

    if args.force_mock:
        os.environ.pop("LANGFUSE_PUBLIC_KEY", None)
        os.environ.pop("LANGFUSE_SECRET_KEY", None)

    # Load .env
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    print("\n" + "=" * 60)
    print("  TASK 2: LLM Observability & Cost Tracking — Smoke Tests")
    print("=" * 60)

    results = {
        "CostCalculator":  test_cost_calculator(),
        "PIIMasker":       test_pii_masker(),
        "MockTracer":      test_mock_tracer(),
        "LangfuseCloud":   test_langfuse_cloud(),
    }

    print("\n" + "=" * 60)
    print("  RESULTS SUMMARY")
    print("=" * 60)
    passed = 0
    for name, ok_flag in results.items():
        status = PASS if ok_flag else FAIL
        print(f"  {status}  {name}")
        if ok_flag:
            passed += 1

    print(f"\n  {passed}/{len(results)} tests passed")
    print("=" * 60 + "\n")
    sys.exit(0 if passed == len(results) else 1)


if __name__ == "__main__":
    main()
