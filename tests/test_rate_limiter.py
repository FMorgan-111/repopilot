"""Tests for the token-bucket rate limiter."""

import asyncio

import pytest

from src.rate_limiter import RateLimiter, get_github_limiter


# ---------------------------------------------------------------------------
# Unit tests for RateLimiter
# ---------------------------------------------------------------------------


def test_initial_tokens_full():
    """New RateLimiter should start with `rate` tokens."""
    rl = RateLimiter(rate=100, per=3600)
    assert rl.tokens == 100.0
    assert rl.rate == 100
    assert rl.per == 3600


async def test_acquire_consumes_token():
    """acquire() should decrement tokens by 1."""
    rl = RateLimiter(rate=10, per=3600)
    assert rl.tokens == 10.0

    await rl.acquire()

    # After refill (immediate, no time elapsed), tokens = min(10, 10) - 1 = 9
    assert 8.9 <= rl.tokens <= 9.0


async def test_acquire_multiple_consumes_proportionally():
    """Multiple acquires consume multiple tokens."""
    rl = RateLimiter(rate=10, per=3600)

    for _ in range(5):
        await rl.acquire()

    # ~5 tokens consumed, should have ~5 left
    assert 4.9 <= rl.tokens <= 5.1


async def test_acquire_blocks_when_empty(monkeypatch):
    """When tokens are exhausted, acquire should call asyncio.sleep."""
    rl = RateLimiter(rate=1, per=3600)

    # Consume the only token
    await rl.acquire()
    assert rl.tokens < 0.1  # near zero/negative

    # Now acquire again — should block (call asyncio.sleep)
    sleep_calls = []

    async def fake_sleep(duration):
        sleep_calls.append(duration)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)

    await rl.acquire()

    assert len(sleep_calls) == 1
    assert sleep_calls[0] > 0  # should wait for at least some time


async def test_acquire_does_not_block_when_tokens_available(monkeypatch):
    """When tokens are available, acquire should NOT call asyncio.sleep."""
    rl = RateLimiter(rate=10, per=3600)

    sleep_calls = []

    async def fake_sleep(duration):
        sleep_calls.append(duration)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)

    await rl.acquire()

    assert len(sleep_calls) == 0


async def test_refill_over_time(monkeypatch):
    """Tokens refill proportionally to elapsed time."""
    rl = RateLimiter(rate=3600, per=3600)  # 1 token per second

    # Consume all tokens
    for _ in range(3600):
        await rl.acquire()

    assert rl.tokens < 0.1

    # Simulate 10 seconds passing by monkeypatching time.monotonic
    import time
    original_monotonic = time.monotonic
    base = original_monotonic()

    call_count = 0

    def fake_monotonic():
        nonlocal call_count
        call_count += 1
        # First call: original time (for refill calc), subsequent: +10s
        if call_count == 1:
            return base + 10
        return base + 10

    monkeypatch.setattr(time, "monotonic", fake_monotonic)

    # Acquire — refill should add ~10 tokens
    await rl.acquire()

    # After refill + consume, should have ~9 tokens
    assert 8.5 <= rl.tokens <= 10.0


# ---------------------------------------------------------------------------
# update_from_headers tests
# ---------------------------------------------------------------------------


async def test_update_from_headers_with_remaining():
    """X-RateLimit-Remaining header should update tokens downward."""
    rl = RateLimiter(rate=4500, per=3600)
    original = rl.tokens

    await rl.update_from_headers({"X-RateLimit-Remaining": "100"})

    # Tokens should be capped at 100 (lower than original)
    assert rl.tokens == 100.0
    assert rl.tokens < original


async def test_update_from_headers_never_lowers_below_current():
    """If remaining > current tokens, we don't raise (only cap downward)."""
    rl = RateLimiter(rate=100, per=3600)

    # Consume some tokens
    for _ in range(50):
        await rl.acquire()

    current = rl.tokens  # ~50

    # Headers say 4000 remaining — we don't raise above current
    await rl.update_from_headers({"X-RateLimit-Remaining": "4000"})

    # Should NOT increase beyond what we already have
    assert rl.tokens <= current + 1  # allow tiny refill drift


async def test_update_from_headers_no_header_is_noop():
    """Missing X-RateLimit-Remaining should not crash or change tokens."""
    rl = RateLimiter(rate=100, per=3600)
    original = rl.tokens

    # Empty headers
    await rl.update_from_headers({})
    assert rl.tokens == original

    # Other headers only
    await rl.update_from_headers({"Content-Type": "application/json"})
    assert rl.tokens == original


async def test_update_from_headers_non_integer_value():
    """Non-integer X-RateLimit-Remaining should be ignored gracefully."""
    rl = RateLimiter(rate=100, per=3600)
    original = rl.tokens

    await rl.update_from_headers({"X-RateLimit-Remaining": "not-a-number"})

    assert rl.tokens == original


async def test_update_from_headers_zero_remaining():
    """X-RateLimit-Remaining: 0 should set tokens to 0."""
    rl = RateLimiter(rate=100, per=3600)

    await rl.update_from_headers({"X-RateLimit-Remaining": "0"})

    assert rl.tokens == 0.0


# ---------------------------------------------------------------------------
# Global singleton tests
# ---------------------------------------------------------------------------


def test_get_github_limiter_returns_singleton(monkeypatch):
    """Multiple calls to get_github_limiter return the same instance."""
    # Reset the singleton for test isolation
    import src.rate_limiter as rl_mod
    monkeypatch.setattr(rl_mod, "_github_limiter", None)

    limiter1 = get_github_limiter()
    limiter2 = get_github_limiter()

    assert limiter1 is limiter2


def test_get_github_limiter_authenticated_rate(monkeypatch):
    """With GITHUB_TOKEN, rate should be 4500."""
    import src.rate_limiter as rl_mod
    monkeypatch.setattr(rl_mod, "_github_limiter", None)
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_test123")

    limiter = get_github_limiter()

    assert limiter.rate == 4500


def test_get_github_limiter_unauthenticated_rate(monkeypatch):
    """Without GITHUB_TOKEN, rate should be 55."""
    import src.rate_limiter as rl_mod
    monkeypatch.setattr(rl_mod, "_github_limiter", None)
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)

    limiter = get_github_limiter()

    assert limiter.rate == 55


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


async def test_acquire_with_rate_one(monkeypatch):
    """RateLimiter with rate=1 should handle acquire gracefully."""
    rl = RateLimiter(rate=1, per=3600)

    # Consume the only token
    await rl.acquire()
    assert rl.tokens < 1.0

    sleep_calls = []

    async def fake_sleep(duration):
        sleep_calls.append(duration)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)

    # Next acquire should sleep
    await rl.acquire()
    assert len(sleep_calls) == 1


async def test_update_from_headers_concurrent_safety():
    """Multiple concurrent updates should not corrupt tokens."""
    rl = RateLimiter(rate=100, per=3600)

    async def update(val):
        await rl.update_from_headers({"X-RateLimit-Remaining": str(val)})

    # Run concurrent updates
    await asyncio.gather(
        update(10),
        update(20),
        update(5),
    )

    # After concurrent updates, tokens should be min(original, min of headers)
    # = min(100, 5) = 5 (or close to it; exact depends on ordering)
    assert rl.tokens <= 20.0  # at most 20, given concurrent writes
