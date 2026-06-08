"""Token-bucket rate limiter for GitHub API.

GitHub API limits: 5000 req/h authenticated, 60 req/h unauthenticated.
We default to 4500 for authenticated (buffer) and 55 for unauthenticated.
"""

import asyncio
import os
import time

# ---------------------------------------------------------------------------
# RateLimiter
# ---------------------------------------------------------------------------


class RateLimiter:
    """Token-bucket rate limiter."""

    def __init__(self, rate: int = 4500, per: int = 3600):
        self.rate = rate          # tokens per `per` seconds
        self.per = per            # window size in seconds
        self.tokens = float(rate)
        self.last_refill = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        """Acquire one token, sleeping if none are available."""
        async with self._lock:
            now = time.monotonic()
            elapsed = now - self.last_refill
            new_tokens = elapsed * (self.rate / self.per)
            self.tokens = min(self.tokens + new_tokens, float(self.rate))
            self.last_refill = now

            if self.tokens >= 1:
                self.tokens -= 1
                return

            # Calculate how long to wait for 1 token
            deficit = 1 - self.tokens
            wait = deficit * (self.per / self.rate)
            self.tokens = 0.0  # consume what little we have

        # Wait outside the lock so other coroutines can proceed
        await asyncio.sleep(wait)

        # Deduct the token we just waited for
        async with self._lock:
            self.tokens = max(self.tokens - 1, -1.0)
            self.last_refill = time.monotonic()

    async def update_from_headers(self, headers: dict) -> None:
        """Sync internal token count from *X-RateLimit-Remaining* response header.

        We only ever *lower* our token count to match reality, never raise it
        (GitHub's view is authoritative; our bucket may be more optimistic).
        """
        remaining = headers.get("X-RateLimit-Remaining")
        if remaining is None:
            return
        try:
            remaining = int(remaining)
        except (TypeError, ValueError):
            return
        async with self._lock:
            self.tokens = min(self.tokens, float(remaining))


# ---------------------------------------------------------------------------
# Global singleton
# ---------------------------------------------------------------------------

_github_limiter: RateLimiter | None = None


def get_github_limiter() -> RateLimiter:
    """Return the global :class:`RateLimiter` singleton, creating it on first call."""
    global _github_limiter
    if _github_limiter is None:
        rate = 4500 if os.getenv("GITHUB_TOKEN") else 55
        _github_limiter = RateLimiter(rate=rate)
    return _github_limiter
