"""Unit tests for `assistant.rate_limiter` -- real threads, real timing (small windows).

No mocking of the clock or the thread -- `ExternalCallQueue` runs a real
background thread against a real `TokenBucketRateLimiter`, same as it
would in production, just with millisecond-scale windows so the tests
stay fast.
"""

from __future__ import annotations

import threading
import time

import pytest

from assistant.rate_limiter import ExternalCallQueue, QueueFullError, TokenBucketRateLimiter

# -- TokenBucketRateLimiter -----------------------------------------------------


def test_rejects_non_positive_max_calls() -> None:
    with pytest.raises(ValueError, match="max_calls"):
        TokenBucketRateLimiter(max_calls=0, window_seconds=1.0)


def test_rejects_non_positive_window() -> None:
    with pytest.raises(ValueError, match="window_seconds"):
        TokenBucketRateLimiter(max_calls=1, window_seconds=0.0)


def test_try_acquire_allows_up_to_max_calls_then_denies() -> None:
    limiter = TokenBucketRateLimiter(max_calls=2, window_seconds=10.0)
    assert limiter.try_acquire() is True
    assert limiter.try_acquire() is True
    assert limiter.try_acquire() is False


def test_try_acquire_refills_over_time() -> None:
    limiter = TokenBucketRateLimiter(max_calls=1, window_seconds=0.1)
    assert limiter.try_acquire() is True
    assert limiter.try_acquire() is False
    time.sleep(0.15)
    assert limiter.try_acquire() is True


def test_seconds_until_next_token_is_zero_when_available() -> None:
    limiter = TokenBucketRateLimiter(max_calls=1, window_seconds=1.0)
    assert limiter.seconds_until_next_token() == 0.0


def test_seconds_until_next_token_is_positive_after_exhausted() -> None:
    limiter = TokenBucketRateLimiter(max_calls=1, window_seconds=1.0)
    limiter.try_acquire()
    wait = limiter.seconds_until_next_token()
    assert 0.0 < wait <= 1.0


def test_try_acquire_is_thread_safe_never_exceeds_capacity() -> None:
    limiter = TokenBucketRateLimiter(max_calls=5, window_seconds=10.0)
    successes = []
    lock = threading.Lock()

    def worker() -> None:
        if limiter.try_acquire():
            with lock:
                successes.append(1)

    threads = [threading.Thread(target=worker) for _ in range(50)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(successes) == 5


# -- ExternalCallQueue -----------------------------------------------------------


def test_rejects_max_queue_size_below_one() -> None:
    limiter = TokenBucketRateLimiter(max_calls=1, window_seconds=1.0)
    with pytest.raises(ValueError, match="maxsize=0"):
        ExternalCallQueue(worker_fn=str, rate_limiter=limiter, max_queue_size=0)


def test_submit_and_resolve_real_worker_end_to_end() -> None:
    limiter = TokenBucketRateLimiter(max_calls=10, window_seconds=1.0)
    queue = ExternalCallQueue(worker_fn=lambda text: text.upper(), rate_limiter=limiter)
    try:
        future = queue.submit("hello")
        assert future.result(timeout=2.0) == "HELLO"
    finally:
        queue.shutdown()


def test_worker_exception_resolves_future_with_that_exception() -> None:
    limiter = TokenBucketRateLimiter(max_calls=10, window_seconds=1.0)

    def failing_worker(text: str) -> str:
        raise RuntimeError(f"boom: {text}")

    queue = ExternalCallQueue(worker_fn=failing_worker, rate_limiter=limiter)
    try:
        future = queue.submit("x")
        with pytest.raises(RuntimeError, match="boom: x"):
            future.result(timeout=2.0)
    finally:
        queue.shutdown()


def test_rate_limiter_actually_throttles_worker_calls() -> None:
    # 2 calls per 0.2s window -- submitting 3 items should take noticeably
    # longer than submitting 2, since the 3rd has to wait for a refill.
    limiter = TokenBucketRateLimiter(max_calls=2, window_seconds=0.2)
    call_times: list[float] = []
    lock = threading.Lock()

    def worker(_: str) -> None:
        with lock:
            call_times.append(time.monotonic())

    queue = ExternalCallQueue(worker_fn=worker, rate_limiter=limiter, max_queue_size=5)
    try:
        futures = [queue.submit(str(i)) for i in range(3)]
        for future in futures:
            future.result(timeout=2.0)
    finally:
        queue.shutdown()

    assert len(call_times) == 3
    # First two should run almost immediately (within the initial burst);
    # the third must wait for at least one refill interval.
    assert call_times[2] - call_times[0] >= 0.1


def test_submit_raises_queue_full_when_backlog_saturated() -> None:
    # Generous rate limit -- `blocking_worker`'s Event is what controls
    # timing here, not the rate limiter (a tight limiter would starve "b"
    # of a token entirely, since this test wants both "a" and "b" to
    # eventually complete once released).
    limiter = TokenBucketRateLimiter(max_calls=10, window_seconds=1.0)
    release = threading.Event()

    def blocking_worker(_: str) -> str:
        release.wait(timeout=5.0)
        return "done"

    queue = ExternalCallQueue(worker_fn=blocking_worker, rate_limiter=limiter, max_queue_size=1)
    try:
        first = queue.submit("a")  # dequeued immediately, blocks in blocking_worker
        time.sleep(0.05)  # let the worker thread actually pick it up
        second = queue.submit("b")  # fills the 1-slot backlog
        with pytest.raises(QueueFullError):
            queue.submit("c")
        release.set()
        assert first.result(timeout=2.0) == "done"
        assert second.result(timeout=2.0) == "done"
    finally:
        release.set()
        queue.shutdown()


def test_shutdown_cancels_items_still_waiting_for_a_rate_limit_token() -> None:
    limiter = TokenBucketRateLimiter(max_calls=1, window_seconds=100.0)  # exhausted after 1
    calls: list[str] = []

    def worker(text: str) -> str:
        calls.append(text)
        return text

    queue = ExternalCallQueue(worker_fn=worker, rate_limiter=limiter, max_queue_size=5)
    try:
        first = queue.submit("a")  # consumes the only token
        second = queue.submit("b")  # will be stuck waiting for a token that never refills in time
        assert first.result(timeout=2.0) == "a"
        time.sleep(0.05)  # let the worker actually start waiting on "b"
    finally:
        queue.shutdown()

    assert second.cancelled()
    assert calls == ["a"]


def test_pending_count_reflects_queued_not_yet_dequeued_items() -> None:
    limiter = TokenBucketRateLimiter(max_calls=1, window_seconds=100.0)
    release = threading.Event()

    def blocking_worker(_: str) -> str:
        release.wait(timeout=5.0)
        return "done"

    queue = ExternalCallQueue(worker_fn=blocking_worker, rate_limiter=limiter, max_queue_size=5)
    try:
        queue.submit("a")  # dequeued, running (holds the worker)
        time.sleep(0.05)
        queue.submit("b")
        queue.submit("c")
        assert queue.pending_count == 2
    finally:
        release.set()
        queue.shutdown()


def test_shutdown_stops_the_worker_thread() -> None:
    limiter = TokenBucketRateLimiter(max_calls=1, window_seconds=1.0)
    queue = ExternalCallQueue(worker_fn=str, rate_limiter=limiter)
    assert queue.is_running
    queue.shutdown()
    assert not queue.is_running
