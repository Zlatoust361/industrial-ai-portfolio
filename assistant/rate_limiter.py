"""Rate-limited, bounded-wait queue for external LLM API calls (week 11, day 2).

The device this runs on sits on an industrial line with a potentially
unstable/limited internet uplink -- a chat request that falls through to
the external Claude API must never just hang on the network. Two pieces:

- `TokenBucketRateLimiter`: caps how many external calls can go out per
  time window, refilled continuously (not "reset every N seconds", which
  would let a caller burst right at each boundary).
- `ExternalCallQueue`: a bounded backlog + one background worker thread
  that drains it through the rate limiter. `submit()` itself never
  touches the network and never blocks on it -- it either enqueues
  instantly and returns a `concurrent.futures.Future`, or raises
  `QueueFullError` immediately if the backlog is already full. The
  CALLER decides how long (if at all) to wait on that `Future` --
  `llm_parser.py`'s `parse()` waits up to `max_wait_seconds` and returns
  a `"pending"` result rather than blocking further if the network is
  slow or the rate limit is currently exhausted.
"""

from __future__ import annotations

import queue
import threading
import time
from collections.abc import Callable
from concurrent.futures import Future
from typing import TypeVar

R = TypeVar("R")

_POLL_INTERVAL_S = 0.05
"""How often the worker thread re-checks the rate limiter / stop signal
while waiting for a token -- small enough that `shutdown()` stays
responsive, large enough not to busy-spin."""

_QUEUE_GET_TIMEOUT_S = 0.2
"""How long the worker thread blocks on an empty queue between checking
the stop signal -- bounds how long `shutdown()` can take to notice an
idle worker."""


class QueueFullError(RuntimeError):
    """Raised by `ExternalCallQueue.submit` when the backlog is already at `max_queue_size`."""


class TokenBucketRateLimiter:
    """Thread-safe token bucket: at most `max_calls` per `window_seconds`, refilled continuously."""

    def __init__(self, max_calls: int, window_seconds: float) -> None:
        """
        Args:
            max_calls: Bucket capacity -- also the burst limit (a caller
                that hasn't made a call in a while can use up to this
                many tokens back-to-back before being throttled).
            window_seconds: Refill window, in seconds.

        Raises:
            ValueError: If either argument is non-positive.
        """
        if max_calls <= 0:
            raise ValueError(f"max_calls must be positive, got {max_calls}.")
        if window_seconds <= 0:
            raise ValueError(f"window_seconds must be positive, got {window_seconds}.")
        self._max_calls = max_calls
        self._refill_rate = max_calls / window_seconds
        self._tokens = float(max_calls)
        self._last_refill = time.monotonic()
        self._lock = threading.Lock()

    def _refill_locked(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last_refill
        self._tokens = min(self._max_calls, self._tokens + elapsed * self._refill_rate)
        self._last_refill = now

    def try_acquire(self) -> bool:
        """Attempt to consume one token; never blocks.

        Returns:
            `True` if a token was available and consumed, `False` otherwise.
        """
        with self._lock:
            self._refill_locked()
            if self._tokens >= 1.0:
                self._tokens -= 1.0
                return True
            return False

    def seconds_until_next_token(self) -> float:
        """Estimated wait until `try_acquire()` would next succeed.

        Returns:
            `0.0` if a token is available right now, otherwise the
            estimated seconds until one refills. Advisory only -- another
            caller could consume that token first.
        """
        with self._lock:
            self._refill_locked()
            if self._tokens >= 1.0:
                return 0.0
            return (1.0 - self._tokens) / self._refill_rate


class ExternalCallQueue:
    """Bounded backlog + one background worker thread draining it through a rate limiter."""

    def __init__(
        self,
        worker_fn: Callable[[str], R],
        rate_limiter: TokenBucketRateLimiter,
        max_queue_size: int = 20,
    ) -> None:
        """
        Args:
            worker_fn: Called with each submitted item, ON THE WORKER
                THREAD, once a rate-limit token is available. Its return
                value resolves the item's `Future`; an exception it
                raises resolves the `Future` with that exception instead
                of crashing the worker loop.
            rate_limiter: Gates how often `worker_fn` actually runs.
            max_queue_size: Backlog cap. Must be `>= 1` --
                `queue.Queue(maxsize=0)` means UNBOUNDED in the stdlib,
                the opposite of what a bounded backlog needs, so this is
                rejected explicitly rather than silently doing the wrong
                thing.

        Raises:
            ValueError: If `max_queue_size < 1`.
        """
        if max_queue_size < 1:
            raise ValueError(
                f"max_queue_size must be >= 1, got {max_queue_size} "
                "(queue.Queue(maxsize=0) means UNBOUNDED, not zero-capacity)."
            )
        self._worker_fn = worker_fn
        self._rate_limiter = rate_limiter
        self._queue: queue.Queue[tuple[str, Future]] = queue.Queue(maxsize=max_queue_size)
        self._stop_event = threading.Event()
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="external-call-queue-worker"
        )
        self._thread.start()

    def submit(self, item: str) -> Future:
        """Enqueue `item` for processing; never touches the network, never blocks on it.

        Args:
            item: Passed to `worker_fn` once dequeued and rate-limit-admitted.

        Returns:
            A `Future` that resolves to `worker_fn(item)`'s return value
            (or its exception). The caller decides how long to wait on
            it -- `.result(timeout=...)` -- or not wait at all.

        Raises:
            QueueFullError: If the backlog is already at `max_queue_size`
                -- returned immediately, not after any wait.
        """
        future: Future = Future()
        try:
            self._queue.put_nowait((item, future))
        except queue.Full as exc:
            raise QueueFullError(
                f"External-call queue is full ({self._queue.maxsize} pending) -- try again later."
            ) from exc
        return future

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                item, future = self._queue.get(timeout=_QUEUE_GET_TIMEOUT_S)
            except queue.Empty:
                continue

            if future.cancelled():
                continue  # caller gave up on it while it was still queued

            # Wait for a rate-limit token BEFORE marking the future
            # RUNNING: a `Future` can only be cancelled while still
            # PENDING (`Future.cancel()` is a no-op once RUNNING), so
            # doing this the other way around would make `shutdown()`
            # unable to actually cancel an item stuck waiting on a token
            # that isn't refilling -- it would stay RUNNING forever with
            # no result ever set.
            if not self._wait_for_token():
                # shutdown() was called while still waiting -- still PENDING, so this works.
                future.cancel()
                continue

            if not future.set_running_or_notify_cancel():
                continue  # caller cancelled it while we were waiting for a token

            try:
                result = self._worker_fn(item)
            except Exception as exc:  # noqa: BLE001 -- any worker failure must resolve
                future.set_exception(exc)  # the future, never crash this loop or hang it
            else:
                future.set_result(result)

    def _wait_for_token(self) -> bool:
        """Blocks (polling) until a rate-limit token is available or shutdown is requested.

        Returns:
            `True` once a token was acquired, `False` if `shutdown()` was
            called first (the caller should treat the item as cancelled).
        """
        while not self._stop_event.is_set():
            if self._rate_limiter.try_acquire():
                return True
            time.sleep(_POLL_INTERVAL_S)
        return False

    def shutdown(self, wait: bool = True) -> None:
        """Stop the worker thread.

        An item already dequeued and mid-call finishes; one still waiting
        on a rate-limit token is cancelled instead.

        Args:
            wait: If `True`, blocks (briefly) until the worker thread exits.
        """
        self._stop_event.set()
        if wait:
            self._thread.join(timeout=_QUEUE_GET_TIMEOUT_S + 1.0)

    @property
    def pending_count(self) -> int:
        """Number of items currently queued (not yet dequeued by the worker)."""
        return self._queue.qsize()

    @property
    def is_running(self) -> bool:
        """`False` once `shutdown()` has fully stopped the worker thread."""
        return self._thread.is_alive()
