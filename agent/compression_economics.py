"""Prompt-cache economics for compaction: does a compaction rewrite repay itself?

Compaction is the ONE sanctioned prompt-cache break (see ``agent/AGENTS.md``). Every commit rewrites
the sent prefix, so the request that follows pays a cache WRITE for that prefix instead of a cache
READ. Providers price a cache write as a multiple of a read (Anthropic-style cache writes cost
~12.5x a read), so a compaction that reclaims very little while rewriting a large prefix can cost
more than it saves for many requests afterwards.

This module decides that trade. It is pure — no I/O, no state, no Hermes imports — so the decision
is unit-testable in isolation and callable from any harness layer.

Two rules shape the design:

* **Window pressure outranks economics.** Deferring is only ever safe while there is headroom left.
  Once usage crosses :data:`PRESSURE_OVERRIDE_FRACTION` of the window the economic gate stands down
  and compaction proceeds, because riding into the provider's hard context limit is strictly worse
  than an unprofitable rewrite. This mirrors SoL-Pi's dual check (economic ratio AND observed
  context pressure) rather than gating on cost alone.
* **Fail open.** Every ambiguous input resolves to "compact now". A caller that cannot measure the
  reclaim should compact, not guess its way into an overflow.

The gate is opt-in: a ``cache_write_read_ratio`` of ``0`` disables it and reproduces the historical
behaviour (compact whenever the trigger fires), so no existing deployment changes behaviour by
upgrading.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

#: ``cache_write_read_ratio`` value that disables the gate. A provider that charges nothing extra
#: for a cache write (ratio 1.0) also disables it: the rewrite penalty is zero.
DISABLED_RATIO = 0.0

#: Cache-write price as a multiple of the cache-read price, used when the caller does not configure
#: one. Matches Anthropic's published 5-minute-cache write multiplier, and SoL-Pi's default.
DEFAULT_CACHE_WRITE_READ_RATIO = 12.5

#: How many subsequent requests the rewrite penalty may take to repay before compaction is deferred.
#: A compaction is a bet on future traffic: reclaiming ``R`` tokens saves ``R`` read-equivalent
#: tokens per later request, while the rewrite costs ``(ratio - 1) * new_prefix`` up front. If that
#: bet needs more than a handful of requests to break even, the reclaim was trivial relative to the
#: prefix being rewritten and the commit is more likely waste than saving.
DEFAULT_PAYBACK_HORIZON_REQUESTS = 3

#: Fraction of the context window at which the economic gate stands down and compaction proceeds
#: regardless of payback. Above this, deferring risks the provider's hard context limit.
PRESSURE_OVERRIDE_FRACTION = 0.85

#: Smallest ratio above 1.0 worth modelling. Below this the rewrite penalty is noise.
_MIN_MODELLED_RATIO = 1.0


def resolve_cache_write_read_ratio(raw: Any) -> float:
    """Coerce a configured ``cache_write_read_ratio`` to a usable float.

    ``None``/unparseable/non-finite/negative values resolve to :data:`DISABLED_RATIO` — the gate is
    opt-in, so a malformed setting must not silently enable an unmodelled cost model.
    """
    if raw is None or isinstance(raw, bool):
        return DISABLED_RATIO
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return DISABLED_RATIO
    if value != value or value in (float("inf"), float("-inf")):  # NaN / inf
        return DISABLED_RATIO
    return value if value > 0 else DISABLED_RATIO


def resolve_payback_horizon_requests(raw: Any) -> int:
    """Coerce a configured horizon to a non-negative int, falling back to the default."""
    if raw is None or isinstance(raw, bool):
        return DEFAULT_PAYBACK_HORIZON_REQUESTS
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return DEFAULT_PAYBACK_HORIZON_REQUESTS
    return max(0, value)


def cache_rewrite_penalty_tokens(new_prefix_tokens: int, ratio: float) -> int:
    """Read-equivalent tokens lost to rewriting ``new_prefix_tokens`` of cache.

    The penalty is the *extra* cost of a write over a read: ``new_prefix * (ratio - 1)``. A ratio at
    or below 1.0 means writes are not priced above reads, so there is nothing to repay.
    """
    try:
        prefix = max(0, int(new_prefix_tokens))
    except (TypeError, ValueError):
        return 0
    if ratio <= _MIN_MODELLED_RATIO or prefix <= 0:
        return 0
    return int(prefix * (ratio - _MIN_MODELLED_RATIO))


def payback_requests(reclaimed_tokens: int, new_prefix_tokens: int, ratio: float) -> float:
    """Requests needed for ``reclaimed_tokens`` of savings to repay the cache rewrite.

    ``0.0`` when there is nothing to repay; ``inf`` when nothing is reclaimed but a rewrite is still
    charged (the caller must treat that as "never repays").
    """
    penalty = cache_rewrite_penalty_tokens(new_prefix_tokens, ratio)
    if penalty <= 0:
        return 0.0
    try:
        reclaimed = int(reclaimed_tokens)
    except (TypeError, ValueError):
        return float("inf")
    if reclaimed <= 0:
        return float("inf")
    return penalty / reclaimed


def evaluate_compaction_economics(
    *,
    reclaimed_tokens: int,
    new_prefix_tokens: int,
    ratio: float,
    horizon_requests: int = DEFAULT_PAYBACK_HORIZON_REQUESTS,
    context_pressure: Optional[float] = None,
) -> Tuple[bool, str]:
    """Decide whether to DEFER a compaction that the trigger already approved.

    Returns ``(defer, reason)``. ``reason`` is ``""`` whenever the caller should proceed, so a
    caller can log it directly on the deferral path only.

    Proceed (never defer) when any of these hold, in order:

    1. the gate is disabled (``ratio <= 0``);
    2. ``context_pressure`` is at or above :data:`PRESSURE_OVERRIDE_FRACTION` — window safety wins;
    3. nothing is reclaimed, or the rewrite carries no penalty;
    4. the penalty repays within ``horizon_requests``.

    A non-finite ``context_pressure`` is treated as "unknown" and ignored rather than as pressure,
    so a caller with a bad window reading still gets the economic gate (which is the conservative
    side for cost, and is itself fail-open on overflow via rule 3's neighbours in the caller).
    """
    if ratio <= 0:
        return False, ""
    pressure = _finite_or_none(context_pressure)
    if pressure is not None and pressure >= PRESSURE_OVERRIDE_FRACTION:
        return False, ""
    try:
        horizon = max(0, int(horizon_requests))
    except (TypeError, ValueError):
        horizon = DEFAULT_PAYBACK_HORIZON_REQUESTS
    payback = payback_requests(reclaimed_tokens, new_prefix_tokens, ratio)
    if payback <= 0:
        return False, ""
    if payback <= horizon:
        return False, ""
    if payback == float("inf"):
        return True, "cache-economics:no-reclaim"
    return True, f"cache-economics:{payback:.1f}req"


def _finite_or_none(value: Any) -> Optional[float]:
    """Return ``value`` as a finite float, or ``None`` when it cannot be trusted."""
    if value is None or isinstance(value, bool):
        return None
    try:
        coerced = float(value)
    except (TypeError, ValueError):
        return None
    if coerced != coerced or coerced in (float("inf"), float("-inf")):
        return None
    return coerced
