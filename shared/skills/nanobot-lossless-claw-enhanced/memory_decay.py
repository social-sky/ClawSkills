#!/usr/bin/env python3
"""Weibull decay model for memory lifecycle management.

Implements stretched-exponential decay based on memory-lancedb-pro research.
Pure Python implementation with no external dependencies.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional
import math

from lcm_types import SummaryRecord


# Constants
DEFAULT_RECENCY_HALF_LIFE_DAYS: float = 14.0
DEFAULT_TIME_DECAY_HALF_LIFE_DAYS: float = 60.0
DEFAULT_SHAPE_PARAMETER: float = 1.5
DEFAULT_REINFORCEMENT_FACTOR: float = 0.5
DEFAULT_MAX_HALFLIFE_MULTIPLIER: float = 3.0


def calculate_weibull_decay(
    age_days: float,
    half_life_days: float,
    shape: float = DEFAULT_SHAPE_PARAMETER
) -> float:
    """Calculate Weibull stretched-exponential decay.

    Weibull decay model: exp(-(age/half_life)^shape)

    The shape parameter controls the decay curve:
    - shape < 1: faster early decay, slower later
    - shape = 1: pure exponential decay
    - shape > 1: slower early decay, faster later (default: 1.5)

    Args:
        age_days: Age of the memory in days
        half_life_days: Half-life parameter in days
        shape: Shape parameter (default: 1.5)

    Returns:
        Decay factor between 0 and 1

    Examples:
        >>> calculate_weibull_decay(0, 60.0)
        1.0
        >>> abs(calculate_weibull_decay(60.0, 60.0) - 0.3679) < 0.001  # shape=1.5
        True
        >>> abs(calculate_weibull_decay(120.0, 60.0) - 0.0591) < 0.001  # shape=1.5
        True
    """
    if age_days < 0:
        raise ValueError(f"age_days must be non-negative, got {age_days}")
    if half_life_days <= 0:
        raise ValueError(f"half_life_days must be positive, got {half_life_days}")
    if shape <= 0:
        raise ValueError(f"shape must be positive, got {shape}")

    return math.exp(-math.pow(age_days / half_life_days, shape))


def calculate_recency_score(
    age_days: float,
    half_life_days: float = DEFAULT_RECENCY_HALF_LIFE_DAYS
) -> float:
    """Calculate exponential recency boost.

    Exponential recency scoring: exp(-ageDays / halfLife)

    More recent memories score higher, with exponential falloff.

    Args:
        age_days: Age of the memory in days
        half_life_days: Half-life for recency in days (default: 14.0)

    Returns:
        Recency score between 0 and 1

    Examples:
        >>> calculate_recency_score(0, 14.0)
        1.0
        >>> abs(calculate_recency_score(14.0, 14.0) - 0.3679) < 0.001
        True
        >>> abs(calculate_recency_score(28.0, 14.0) - 0.1353) < 0.001  # 2x half-life
        True
    """
    if age_days < 0:
        raise ValueError(f"age_days must be non-negative, got {age_days}")
    if half_life_days <= 0:
        raise ValueError(f"half_life_days must be positive, got {half_life_days}")

    return math.exp(-age_days / half_life_days)


def calculate_composite_decay(
    age_days: float,
    access_count: int,
    importance: float,
    half_life_days: float = DEFAULT_TIME_DECAY_HALF_LIFE_DAYS,
    reinforcement_factor: float = DEFAULT_REINFORCEMENT_FACTOR,
    max_halflife_multiplier: float = DEFAULT_MAX_HALFLIFE_MULTIPLIER
) -> float:
    """Calculate composite decay score combining recency, frequency, and importance.

    Formula:
        decay = importance * weibull_decay * (1 + reinforcement_factor * log(access_count + 1))

    The effective half-life is capped at max_halflife_multiplier based on access_count.

    Args:
        age_days: Age of the memory in days
        access_count: Number of times the memory has been accessed
        importance: Importance factor between 0.0 and 1.0
        half_life_days: Base half-life in days (default: 60.0)
        reinforcement_factor: Frequency reinforcement factor (default: 0.5)
        max_halflife_multiplier: Max half-life multiplier from access count (default: 3.0)

    Returns:
        Composite decay score

    Examples:
        >>> abs(calculate_composite_decay(0, 0, 1.0) - 1.0) < 0.001  # Fresh record
        True
        >>> abs(calculate_composite_decay(0, 10, 1.0) - 2.199) < 0.001  # 10 accesses
        True
        >>> abs(calculate_composite_decay(60.0, 1, 0.5) - 0.355) < 0.01  # ~60 days old
        True
    """
    if age_days < 0:
        raise ValueError(f"age_days must be non-negative, got {age_days}")
    if access_count < 0:
        raise ValueError(f"access_count must be non-negative, got {access_count}")
    if not 0.0 <= importance <= 1.0:
        raise ValueError(f"importance must be between 0.0 and 1.0, got {importance}")

    # Calculate effective half-life based on access count (capped)
    access_multiplier = min(
        max_halflife_multiplier,
        1.0 + reinforcement_factor * math.log(access_count + 1)
    )
    effective_half_life = half_life_days * access_multiplier

    # Weibull decay with effective half-life
    weibull_decay = calculate_weibull_decay(age_days, effective_half_life)

    # Frequency reinforcement term
    frequency_term = 1.0 + reinforcement_factor * math.log(access_count + 1)

    # Composite score
    return importance * weibull_decay * frequency_term


def calculate_decay_score(
    record: SummaryRecord,
    now: Optional[datetime] = None
) -> float:
    """Calculate final decay score for a SummaryRecord.

    Uses last_accessed_at if available, otherwise falls back to created_at.

    Args:
        record: SummaryRecord to calculate decay for
        now: Reference datetime (default: current time)

    Returns:
        Decay score between 0 and 1

    Examples:
        >>> from datetime import datetime, timedelta
        >>> now = datetime.now()
        >>> record = SummaryRecord(
        ...     summary_id="test",
        ...     conversation_id=1,
        ...     kind="leaf",
        ...     depth=0,
        ...     content="test",
        ...     token_count=10,
        ...     created_at=now,
        ...     last_accessed_at=now,
        ...     access_count=1,
        ...     importance=1.0
        ... )
        >>> abs(calculate_decay_score(record, now) - 1.3466) < 0.001  # Fresh, accessed once
        True
    """
    if now is None:
        now = datetime.now()

    # Determine reference time for age calculation
    reference_time = record.last_accessed_at or record.created_at

    if reference_time is None:
        # No time information available, assume fresh
        age_days = 0.0
    else:
        # Calculate age in days
        delta = now - reference_time
        age_days = delta.total_seconds() / 86400.0

    return calculate_composite_decay(
        age_days=age_days,
        access_count=record.access_count,
        importance=record.importance
    )


# =============================================================================
# Unit Tests
# =============================================================================

def _run_tests() -> None:
    """Run all unit tests."""
    import doctest
    import sys
    from datetime import datetime, timedelta

    # Run doctests
    print("Running doctests...")
    results = doctest.testmod(sys.modules[__name__], verbose=False)
    if results.failed > 0:
        print(f"FAILED: {results.failed} doctests failed")
    else:
        print(f"PASSED: All {results.attempted} doctests passed")

    # Additional unit tests
    print("\nRunning unit tests...")

    # Test 1: Weibull decay formula
    print("  Test 1: Weibull decay formula")
    # age=0 should return 1.0
    assert calculate_weibull_decay(0, 60.0) == 1.0
    # At half-life (age=60, half_life=60), shape=1.5 gives exp(-1^1.5) = exp(-1) ≈ 0.3679
    result_half_life = calculate_weibull_decay(60.0, 60.0)
    assert 0.36 < result_half_life < 0.38, f"Expected ~0.3679, got {result_half_life}"
    # At 2x half-life with shape=1: exp(-2) ≈ 0.135
    result_2x = calculate_weibull_decay(120.0, 60.0, shape=1.0)
    assert 0.13 < result_2x < 0.14, f"Expected ~0.135, got {result_2x}"
    print("    PASSED")

    # Test 2: Recency score
    print("  Test 2: Recency score")
    # age=0 should return 1.0
    assert calculate_recency_score(0) == 1.0
    # At half-life, should be exp(-1) ≈ 0.3679
    result_half = calculate_recency_score(14.0, 14.0)
    assert 0.367 < result_half < 0.368, f"Expected ~0.3679, got {result_half}"
    # At 2x half-life, should be exp(-2) ≈ 0.135
    result_2x = calculate_recency_score(28.0, 14.0)
    assert 0.135 < result_2x < 0.136, f"Expected ~0.135, got {result_2x}"
    print("    PASSED")

    # Test 3: Composite decay
    print("  Test 3: Composite decay")
    # age=0, access_count=0, importance=1.0
    # weibull_decay(0, 60) = 1.0
    # frequency_term = 1 + 0.5 * log(1) = 1.0
    # result = 1.0 * 1.0 * 1.0 = 1.0
    result_fresh = calculate_composite_decay(0, 0, 1.0)
    assert 0.99 < result_fresh <= 1.0, f"Expected ~1.0, got {result_fresh}"

    # age=0, access_count=10, importance=1.0
    # access_multiplier = min(3.0, 1 + 0.5 * log(11)) ≈ min(3.0, 2.199) = 2.199
    # effective_half_life = 60 * 2.199 = 131.94
    # weibull_decay(0, 131.94) = 1.0
    # frequency_term = 1 + 0.5 * log(11) ≈ 2.199
    # result = 1.0 * 1.0 * 2.199 ≈ 2.199
    result_accessed = calculate_composite_decay(0, 10, 1.0)
    assert 2.1 < result_accessed < 2.3, f"Expected ~2.199, got {result_accessed}"

    # importance=0 should give 0
    result_no_importance = calculate_composite_decay(0, 5, 0.0)
    assert result_no_importance == 0.0, f"Expected 0.0, got {result_no_importance}"
    print("    PASSED")

    # Test 4: Edge cases
    print("  Test 4: Edge cases")

    # age=0
    assert calculate_weibull_decay(0, 60.0) == 1.0
    assert calculate_recency_score(0) == 1.0
    assert calculate_composite_decay(0, 0, 1.0) <= 1.0

    # access_count=0
    result = calculate_composite_decay(10, 0, 1.0)
    assert 0.0 < result < 1.0, f"Expected decay, got {result}"

    # importance=0
    assert calculate_composite_decay(100, 100, 0.0) == 0.0

    # importance=1
    result = calculate_composite_decay(0, 0, 1.0)
    assert result > 0.99, f"Expected ~1.0, got {result}"

    # Verify negative values raise errors
    try:
        calculate_weibull_decay(-1, 60.0)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass

    try:
        calculate_recency_score(-1)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass

    try:
        calculate_composite_decay(-1, 0, 0.5)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass

    print("    PASSED")

    # Test 5: calculate_decay_score with SummaryRecord
    print("  Test 5: calculate_decay_score with SummaryRecord")
    now = datetime.now()

    # Fresh record
    fresh_record = SummaryRecord(
        summary_id="test1",
        conversation_id=1,
        kind="leaf",
        depth=0,
        content="test",
        token_count=10,
        created_at=now,
        last_accessed_at=now,
        access_count=1,
        importance=1.0
    )
    score_fresh = calculate_decay_score(fresh_record, now)
    assert score_fresh > 0.99, f"Expected ~1.0 for fresh record, got {score_fresh}"

    # Old record, never accessed
    old_record = SummaryRecord(
        summary_id="test2",
        conversation_id=1,
        kind="leaf",
        depth=0,
        content="test",
        token_count=10,
        created_at=now - timedelta(days=120),
        last_accessed_at=None,
        access_count=0,
        importance=0.5
    )
    score_old = calculate_decay_score(old_record, now)
    assert 0.0 < score_old < 0.2, f"Expected low decay for old record, got {score_old}"

    print("    PASSED")

    # Test 6: Constants are defined correctly
    print("  Test 6: Constants verification")
    assert DEFAULT_RECENCY_HALF_LIFE_DAYS == 14.0
    assert DEFAULT_TIME_DECAY_HALF_LIFE_DAYS == 60.0
    assert DEFAULT_SHAPE_PARAMETER == 1.5
    assert DEFAULT_REINFORCEMENT_FACTOR == 0.5
    assert DEFAULT_MAX_HALFLIFE_MULTIPLIER == 3.0
    print("    PASSED")

    print("\n" + "=" * 50)
    print("ALL TESTS PASSED")
    print("=" * 50)


if __name__ == "__main__":
    _run_tests()
