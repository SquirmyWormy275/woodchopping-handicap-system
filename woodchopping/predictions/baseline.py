"""
Baseline Prediction Functions for Woodchopping Handicap System

This module provides statistical baseline calculations using historical performance data.
It implements cascading fallback logic to handle sparse data scenarios.

Functions:
    calculate_performance_weight() - Calculate exponential time-decay weight for historical results
    get_competitor_historical_times_flexible() - Get competitor's historical times with fallback
    get_event_baseline_flexible() - Calculate event baseline with cascading fallback
"""

import statistics
from typing import List, Tuple, Optional
from datetime import datetime, timedelta
import pandas as pd
import numpy as np


def calculate_performance_weight(
    result_date: Optional[datetime],
    reference_date: Optional[datetime] = None,
    half_life_days: int = 730
) -> float:
    """
    Calculate exponential time-decay weight for a historical performance result.

    Uses exponential decay formula: weight = 0.5^(days_old / half_life_days)

    This ensures recent performances have much greater influence than old performances,
    which is critical for aging competitors whose recent ability differs from their peak.

    Args:
        result_date: Date when the performance occurred (datetime or None)
        reference_date: Date to calculate age from (defaults to today)
        half_life_days: Number of days for weight to decay to 0.5 (default: 730 = 2 years)

    Returns:
        float: Weight between 0.0 and 1.0
            - Returns 1.0 if result_date is None (missing date = no decay)
            - Returns 1.0 if result is from today
            - Returns 0.5 if result is exactly half_life_days old
            - Returns ~0.0 for very old results (15+ years)

    Weight Examples (730-day / 2-year half-life):
        Current season (0-180 days): weight 0.87-1.00
        Last season (365 days): weight 0.71
        2 years ago (730 days): weight 0.50
        3 years ago (1095 days): weight 0.35
        4 years ago (1460 days): weight 0.25
        6 years ago (2190 days): weight 0.125
        10 years ago (3650 days): weight 0.031 (~3% of current)
        15 years ago (5475 days): weight 0.006 (~0.6% of current, essentially zero)

    Example:
        >>> from datetime import datetime
        >>> result_from_2020 = datetime(2020, 7, 15)
        >>> today = datetime(2025, 7, 15)  # 5 years = 1825 days later
        >>> weight = calculate_performance_weight(result_from_2020, today, 730)
        >>> print(f"{weight:.3f}")  # Should be ~0.177 (2^(-1825/730) = 2^(-2.5))
        0.177

    Note:
        - Designed for seasonal sport (American woodchopping: April-September)
        - 730-day (2-year) half-life balances recent trajectory vs career history
        - Performances from 10+ years ago have weight < 0.05
        - Missing dates return weight 1.0 to maintain backward compatibility
    """
    # If no date provided, return full weight (backward compatibility)
    if result_date is None or pd.isna(result_date):
        return 1.0

    # Default to current date if not specified
    if reference_date is None:
        reference_date = datetime.now()

    # Calculate days between dates
    try:
        # Handle both datetime and pandas Timestamp objects
        if isinstance(result_date, str):
            result_date = pd.to_datetime(result_date)
        if isinstance(reference_date, str):
            reference_date = pd.to_datetime(reference_date)

        days_old = (reference_date - result_date).days

        # Can't have negative age (future dates get full weight)
        if days_old < 0:
            return 1.0

        # Exponential decay: weight = 0.5^(days_old / half_life)
        # This is equivalent to: weight = 2^(-days_old / half_life)
        weight = 0.5 ** (days_old / half_life_days)

        return weight

    except Exception as e:
        # If date parsing fails, return full weight
        print(f"Warning: Failed to calculate weight for date {result_date}: {e}")
        return 1.0


def get_competitor_historical_times_flexible(
    competitor_name: str,
    species: str,
    event_code: str,
    results_df: pd.DataFrame,
    return_weights: bool = False,
    reference_date: Optional[datetime] = None
) -> Tuple[List, str]:
    """
    Get competitor's historical times with flexible fallback logic.

    Tries in order:
    1. Exact match: competitor + event + species
    2. Fallback: competitor + event (any species)

    Args:
        competitor_name: Competitor's name
        species: Wood species code
        event_code: Event type (SB or UH)
        results_df: Historical results DataFrame
        return_weights: If True, returns tuples of (time, date, weight) instead of just times
        reference_date: Date to calculate weights from (defaults to today)

    Returns:
        Tuple of (list of data, data source description)

        If return_weights=False (default):
            Returns (List[float], str) - list of times and source description

        If return_weights=True:
            Returns (List[Tuple[float, datetime, float]], str)
            Each tuple is (time, date, weight) where weight is calculated via time-decay

        Returns ([], error_message) if no data found

    Example (backward compatible):
        >>> times, source = get_competitor_historical_times_flexible(
        ...     "John Smith", "White Pine", "SB", results_df
        ... )
        >>> if times:
        ...     avg = statistics.mean(times)

    Example (with weights):
        >>> data, source = get_competitor_historical_times_flexible(
        ...     "John Smith", "White Pine", "SB", results_df, return_weights=True
        ... )
        >>> if data:
        ...     weighted_avg = sum(t*w for t,d,w in data) / sum(w for t,d,w in data)
    """
    if results_df is None or results_df.empty:
        return [], "no data available"

    # Match competitor and event (required)
    name_match = results_df["competitor_name"].astype(str).str.strip().str.lower() == str(competitor_name).strip().lower()
    event_match = results_df["event"].astype(str).str.strip().str.upper() == event_code

    # Try exact species match first
    if species and "species" in results_df.columns:
        species_match = results_df["species"].astype(str).str.strip().str.lower() == str(species).strip().lower()
        exact_matches = results_df[name_match & event_match & species_match]

        data = []
        for _, row in exact_matches.iterrows():
            time = row.get('raw_time')
            if time is not None and time > 0:
                if return_weights:
                    date = row.get('date')
                    weight = calculate_performance_weight(date, reference_date)
                    data.append((time, date, weight))
                else:
                    data.append(time)

        if data:
            return data, f"on {species} (exact match)"

    # Fallback: any species for this competitor and event
    any_species_matches = results_df[name_match & event_match]
    data = []
    for _, row in any_species_matches.iterrows():
        time = row.get('raw_time')
        if time is not None and time > 0:
            if return_weights:
                date = row.get('date')
                weight = calculate_performance_weight(date, reference_date)
                data.append((time, date, weight))
            else:
                data.append(time)

    if data:
        return data, "on various wood types"

    return [], "no competitor history found"


def get_event_baseline_flexible(
    species: str,
    diameter: float,
    event_code: str,
    results_df: pd.DataFrame
) -> Tuple[Optional[float], str]:
    """
    Calculate baseline with cascading fallback.

    Tries in order:
    1. Species + diameter range + event
    2. Diameter range + event (any species)
    3. Event only (any species, any diameter)

    Args:
        species: Wood species code
        diameter: Diameter in mm
        event_code: Event type (SB or UH)
        results_df: Historical results DataFrame

    Returns:
        Tuple of (baseline_time, data source description)
        Returns (None, error_message) if insufficient data

    Example:
        >>> baseline, source = get_event_baseline_flexible(
        ...     "White Pine", 300, "SB", results_df
        ... )
        >>> if baseline:
        ...     print(f"Event baseline: {baseline:.1f}s from {source}")
    """
    if results_df is None or results_df.empty:
        return None, "no data available"

    event_match = results_df["event"].astype(str).str.strip().str.upper() == event_code

    # Try species + diameter range + event
    if species and "species" in results_df.columns and "size_mm" in results_df.columns:
        species_match = results_df["species"].astype(str).str.strip().str.lower() == str(species).strip().lower()
        diameter_match = (results_df["size_mm"] >= diameter - 25) & (results_df["size_mm"] <= diameter + 25)

        exact_matches = results_df[species_match & diameter_match & event_match]
        times = []
        for _, row in exact_matches.iterrows():
            time = row.get('raw_time')
            if time is not None and time > 0:
                times.append(time)

        if len(times) >= 3:
            return statistics.mean(times), f"species/size average ({len(times)} performances)"

    # Fallback: diameter range + event (any species)
    if "size_mm" in results_df.columns:
        diameter_match = (results_df["size_mm"] >= diameter - 25) & (results_df["size_mm"] <= diameter + 25)
        size_matches = results_df[diameter_match & event_match]

        times = []
        for _, row in size_matches.iterrows():
            time = row.get('raw_time')
            if time is not None and time > 0:
                times.append(time)

        if len(times) >= 3:
            return statistics.mean(times), f"size average ({len(times)} performances)"

    # Final fallback: event only (all data for this event type)
    event_only = results_df[event_match]
    times = []
    for _, row in event_only.iterrows():
        time = row.get('raw_time')
        if time is not None and time > 0:
            times.append(time)

    if len(times) >= 3:
        return statistics.mean(times), f"event average ({len(times)} performances)"

    return None, "insufficient data"
