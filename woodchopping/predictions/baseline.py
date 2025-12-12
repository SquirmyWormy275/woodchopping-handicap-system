"""
Baseline Prediction Functions for Woodchopping Handicap System

This module provides statistical baseline calculations using historical performance data.
It implements cascading fallback logic to handle sparse data scenarios.

Functions:
    get_competitor_historical_times_flexible() - Get competitor's historical times with fallback
    get_event_baseline_flexible() - Calculate event baseline with cascading fallback
"""

import statistics
from typing import List, Tuple, Optional
import pandas as pd


def get_competitor_historical_times_flexible(
    competitor_name: str,
    species: str,
    event_code: str,
    results_df: pd.DataFrame
) -> Tuple[List[float], str]:
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

    Returns:
        Tuple of (list of times, data source description)
        Returns ([], error_message) if no data found

    Example:
        >>> times, source = get_competitor_historical_times_flexible(
        ...     "John Smith", "White Pine", "SB", results_df
        ... )
        >>> if times:
        ...     avg = statistics.mean(times)
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

        times = []
        for _, row in exact_matches.iterrows():
            time = row.get('raw_time')
            if time is not None and time > 0:
                times.append(time)

        if times:
            return times, f"on {species} (exact match)"

    # Fallback: any species for this competitor and event
    any_species_matches = results_df[name_match & event_match]
    times = []
    for _, row in any_species_matches.iterrows():
        time = row.get('raw_time')
        if time is not None and time > 0:
            times.append(time)

    if times:
        return times, "on various wood types"

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
