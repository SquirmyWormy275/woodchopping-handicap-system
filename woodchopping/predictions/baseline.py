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
from typing import List, Tuple, Optional, Dict
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from config import data_req
from woodchopping.data import standardize_results_data, load_wood_data
from woodchopping.predictions.qaa_scaling import scale_time_qaa

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


def compute_robust_weighted_mean(
    historical_data: List[Tuple[float, Optional[datetime], float]]
) -> Tuple[Optional[float], float, float]:
    """
    Compute a robust weighted mean from (time, date, weight) tuples.

    Returns:
        Tuple of (mean_or_none, avg_weight, effective_n)
    """
    if not historical_data:
        return None, 0.0, 0.0

    times = np.array([t for t, _, _ in historical_data], dtype=float)
    weights = np.array([w for _, _, w in historical_data], dtype=float)

    weight_sum = weights.sum()
    if weight_sum <= 0:
        return float(np.mean(times)), 0.0, 0.0

    # Use median/MAD clipping to reduce influence of extreme values
    median = float(np.median(times))
    if len(times) < 5:
        return median, float(weight_sum / len(times)), float(weight_sum)

    mad = float(np.median(np.abs(times - median)))
    if mad > 0:
        clip_low = median - 2.5 * mad
        clip_high = median + 2.5 * mad
        clipped = np.clip(times, clip_low, clip_high)
    else:
        clipped = times

    mean = float(np.average(clipped, weights=weights))
    avg_weight = float(weight_sum / len(times))
    effective_n = float(weight_sum)
    return mean, avg_weight, effective_n


def compute_robust_mean(times: List[float]) -> Optional[float]:
    """Compute a robust mean using the weighted helper with unit weights."""
    if not times:
        return None
    data = [(t, None, 1.0) for t in times]
    mean, _, _ = compute_robust_weighted_mean(data)
    return mean


def apply_shrinkage(
    competitor_baseline: float,
    effective_n: float,
    event_baseline: Optional[float],
    shrinkage_k: float = 5.0
) -> float:
    """
    Apply simple empirical-Bayes shrinkage toward event baseline.
    """
    if event_baseline is None or effective_n <= 0:
        return competitor_baseline

    weight = effective_n / (effective_n + shrinkage_k)
    return (weight * competitor_baseline) + ((1.0 - weight) * event_baseline)


_wood_hardness_cache: Optional[Dict[str, float]] = None
_species_exponent_cache: Dict[str, float] = {}


def _build_wood_hardness_index(wood_df: pd.DataFrame) -> Dict[str, float]:
    """
    Build a composite hardness index using all available wood properties.

    Returns a mapping of speciesID -> hardness_factor (around 1.0).
    """
    if wood_df is None or wood_df.empty:
        return {}

    numeric_cols = ['janka_hard', 'spec_gravity', 'crush_strength', 'shear', 'MOR', 'MOE']
    available = [c for c in numeric_cols if c in wood_df.columns]
    if not available:
        return {}

    df = wood_df.copy()
    for col in available:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Z-score normalize each column to combine scales
    zcols = []
    for col in available:
        mean = df[col].mean()
        std = df[col].std()
        if std and std > 0:
            z = (df[col] - mean) / std
        else:
            z = pd.Series(0.0, index=df.index)
        zcols.append(z)

    if not zcols:
        return {}

    composite = sum(zcols) / len(zcols)

    # Map composite z-score to a bounded factor
    factor = (1.0 + (composite * 0.05)).clip(lower=0.85, upper=1.15)

    hardness_map: Dict[str, float] = {}
    if 'speciesID' in df.columns:
        for idx, row in df.iterrows():
            code = str(row['speciesID']).strip()
            if code:
                hardness_map[code] = float(factor.loc[idx])

    return hardness_map


def get_species_hardness_factor(species_code: str, wood_df: Optional[pd.DataFrame] = None) -> float:
    """Return composite hardness factor for a species (defaults to 1.0)."""
    global _wood_hardness_cache
    if wood_df is None:
        wood_df = load_wood_data()

    if _wood_hardness_cache is None:
        _wood_hardness_cache = _build_wood_hardness_index(wood_df)

    return _wood_hardness_cache.get(str(species_code).strip(), 1.0)


def calibrate_species_exponent(
    results_df: pd.DataFrame,
    wood_df: Optional[pd.DataFrame],
    event_code: str,
    min_species_samples: int = 5
) -> float:
    """
    Calibrate exponent for species hardness normalization from historical data.
    """
    cache_key = str(event_code).upper()
    if cache_key in _species_exponent_cache:
        return _species_exponent_cache[cache_key]

    if results_df is None or results_df.empty:
        _species_exponent_cache[cache_key] = 0.4
        return 0.4

    df = results_df[results_df['event'] == event_code].copy()
    if df.empty or 'species' not in df.columns or 'size_mm' not in df.columns:
        _species_exponent_cache[cache_key] = 0.4
        return 0.4

    wood_df = wood_df if wood_df is not None else load_wood_data()

    # Normalize all times to 300mm to isolate species effect
    def _scale_to_standard(row):
        time_val = row.get('raw_time')
        size_val = row.get('size_mm')
        species_val = row.get('species')
        if pd.isna(time_val) or pd.isna(size_val):
            return np.nan
        return scale_time_qaa(
            float(time_val),
            float(size_val),
            300.0,
            str(species_val).strip(),
            quality=5,
            wood_df=wood_df
        )[0]

    df['scaled_time'] = df.apply(_scale_to_standard, axis=1)

    # Aggregate by species
    grouped = df.groupby('species')['scaled_time'].agg(['median', 'count']).reset_index()
    grouped = grouped[grouped['count'] >= min_species_samples].copy()
    if grouped.empty:
        _species_exponent_cache[cache_key] = 0.4
        return 0.4

    grouped['hardness_factor'] = grouped['species'].apply(
        lambda s: get_species_hardness_factor(s, wood_df)
    )

    # Fit exponent by minimizing MAE in log space
    hardness = grouped['hardness_factor'].values.astype(float)
    times = grouped['median'].values.astype(float)
    counts = grouped['count'].values.astype(float)

    if np.any(hardness <= 0) or np.any(times <= 0):
        _species_exponent_cache[cache_key] = 0.4
        return 0.4

    log_h = np.log(hardness)
    log_t = np.log(times)

    if np.allclose(log_h, 0):
        _species_exponent_cache[cache_key] = 0.4
        return 0.4

    weights = counts / counts.sum() if counts.sum() > 0 else None

    best_exp = 0.4
    best_err = float('inf')
    candidates = np.arange(0.1, 2.51, 0.05)

    for exp in candidates:
        residuals = log_t - (exp * log_h)
        if weights is not None:
            log_k = float(np.average(residuals, weights=weights))
            pred = log_k + (exp * log_h)
            err = float(np.average(np.abs(log_t - pred), weights=weights))
        else:
            log_k = float(np.mean(residuals))
            pred = log_k + (exp * log_h)
            err = float(np.mean(np.abs(log_t - pred)))

        if err < best_err:
            best_err = err
            best_exp = exp

    exponent = float(np.clip(best_exp, 0.1, 2.5))
    _species_exponent_cache[cache_key] = exponent
    return exponent


def normalize_time_to_target(
    time_val: float,
    hist_species: str,
    hist_diameter: float,
    target_species: str,
    target_diameter: float,
    event_code: str,
    wood_df: Optional[pd.DataFrame],
    results_df: Optional[pd.DataFrame] = None,
    quality: Optional[float] = None
) -> Tuple[float, str]:
    """
    Normalize a historical time to target species and diameter.
    """
    if time_val is None:
        return time_val, "no time"

    quality_val = int(quality) if quality is not None and not pd.isna(quality) else 5
    quality_val = max(1, min(10, quality_val))

    normalized = float(time_val)
    notes = []

    # Normalize historical time to quality 5 reference
    if quality_val != 5:
        quality_offset = quality_val - 5
        quality_factor = 1.0 + (quality_offset * 0.02)
        if quality_factor > 0:
            normalized = normalized / quality_factor
            notes.append(f"Quality normalized: {quality_val} -> 5")

    if hist_diameter and target_diameter and hist_diameter != target_diameter:
        normalized, explanation = scale_time_qaa(
            float(time_val),
            float(hist_diameter),
            float(target_diameter),
            str(hist_species).strip() if hist_species is not None else "",
            quality=5,
            wood_df=wood_df
        )
        notes.append(explanation)

    hist_species_clean = str(hist_species).strip() if hist_species is not None and not pd.isna(hist_species) else ""
    target_species_clean = str(target_species).strip() if target_species is not None else ""

    if hist_species_clean and target_species_clean:
        hist_factor = get_species_hardness_factor(hist_species_clean, wood_df)
        target_factor = get_species_hardness_factor(target_species_clean, wood_df)

        if hist_factor > 0 and target_factor > 0 and hist_factor != target_factor:
            exponent = calibrate_species_exponent(
                results_df=results_df,
                wood_df=wood_df,
                event_code=event_code
            )
            normalized = normalized * ((target_factor / hist_factor) ** exponent)
            notes.append(f"Species normalization: {hist_factor:.3f} -> {target_factor:.3f} (exp {exponent:.2f})")

    return normalized, "; ".join(notes)


def get_competitor_historical_times_normalized(
    competitor_name: str,
    species: str,
    diameter: float,
    event_code: str,
    results_df: pd.DataFrame,
    return_weights: bool = False,
    reference_date: Optional[datetime] = None,
    wood_df: Optional[pd.DataFrame] = None
) -> Tuple[List, str, Dict[str, float]]:
    """
    Get competitor historical times normalized to target species/diameter.
    """
    if results_df is None or results_df.empty:
        return [], "no data available", {'scaled': False, 'max_diameter_diff': 0.0, 'species_normalized': False}

    results_df, _ = standardize_results_data(results_df)
    wood_df = wood_df if wood_df is not None else load_wood_data()

    name_match = results_df["competitor_name"].astype(str).str.strip().str.lower() == str(competitor_name).strip().lower()
    event_match = results_df["event"].astype(str).str.strip().str.upper() == event_code

    # Try exact species match first
    if species and "species" in results_df.columns:
        species_match = results_df["species"].astype(str).str.strip().str.lower() == str(species).strip().lower()
        matches = results_df[name_match & event_match & species_match]
        data_source = f"on {species} (exact match)"
    else:
        matches = results_df[name_match & event_match]
        data_source = "on various wood types"

    if matches.empty or len(matches) < data_req.MIN_HISTORICAL_TIMES:
        # Fallback: include any species for competitor+event if exact data is sparse
        any_species = results_df[name_match & event_match]
        if not any_species.empty:
            matches = pd.concat([matches, any_species]).drop_duplicates()
            data_source = "mixed species (normalized)"

    data = []
    meta = {'scaled': False, 'max_diameter_diff': 0.0, 'species_normalized': False}

    for _, row in matches.iterrows():
        time_val = row.get('raw_time')
        if time_val is None or time_val <= 0:
            continue

        hist_species = row.get('species')
        hist_diameter = row.get('size_mm')
        hist_quality = row.get('quality', 5)

        normalized, _ = normalize_time_to_target(
            float(time_val),
            str(hist_species).strip(),
            float(hist_diameter) if hist_diameter is not None else None,
            str(species).strip(),
            float(diameter) if diameter is not None else None,
            event_code,
            wood_df,
            results_df=results_df,
            quality=hist_quality
        )

        if hist_diameter is not None and diameter is not None:
            diff = abs(float(hist_diameter) - float(diameter))
            if diff > 0:
                meta['scaled'] = True
                meta['max_diameter_diff'] = max(meta['max_diameter_diff'], diff)

        if (
            hist_species is not None and not pd.isna(hist_species)
            and species and str(hist_species).strip().lower() != str(species).strip().lower()
        ):
            meta['species_normalized'] = True

        if return_weights:
            date = row.get('date')
            weight = calculate_performance_weight(date, reference_date)
            data.append((normalized, date, weight))
        else:
            data.append(normalized)

    if data:
        return data, data_source, meta

    return [], "no competitor history found", meta


def predict_baseline_time(
    competitor_name: str,
    species: str,
    diameter: float,
    quality: Optional[float],
    event_code: str,
    results_df: pd.DataFrame,
    wood_df: Optional[pd.DataFrame] = None
) -> Tuple[Optional[float], str, str]:
    """
    Predict time using baseline-only logic with normalization.

    Returns:
        Tuple of (predicted_time, confidence, explanation)
    """
    if results_df is None or results_df.empty:
        return None, "LOW", "no data available"

    results_df, _ = standardize_results_data(results_df)
    wood_df = wood_df if wood_df is not None else load_wood_data()

    historical_data, data_source, normalization_meta = get_competitor_historical_times_normalized(
        competitor_name,
        species,
        diameter,
        event_code,
        results_df,
        return_weights=True,
        wood_df=wood_df
    )

    if len(historical_data) >= 3:
        baseline, avg_weight, effective_n = compute_robust_weighted_mean(historical_data)
        event_baseline, event_source = get_event_baseline_flexible(species, diameter, event_code, results_df, wood_df=wood_df)
        if event_baseline is not None and baseline is not None:
            baseline = apply_shrinkage(baseline, effective_n, event_baseline)
            explanation = f"Robust history ({data_source}, {len(historical_data)} results) + shrinkage to {event_source}"
        else:
            explanation = f"Robust history ({data_source}, {len(historical_data)} results)"
        confidence = "HIGH"
    elif len(historical_data) > 0:
        baseline, avg_weight, effective_n = compute_robust_weighted_mean(historical_data)
        event_baseline, event_source = get_event_baseline_flexible(species, diameter, event_code, results_df, wood_df=wood_df)
        if event_baseline is not None and baseline is not None:
            baseline = apply_shrinkage(baseline, effective_n, event_baseline)
            explanation = f"Limited history ({data_source}, {len(historical_data)} results) + shrinkage to {event_source}"
        else:
            explanation = f"Limited history ({data_source}, {len(historical_data)} results)"
        confidence = "MEDIUM"
    else:
        baseline, baseline_source = get_event_baseline_flexible(species, diameter, event_code, results_df, wood_df=wood_df)
        if baseline is not None:
            explanation = f"Event baseline ({baseline_source})"
            confidence = "LOW"
        else:
            baseline = None
            explanation = "No baseline available"
            confidence = "LOW"

    if baseline is None:
        return None, confidence, explanation

    quality_val = int(quality) if quality is not None and not pd.isna(quality) else 5
    quality_val = max(1, min(10, quality_val))
    if quality_val != 5:
        quality_offset = quality_val - 5
        quality_factor = 1.0 + (quality_offset * 0.02)
        baseline = baseline * quality_factor
        adjustment_pct = (quality_factor - 1.0) * 100
        if quality_val < 5:
            explanation += f" [Quality {quality_val}/10: softer, {adjustment_pct:+.0f}%]"
        else:
            explanation += f" [Quality {quality_val}/10: harder, {adjustment_pct:+.0f}%]"

    if normalization_meta.get('max_diameter_diff', 0.0) > 25 or normalization_meta.get('species_normalized', False):
        if confidence == "HIGH":
            confidence = "MEDIUM"
        elif confidence == "MEDIUM":
            confidence = "LOW"
        explanation += " [Normalized sizes/species]"

    return baseline, confidence, explanation


def backtest_baseline_predictions(
    results_df: pd.DataFrame,
    wood_df: Optional[pd.DataFrame] = None,
    event_code: Optional[str] = None,
    limit: Optional[int] = None
) -> Dict[str, float]:
    """
    Backtest baseline predictions using leave-one-out evaluation.
    """
    if results_df is None or results_df.empty:
        return {'count': 0}

    results_df, _ = standardize_results_data(results_df)
    wood_df = wood_df if wood_df is not None else load_wood_data()
    if event_code:
        results_df = results_df[results_df['event'] == event_code].copy()

    if results_df.empty:
        return {'count': 0}

    if limit and limit < len(results_df):
        results_df = results_df.sample(n=limit, random_state=42)

    errors = []
    abs_errors = []
    sq_errors = []
    pct_errors = []

    for idx, row in results_df.iterrows():
        train_df = results_df.drop(index=idx)
        pred, _, _ = predict_baseline_time(
            row.get('competitor_name'),
            row.get('species'),
            row.get('size_mm'),
            row.get('quality'),
            row.get('event'),
            train_df,
            wood_df=wood_df
        )

        actual = row.get('raw_time')
        if pred is None or actual is None or actual <= 0:
            continue

        err = pred - actual
        errors.append(err)
        abs_errors.append(abs(err))
        sq_errors.append(err ** 2)
        if actual > 0:
            pct_errors.append(abs(err) / actual)

    count = len(errors)
    if count == 0:
        return {'count': 0}

    mae = float(np.mean(abs_errors))
    rmse = float(np.sqrt(np.mean(sq_errors)))
    mape = float(np.mean(pct_errors)) * 100 if pct_errors else 0.0
    bias = float(np.mean(errors))

    return {
        'count': count,
        'mae': mae,
        'rmse': rmse,
        'mape': mape,
        'bias': bias
    }


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

    results_df, _ = standardize_results_data(results_df)

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
    results_df: pd.DataFrame,
    wood_df: Optional[pd.DataFrame] = None
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

    results_df, _ = standardize_results_data(results_df)
    wood_df = wood_df if wood_df is not None else load_wood_data()

    event_match = results_df["event"].astype(str).str.strip().str.upper() == event_code

    # Try species + diameter range + event
    if species and "species" in results_df.columns and "size_mm" in results_df.columns:
        species_match = results_df["species"].astype(str).str.strip().str.lower() == str(species).strip().lower()
        diameter_match = (results_df["size_mm"] >= diameter - 25) & (results_df["size_mm"] <= diameter + 25)

        exact_matches = results_df[species_match & diameter_match & event_match]
        times = []
        for _, row in exact_matches.iterrows():
            time_val = row.get('raw_time')
            if time_val is not None and time_val > 0:
                normalized, _ = normalize_time_to_target(
                    float(time_val),
                    str(row.get('species')).strip(),
                    float(row.get('size_mm')) if row.get('size_mm') is not None else None,
                    str(species).strip(),
                    float(diameter),
                    event_code,
                    wood_df,
                    results_df=results_df,
                    quality=row.get('quality', 5)
                )
                times.append(normalized)

        if len(times) >= 3:
            mean_val = compute_robust_mean(times)
            return mean_val, f"species/size normalized average ({len(times)} performances)"

    # Fallback: diameter range + event (any species)
    if "size_mm" in results_df.columns:
        diameter_match = (results_df["size_mm"] >= diameter - 25) & (results_df["size_mm"] <= diameter + 25)
        size_matches = results_df[diameter_match & event_match]

        times = []
        for _, row in size_matches.iterrows():
            time_val = row.get('raw_time')
            if time_val is not None and time_val > 0:
                normalized, _ = normalize_time_to_target(
                    float(time_val),
                    str(row.get('species')).strip(),
                    float(row.get('size_mm')) if row.get('size_mm') is not None else None,
                    str(species).strip(),
                    float(diameter),
                    event_code,
                    wood_df,
                    results_df=results_df,
                    quality=row.get('quality', 5)
                )
                times.append(normalized)

        if len(times) >= 3:
            mean_val = compute_robust_mean(times)
            return mean_val, f"size normalized average ({len(times)} performances)"

    # Final fallback: event only (all data for this event type)
    event_only = results_df[event_match]
    times = []
    for _, row in event_only.iterrows():
        time_val = row.get('raw_time')
        if time_val is not None and time_val > 0:
            normalized, _ = normalize_time_to_target(
                float(time_val),
                str(row.get('species')).strip(),
                float(row.get('size_mm')) if row.get('size_mm') is not None else None,
                str(species).strip(),
                float(diameter),
                event_code,
                wood_df,
                results_df=results_df,
                quality=row.get('quality', 5)
            )
            times.append(normalized)

    if len(times) >= 3:
        mean_val = compute_robust_mean(times)
        return mean_val, f"event normalized average ({len(times)} performances)"

    return None, "insufficient data"
