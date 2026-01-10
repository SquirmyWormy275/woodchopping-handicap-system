"""Data validation and cleaning functions for woodchopping competition results."""

import pandas as pd
from typing import List, Tuple, Optional

# Import config
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from config import data_req, events


def validate_results_data(results_df: pd.DataFrame) -> Tuple[Optional[pd.DataFrame], List[str]]:
    """
    Validate and clean historical results data.

    Performs comprehensive validation including:
    - Required columns check
    - Time range validation (0-300s)
    - Diameter validation (150-500mm)
    - Missing data removal
    - Event code validation (SB/UH only)
    - Statistical outlier detection (3x IQR method)

    Args:
        results_df: Raw results DataFrame

    Returns:
        Tuple of (cleaned_df, warnings_list). Returns (None, warnings) if validation fails.
    """
    warnings: List[str] = []

    if results_df is None or results_df.empty:
        return None, ["No data to validate"]

    df = results_df.copy()
    initial_count = len(df)

    # Check for required columns
    required_cols = ['competitor_name', 'event', 'raw_time', 'size_mm', 'species']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        warnings.append(f"Missing columns: {missing_cols}")
        return None, warnings

    # Coerce quality to numeric if present (non-numeric becomes NaN)
    if 'quality' in df.columns:
        df['quality'] = pd.to_numeric(df['quality'], errors='coerce')

    # Remove invalid times (use config constants)
    invalid_times = df[
        (df['raw_time'] <= data_req.MIN_VALID_TIME_SECONDS) |
        (df['raw_time'] > data_req.MAX_VALID_TIME_SECONDS)
    ]
    if not invalid_times.empty:
        warnings.append(
            f"Removed {len(invalid_times)} records with impossible times "
            f"(<{data_req.MIN_VALID_TIME_SECONDS}s or >{data_req.MAX_VALID_TIME_SECONDS}s)"
        )
        df = df[
            (df['raw_time'] > data_req.MIN_VALID_TIME_SECONDS) &
            (df['raw_time'] <= data_req.MAX_VALID_TIME_SECONDS)
        ]

    # Remove invalid sizes (use config constants)
    invalid_sizes = df[
        (df['size_mm'] < data_req.MIN_DIAMETER_MM) |
        (df['size_mm'] > data_req.MAX_DIAMETER_MM)
    ]
    if not invalid_sizes.empty:
        warnings.append(
            f"Removed {len(invalid_sizes)} records with invalid diameters "
            f"(<{data_req.MIN_DIAMETER_MM}mm or >{data_req.MAX_DIAMETER_MM}mm)"
        )
        df = df[
            (df['size_mm'] >= data_req.MIN_DIAMETER_MM) &
            (df['size_mm'] <= data_req.MAX_DIAMETER_MM)
        ]

    # Check for missing competitor names
    missing_names = df[df['competitor_name'].isna() | (df['competitor_name'] == '')]
    if not missing_names.empty:
        warnings.append(f"Removed {len(missing_names)} records with missing competitor names")
        df = df[df['competitor_name'].notna() & (df['competitor_name'] != '')]

    # Check for invalid event codes (use config constants)
    invalid_events = df[~df['event'].isin(events.VALID_EVENTS)]
    if not invalid_events.empty:
        warnings.append(
            f"Removed {len(invalid_events)} records with invalid event codes "
            f"(must be {' or '.join(events.VALID_EVENTS)})"
        )
        df = df[df['event'].isin(events.VALID_EVENTS)]

    # Detect statistical outliers using IQR method (event + size bins when possible)
    outliers_removed = 0
    if 'size_mm' in df.columns:
        df['size_bin'] = (df['size_mm'] / 25.0).round() * 25.0

    for event in events.VALID_EVENTS:
        event_df = df[df['event'] == event]
        if len(event_df) <= 10:
            continue

        # Event-level fallback bounds
        Q1 = event_df['raw_time'].quantile(0.25)
        Q3 = event_df['raw_time'].quantile(0.75)
        IQR = Q3 - Q1
        event_lower = Q1 - data_req.OUTLIER_IQR_MULTIPLIER * IQR
        event_upper = Q3 + data_req.OUTLIER_IQR_MULTIPLIER * IQR

        if 'size_bin' in df.columns:
            for _, group in event_df.groupby('size_bin'):
                if len(group) >= 10:
                    gq1 = group['raw_time'].quantile(0.25)
                    gq3 = group['raw_time'].quantile(0.75)
                    giqr = gq3 - gq1
                    lower = gq1 - data_req.OUTLIER_IQR_MULTIPLIER * giqr
                    upper = gq3 + data_req.OUTLIER_IQR_MULTIPLIER * giqr
                else:
                    lower = event_lower
                    upper = event_upper

                outliers = group[(group['raw_time'] < lower) | (group['raw_time'] > upper)]
                if not outliers.empty:
                    outliers_removed += len(outliers)
                    df = df.drop(index=outliers.index)
        else:
            outliers = event_df[
                (event_df['raw_time'] < event_lower) | (event_df['raw_time'] > event_upper)
            ]
            if not outliers.empty:
                outliers_removed += len(outliers)
                df = df.drop(index=outliers.index)

    if outliers_removed > 0:
        warnings.append(
            f"Removed {outliers_removed} statistical outliers "
            f"(>{data_req.OUTLIER_IQR_MULTIPLIER}x IQR from median)"
        )

    if 'size_bin' in df.columns:
        df = df.drop(columns=['size_bin'])

    final_count = len(df)
    if final_count < initial_count:
        warnings.append(
            f"Data cleaned: {initial_count} -> {final_count} records "
            f"({initial_count - final_count} removed)"
        )

    if final_count < data_req.MIN_ML_TRAINING_RECORDS_TOTAL:
        warnings.append(
            f"Warning: Only {final_count} valid records remaining "
            f"(need {data_req.MIN_ML_TRAINING_RECORDS_TOTAL}+ for ML training)"
        )

    return df, warnings


def standardize_results_data(results_df: pd.DataFrame) -> Tuple[Optional[pd.DataFrame], List[str]]:
    """
    Standardize results data using the shared validation and outlier filtering rules.

    This is a lightweight wrapper around validate_results_data() that:
    - Returns the original DataFrame if validation fails (to avoid hard stops)
    - Tags validated DataFrames to prevent repeated work

    Args:
        results_df: Raw results DataFrame

    Returns:
        Tuple of (cleaned_df_or_original, warnings_list)
    """
    if results_df is None or results_df.empty:
        return results_df, ["No data to validate"]

    # Avoid re-validating the same DataFrame repeatedly
    if getattr(results_df, "attrs", {}).get("validated", False):
        return results_df, []

    cleaned_df, warnings = validate_results_data(results_df)

    if cleaned_df is None or cleaned_df.empty:
        return results_df, warnings

    cleaned_df.attrs["validated"] = True
    return cleaned_df, warnings


def validate_heat_data(heat_assignment_df, wood_selection):
    """
    Validate that heat assignment and wood selection are ready for handicap calculation.

    Args:
        heat_assignment_df: DataFrame of competitors in the heat
        wood_selection: Dictionary with wood characteristics

    Returns:
        bool: True if valid, False if missing required data

    Example:
        >>> if validate_heat_data(heat_df, wood_dict):
        ...     calculate_handicaps(...)
    """
    if heat_assignment_df is None or heat_assignment_df.empty:
        print("\nNo competitors in heat assignment. Use Competitor Menu -> Select Competitors for Heat.")
        return False

    if not wood_selection.get("species") or not wood_selection.get("size_mm"):
        print("\nWood selection incomplete. Use Wood Menu to set species and size.")
        return False

    if not wood_selection.get("event"):
        print("\nEvent not selected. Use Wood Menu -> Select event (SB/UH) or Main Menu option 3.")
        return False

    return True
