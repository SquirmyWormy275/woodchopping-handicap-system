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

    # Detect statistical outliers using IQR method (per event type)
    outliers_removed = 0
    for event in events.VALID_EVENTS:
        event_data = df[df['event'] == event]['raw_time']
        if len(event_data) > 10:  # Only check if we have enough data
            Q1 = event_data.quantile(0.25)
            Q3 = event_data.quantile(0.75)
            IQR = Q3 - Q1
            # Use config constant for IQR multiplier
            lower_bound = Q1 - data_req.OUTLIER_IQR_MULTIPLIER * IQR
            upper_bound = Q3 + data_req.OUTLIER_IQR_MULTIPLIER * IQR

            outliers = df[(df['event'] == event) &
                         ((df['raw_time'] < lower_bound) | (df['raw_time'] > upper_bound))]
            if not outliers.empty:
                outliers_removed += len(outliers)
                df = df[~((df['event'] == event) &
                         ((df['raw_time'] < lower_bound) | (df['raw_time'] > upper_bound)))]

    if outliers_removed > 0:
        warnings.append(
            f"Removed {outliers_removed} statistical outliers "
            f"(>{data_req.OUTLIER_IQR_MULTIPLIER}x IQR from median)"
        )

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
