"""Feature engineering and preprocessing for machine learning models."""

import pandas as pd
from typing import Optional

# Import config
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from config import ml_config

# Import local modules
from woodchopping.data.excel_io import load_wood_data


def engineer_features_for_ml(
    results_df: pd.DataFrame,
    wood_df: Optional[pd.DataFrame] = None
) -> Optional[pd.DataFrame]:
    """
    Engineer 6 features for ML model from historical results.

    Features created:
    1. competitor_avg_time_by_event - Historical average for this event
    2. event_encoded - Binary encoding (SB=0, UH=1)
    3. size_mm - Block diameter (already present)
    4. wood_janka_hardness - Joined from wood properties
    5. wood_spec_gravity - Joined from wood properties
    6. competitor_experience - Count of past events

    Args:
        results_df: DataFrame with historical results
        wood_df: DataFrame with wood properties (optional, will load if not provided)

    Returns:
        DataFrame with engineered features ready for training/prediction, or None if error
    """
    if results_df is None or results_df.empty:
        return None

    # Load wood data if not provided
    if wood_df is None:
        wood_df = load_wood_data()

    # Create a copy to avoid modifying original
    df = results_df.copy()

    # Ensure required columns exist
    required_cols = ['competitor_name', 'event', 'raw_time', 'size_mm', 'species']
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        print(f"Warning: Missing required columns for ML: {missing}")
        return None

    # Remove invalid records
    df = df[df['raw_time'] > 0].copy()
    df = df[df['size_mm'] > 0].copy()

    if df.empty:
        return None

    # Feature 1: Event type encoding (SB=0, UH=1)
    df['event_encoded'] = df['event'].apply(
        lambda x: ml_config.EVENT_ENCODING_SB if str(x).upper() == 'SB' else ml_config.EVENT_ENCODING_UH
    )

    # Feature 2: Competitor average time by event
    competitor_avg = df.groupby(['competitor_name', 'event'])['raw_time'].transform('mean')
    df['competitor_avg_time_by_event'] = competitor_avg

    # Feature 3: Competitor experience (count of past events)
    df['competitor_experience'] = df.groupby('competitor_name').cumcount() + 1

    # Feature 4: Size (already present as size_mm)

    # Features 5 & 6: Join wood properties (janka_hardness, spec_gravity)
    if not wood_df.empty and 'speciesID' in wood_df.columns:
        # Create species code mapping - use speciesID to match with Results sheet species codes
        wood_properties = wood_df[['speciesID', 'janka_hard', 'spec_gravity']].copy()
        wood_properties = wood_properties.rename(columns={
            'speciesID': 'species',
            'janka_hard': 'wood_janka_hardness',
            'spec_gravity': 'wood_spec_gravity'
        })

        # Join wood properties with results
        df = df.merge(wood_properties, on='species', how='left')

        # Fill missing wood properties with median values
        median_janka = df['wood_janka_hardness'].median()
        median_spec_grav = df['wood_spec_gravity'].median()

        df['wood_janka_hardness'] = df['wood_janka_hardness'].fillna(
            median_janka if pd.notna(median_janka) else ml_config.DEFAULT_JANKA_HARDNESS
        )
        df['wood_spec_gravity'] = df['wood_spec_gravity'].fillna(
            median_spec_grav if pd.notna(median_spec_grav) else ml_config.DEFAULT_SPECIFIC_GRAVITY
        )
    else:
        # If wood data not available, use default values from config
        df['wood_janka_hardness'] = ml_config.DEFAULT_JANKA_HARDNESS
        df['wood_spec_gravity'] = ml_config.DEFAULT_SPECIFIC_GRAVITY

    return df
