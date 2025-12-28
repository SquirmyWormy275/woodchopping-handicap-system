"""
Machine Learning Prediction Model for Woodchopping Handicap System

This module provides XGBoost-based time prediction using historical performance data.
Separate models are trained for Standing Block (SB) and Underhand (UH) events.

Functions:
    train_ml_model() - Train XGBoost models for SB and UH events
    predict_time_ml() - Predict time using trained ML model
    perform_cross_validation() - K-fold cross-validation for model accuracy
    display_feature_importance() - Display feature importance rankings
"""

from typing import Optional, Dict, Tuple
import pandas as pd

# Import config
from config import data_req

# Import local data modules
from woodchopping.data import (
    load_results_df,
    load_wood_data,
    validate_results_data,
    engineer_features_for_ml
)

# Import time-decay weighting function
from woodchopping.predictions.baseline import calculate_performance_weight

# ML Libraries
try:
    import xgboost as xgb
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import mean_absolute_error, r2_score
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    print("Warning: XGBoost/scikit-learn not available. ML predictions disabled.")


# Global cache for trained models
_cached_ml_model_sb = None
_cached_ml_model_uh = None
_model_training_data_size = 0
_feature_importance_sb = None
_feature_importance_uh = None


def perform_cross_validation(X, y, model_params: dict, cv_folds: int = 5) -> Optional[dict]:
    """
    Perform k-fold cross-validation to estimate model accuracy.

    Args:
        X: Feature matrix
        y: Target vector
        model_params: XGBoost parameters dict
        cv_folds: Number of cross-validation folds

    Returns:
        dict: CV results with mean and std of metrics
            {
                'mae_mean': float,
                'mae_std': float,
                'r2_mean': float,
                'r2_std': float,
                'mae_scores': array,
                'r2_scores': array
            }
        Returns None if ML libraries not available
    """
    if not ML_AVAILABLE:
        return None

    # Create model with same params as training
    model = xgb.XGBRegressor(**model_params)

    # Perform cross-validation for MAE
    mae_scores = cross_val_score(model, X, y, cv=cv_folds,
                                  scoring='neg_mean_absolute_error',
                                  n_jobs=-1)

    # Perform cross-validation for R²
    r2_scores = cross_val_score(model, X, y, cv=cv_folds,
                                 scoring='r2',
                                 n_jobs=-1)

    return {
        'mae_mean': -mae_scores.mean(),
        'mae_std': mae_scores.std(),
        'r2_mean': r2_scores.mean(),
        'r2_std': r2_scores.std(),
        'mae_scores': -mae_scores,
        'r2_scores': r2_scores
    }


def display_feature_importance(model, event_code: str, feature_names: list) -> None:
    """
    Display feature importance from trained XGBoost model.

    Args:
        model: Trained XGBoost model
        event_code: 'SB' or 'UH'
        feature_names: List of feature names
    """
    global _feature_importance_sb, _feature_importance_uh

    if model is None:
        return

    importance = model.feature_importances_

    # Sort by importance
    importance_pairs = sorted(zip(feature_names, importance),
                             key=lambda x: x[1], reverse=True)

    print(f"\n{'='*70}")
    print(f"  FEATURE IMPORTANCE - {event_code} MODEL")
    print(f"{'='*70}")
    print(f"\n{'Feature':<40} {'Importance':>10}")
    print("-" * 70)

    for name, score in importance_pairs:
        # Create a simple bar chart
        bar_length = int(score * 40)
        bar = '#' * bar_length
        print(f"{name:<40} {score:>9.3f}  {bar}")

    print(f"{'='*70}")

    # Store for later reference
    if event_code == 'SB':
        _feature_importance_sb = dict(importance_pairs)
    else:
        _feature_importance_uh = dict(importance_pairs)


def train_ml_model(
    results_df: Optional[pd.DataFrame] = None,
    wood_df: Optional[pd.DataFrame] = None,
    force_retrain: bool = False,
    event_code: Optional[str] = None
) -> Optional[Dict[str, object]]:
    """
    Train separate XGBoost models for SB and UH events.

    Args:
        results_df: Historical results DataFrame (will load if not provided)
        wood_df: Wood properties DataFrame (will load if not provided)
        force_retrain: Force retraining even if models are cached
        event_code: Specific event to train ('SB' or 'UH'), or None for both

    Returns:
        dict with 'SB' and 'UH' keys containing trained models, or None if insufficient data

    Example:
        >>> models = train_ml_model()
        >>> if models:
        ...     sb_model = models['SB']
        ...     uh_model = models['UH']
    """
    global _cached_ml_model_sb, _cached_ml_model_uh, _model_training_data_size
    global _feature_importance_sb, _feature_importance_uh

    # Check if ML libraries available
    if not ML_AVAILABLE:
        return None

    # Load data if not provided
    if results_df is None:
        results_df = load_results_df()

    if results_df is None or results_df.empty:
        return None

    # Validate and clean data
    print("\n[DATA VALIDATION]")
    validated_df, warnings = validate_results_data(results_df)

    if warnings:
        print(f"Data validation warnings ({len(warnings)} issues):")
        for i, warning in enumerate(warnings[:5], 1):  # Show first 5
            print(f"  {i}. {warning}")
        if len(warnings) > 5:
            print(f"  ... and {len(warnings) - 5} more")

    if validated_df is None or validated_df.empty:
        print("ERROR: No valid data after validation")
        return None

    print(f"Valid records: {len(validated_df)} / {len(results_df)} ({len(results_df) - len(validated_df)} removed)")

    # Check if we can use cached models
    if not force_retrain and _cached_ml_model_sb is not None and _cached_ml_model_uh is not None:
        if len(validated_df) == _model_training_data_size:
            return {'SB': _cached_ml_model_sb, 'UH': _cached_ml_model_uh}

    # Engineer features
    df_engineered = engineer_features_for_ml(validated_df, wood_df)

    if df_engineered is None or len(df_engineered) < data_req.MIN_ML_TRAINING_RECORDS_TOTAL:
        print(f"Insufficient data for ML training: {len(df_engineered) if df_engineered is not None else 0} records (need {data_req.MIN_ML_TRAINING_RECORDS_TOTAL}+)")
        return None

    # Define features and target
    feature_cols = [
        'competitor_avg_time_by_event',
        'event_encoded',
        'size_mm',
        'wood_janka_hardness',
        'wood_spec_gravity',
        'competitor_experience'
    ]

    # Ensure all feature columns exist
    missing = [col for col in feature_cols if col not in df_engineered.columns]
    if missing:
        print(f"Warning: Missing feature columns: {missing}")
        return None

    # Determine which events to train
    events_to_train = []
    if event_code:
        events_to_train = [event_code.upper()]
    else:
        events_to_train = ['SB', 'UH']

    models = {}

    for event in events_to_train:
        print(f"\n[TRAINING {event} MODEL]")

        # Filter data for this event
        event_df = df_engineered[df_engineered['event'] == event].copy()

        if len(event_df) < data_req.MIN_ML_TRAINING_RECORDS_PER_EVENT:
            print(f"Insufficient {event} data: {len(event_df)} records (need {data_req.MIN_ML_TRAINING_RECORDS_PER_EVENT}+)")
            continue

        X = event_df[feature_cols]
        y = event_df['raw_time']

        # Calculate TIME-DECAY WEIGHTS for each training sample
        # Recent performances get higher weight during training
        # Critical for aging competitors: their 10-year-old peak performances shouldn't dominate the model
        sample_weights = event_df['date'].apply(
            lambda d: calculate_performance_weight(d, half_life_days=730)
        )

        # Remove any rows with NaN values
        mask = ~(X.isna().any(axis=1) | y.isna())
        X = X[mask]
        y = y[mask]
        sample_weights = sample_weights[mask]

        if len(X) < data_req.MIN_ML_TRAINING_RECORDS_PER_EVENT:
            print(f"Insufficient valid {event} data after cleaning: {len(X)} records")
            continue

        # Report on time-decay weighting effectiveness
        avg_weight = sample_weights.mean()
        recent_fraction = (sample_weights > 0.5).sum() / len(sample_weights)
        print(f"  Time-decay: avg weight {avg_weight:.2f}, {recent_fraction*100:.0f}% of data from last 2 years")

        # Model parameters
        model_params = {
            'n_estimators': 100,
            'max_depth': 4,
            'learning_rate': 0.1,
            'random_state': 42,
            'objective': 'reg:squarederror',
            'tree_method': 'hist'
        }

        # Perform cross-validation (without weights for simplicity)
        # Note: Could add weighted CV in future if needed
        print(f"Cross-validating {event} model (5-fold)...")
        cv_results = perform_cross_validation(X, y, model_params, cv_folds=5)

        if cv_results:
            print(f"  CV MAE: {cv_results['mae_mean']:.2f}s +/- {cv_results['mae_std']:.2f}s")
            print(f"  CV R2:  {cv_results['r2_mean']:.3f} +/- {cv_results['r2_std']:.3f}")

        # Train final model on all data WITH TIME-DECAY SAMPLE WEIGHTS
        # Recent training examples influence the model more than old examples
        model = xgb.XGBRegressor(**model_params)
        model.fit(X, y, sample_weight=sample_weights)

        # Calculate training metrics
        y_pred = model.predict(X)
        mae = mean_absolute_error(y, y_pred)
        r2 = r2_score(y, y_pred)

        print(f"Final {event} model: {len(X)} records (MAE: {mae:.2f}s, R2: {r2:.3f})")

        # Display feature importance
        display_feature_importance(model, event, feature_cols)

        # Cache the model
        if event == 'SB':
            _cached_ml_model_sb = model
            _feature_importance_sb = model.feature_importances_
        else:  # UH
            _cached_ml_model_uh = model
            _feature_importance_uh = model.feature_importances_

        models[event] = model

    # Update training data size
    _model_training_data_size = len(validated_df)

    if not models:
        print("\nERROR: Failed to train any models")
        return None

    # Return both models (use cached if one wasn't trained)
    return {
        'SB': models.get('SB', _cached_ml_model_sb),
        'UH': models.get('UH', _cached_ml_model_uh)
    }


def predict_time_ml(
    competitor_name: str,
    species: str,
    diameter: float,
    quality: int,
    event_code: str,
    results_df: Optional[pd.DataFrame] = None,
    wood_df: Optional[pd.DataFrame] = None
) -> Tuple[Optional[float], Optional[str], str]:
    """
    Predict time using event-specific trained ML model (separate SB/UH models).

    Args:
        competitor_name: Competitor's name
        species: Wood species code
        diameter: Diameter in mm
        quality: Wood quality (0-10) - not used directly but for consistency
        event_code: Event type (SB or UH)
        results_df: Historical results (optional)
        wood_df: Wood properties (optional)

    Returns:
        tuple: (predicted_time, confidence, explanation) or (None, None, error_msg) if error

    Example:
        >>> time, conf, exp = predict_time_ml("John Smith", "WP", 300, 5, "SB")
        >>> if time:
        ...     print(f"ML prediction: {time:.1f}s (confidence: {conf})")
    """
    global _cached_ml_model_sb, _cached_ml_model_uh

    if not ML_AVAILABLE:
        return None, None, "ML libraries not available"

    # Load data if needed
    if results_df is None:
        results_df = load_results_df()

    if results_df is None or results_df.empty:
        return None, None, "No historical data"

    # Train or get cached models
    models = train_ml_model(results_df, wood_df, force_retrain=False)

    if models is None:
        return None, None, "ML model training failed"

    # Select the appropriate model for this event
    event_upper = event_code.upper()
    if event_upper not in models or models[event_upper] is None:
        return None, None, f"No {event_upper} model available"

    model = models[event_upper]

    # Load wood data for properties
    if wood_df is None:
        wood_df = load_wood_data()

    # Get wood properties
    wood_janka = 500  # Default
    wood_spec_grav = 0.5  # Default

    if wood_df is not None and not wood_df.empty:
        wood_row = wood_df[wood_df['species'] == species]
        if not wood_row.empty:
            wood_janka = wood_row.iloc[0].get('janka_hard', 500)
            wood_spec_grav = wood_row.iloc[0].get('spec_gravity', 0.5)

    # Calculate competitor average time for this event WITH TIME-DECAY WEIGHTING
    # Critical for consistency: same exponential decay as baseline and LLM predictions
    # Recent performances weighted higher than old peak performances (especially for aging competitors)
    comp_data = results_df[
        (results_df['competitor_name'] == competitor_name) &
        (results_df['event'] == event_code)
    ]

    if not comp_data.empty:
        # Apply time-decay weighting: weight = 0.5^(days_old / 730)
        # This ensures recent performances dominate the average
        if 'date' in comp_data.columns:
            weights = comp_data['date'].apply(
                lambda d: calculate_performance_weight(d, half_life_days=730)
            )
            # Calculate weighted average
            competitor_avg = (comp_data['raw_time'] * weights).sum() / weights.sum()
        else:
            # Fallback to simple mean if dates unavailable (backward compatibility)
            competitor_avg = comp_data['raw_time'].mean()

        experience = len(comp_data)
        confidence = "HIGH" if len(comp_data) >= 5 else "MEDIUM"
    else:
        # Fallback: use all competitor data regardless of event
        comp_all_data = results_df[results_df['competitor_name'] == competitor_name]
        if not comp_all_data.empty:
            # Apply time-decay weighting to cross-event data as well
            if 'date' in comp_all_data.columns:
                weights = comp_all_data['date'].apply(
                    lambda d: calculate_performance_weight(d, half_life_days=730)
                )
                competitor_avg = (comp_all_data['raw_time'] * weights).sum() / weights.sum()
            else:
                competitor_avg = comp_all_data['raw_time'].mean()

            experience = len(comp_all_data)
            confidence = "MEDIUM"
        else:
            # New competitor: use event average (time-decay weighted)
            event_data = results_df[results_df['event'] == event_code]
            if not event_data.empty:
                # Apply time-decay to event baseline as well for consistency
                if 'date' in event_data.columns:
                    weights = event_data['date'].apply(
                        lambda d: calculate_performance_weight(d, half_life_days=730)
                    )
                    competitor_avg = (event_data['raw_time'] * weights).sum() / weights.sum()
                else:
                    competitor_avg = event_data['raw_time'].mean()

                experience = 1
                confidence = "LOW"
            else:
                return None, None, "No reference data for prediction"

    # Prepare features
    event_encoded = 0 if event_code.upper() == 'SB' else 1

    features = pd.DataFrame({
        'competitor_avg_time_by_event': [competitor_avg],
        'event_encoded': [event_encoded],
        'size_mm': [diameter],
        'wood_janka_hardness': [wood_janka],
        'wood_spec_gravity': [wood_spec_grav],
        'competitor_experience': [experience]
    })

    # Make prediction using event-specific model
    try:
        base_prediction = model.predict(features)[0]

        # Apply WOOD QUALITY ADJUSTMENT
        # Quality scale: 0-10, where 5 is average
        # Lower quality (0-4) = harder/firmer wood = slower times (positive adjustment)
        # Higher quality (6-10) = softer/easier wood = faster times (negative adjustment)
        # Adjustment: ±2% per quality point from average
        quality = int(quality) if quality is not None else 5
        quality_offset = quality - 5  # Range: -5 to +5
        quality_factor = 1.0 + (-quality_offset * 0.02)  # -10% to +10%

        predicted_time = base_prediction * quality_factor

        # Sanity check: ensure prediction is reasonable
        if predicted_time < 5 or predicted_time > 300:
            return None, None, f"ML prediction out of range ({predicted_time:.1f}s)"

        # Build explanation with quality adjustment info
        explanation = f"{event_upper} ML model ({_model_training_data_size} training records)"

        if quality != 5:
            adjustment_pct = (quality_factor - 1.0) * 100
            if quality < 5:
                explanation += f", quality {quality}/10 (firmer, {adjustment_pct:+.0f}%)"
            else:
                explanation += f", quality {quality}/10 (softer, {adjustment_pct:+.0f}%)"

        return predicted_time, confidence, explanation

    except Exception as e:
        return None, None, f"ML prediction error: {str(e)}"
