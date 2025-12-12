"""
Handicap mark calculation using AI-enhanced predictions.

This module implements the core handicap calculation logic for woodchopping competitions.
It combines multiple prediction methods (baseline, ML, and LLM) to determine optimal
handicap marks that give all competitors equal probability of winning.
"""

from typing import Dict, List, Optional, Callable, Any
import pandas as pd

from config import rules

# Import prediction functions from refactored predictions module
from woodchopping.predictions.prediction_aggregator import (
    get_all_predictions,
    select_best_prediction
)


def calculate_ai_enhanced_handicaps(
    heat_assignment_df: pd.DataFrame,
    species: str,
    diameter: float,
    quality: int,
    event_code: str,
    results_df: pd.DataFrame,
    progress_callback: Optional[Callable[[int, int, str], None]] = None
) -> Optional[List[Dict[str, Any]]]:
    """
    Calculate handicaps using dual prediction system (Baseline + ML + LLM).

    This function processes each competitor in the heat, generates predictions using
    multiple methods, selects the best prediction, and calculates handicap marks
    according to AAA competition rules.

    Args:
        heat_assignment_df: DataFrame containing competitors in heat with 'competitor_name' column
        species: Wood species (e.g., 'Pine', 'Oak')
        diameter: Block diameter in millimeters
        quality: Wood quality rating (0-10 scale, 0=soft/rotten, 10=very hard)
        event_code: Event type ('SB' for Standing Block, 'UH' for Underhand)
        results_df: Historical results DataFrame with competitor performance data
        progress_callback: Optional callback function(current, total, competitor_name)
                          for progress tracking during calculation

    Returns:
        List of dictionaries containing handicap data for each competitor:
        - name: Competitor name
        - predicted_time: Best prediction selected for handicap calculation
        - method_used: Which prediction method was selected (baseline/ML/LLM)
        - confidence: Confidence level of prediction (HIGH/MEDIUM/LOW)
        - explanation: Text explanation of why this prediction was selected
        - predictions: Dict containing all three predictions for display/analysis
        - mark: Calculated handicap mark (seconds delay from front marker)

        Returns None if no valid predictions could be generated.

    Handicap Mark Calculation:
        1. Slowest predicted time receives Mark 3 (front marker, starts first)
        2. Each second faster increases mark by 1 (delayed start)
        3. Marks are rounded UP to whole seconds (ceiling logic)
        4. Maximum mark is 183 (180 second time limit + 3 second minimum)
        5. All competitors should theoretically finish simultaneously if predictions perfect

    Example:
        >>> results = calculate_ai_enhanced_handicaps(
        ...     heat_df,
        ...     species='Pine',
        ...     diameter=350,
        ...     quality=5,
        ...     event_code='SB',
        ...     results_df=historical_data
        ... )
        >>> for competitor in results:
        ...     print(f"{competitor['name']}: Mark {competitor['mark']}")
    """
    results = []

    # Ensure quality is an integer
    if quality is None:
        quality = 5
    quality = int(quality)

    total_competitors = len(heat_assignment_df)

    # Run predictions with progress tracking
    for idx, (_, row) in enumerate(heat_assignment_df.iterrows(), 1):
        comp_name = row.get("competitor_name")

        if progress_callback:
            progress_callback(idx, total_competitors, comp_name)

        # Get ALL predictions (baseline, ML, LLM)
        all_preds = get_all_predictions(
            comp_name, species, diameter, quality, event_code, results_df
        )

        # Select best prediction for handicap marks
        predicted_time, method_used, confidence, explanation = select_best_prediction(all_preds)

        if predicted_time is None:
            continue

        results.append({
            'name': comp_name,
            'predicted_time': predicted_time,  # Best prediction for marks
            'method_used': method_used,        # Which method was used
            'confidence': confidence,
            'explanation': explanation,
            'predictions': all_preds           # Store all predictions for display
        })

    # Calculate marks
    if not results:
        return None

    # Sort by predicted time (slowest first)
    results.sort(key=lambda x: x['predicted_time'], reverse=True)

    # Slowest competitor gets mark 3
    slowest_time = results[0]['predicted_time']

    for result in results:
        gap = slowest_time - result['predicted_time']
        mark = 3 + int(gap + 0.999)  # Round up using ceiling logic

        # Apply 180-second maximum rule
        if mark > 183:
            mark = 183

        result['mark'] = mark

    return results
