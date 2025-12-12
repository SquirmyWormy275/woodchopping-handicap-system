"""
Prediction Aggregator for Woodchopping Handicap System

This module aggregates predictions from multiple methods (Baseline, ML, LLM)
and provides comparison, analysis, and selection of the best prediction.

Functions:
    get_all_predictions() - Get predictions from all three methods
    select_best_prediction() - Select best prediction with priority logic
    generate_prediction_analysis_llm() - LLM analysis of prediction differences
    display_dual_predictions() - Display all predictions side-by-side
"""

import statistics
import textwrap
from typing import Dict, List, Tuple, Optional
import pandas as pd

# Import local modules
from woodchopping.data import load_results_df
from woodchopping.predictions.baseline import (
    get_competitor_historical_times_flexible,
    get_event_baseline_flexible
)
from woodchopping.predictions.ml_model import predict_time_ml, _model_training_data_size
from woodchopping.predictions.ai_predictor import predict_competitor_time_with_ai
from woodchopping.predictions.llm import call_ollama


def get_all_predictions(
    competitor_name: str,
    species: str,
    diameter: float,
    quality: int,
    event_code: str,
    results_df: Optional[pd.DataFrame] = None
) -> Dict[str, Dict[str, Optional[str]]]:
    """
    Get predictions from all three methods: Baseline, ML, and LLM.

    Args:
        competitor_name: Competitor's name
        species: Wood species code
        diameter: Diameter in mm
        quality: Wood quality (0-10)
        event_code: Event type (SB or UH)
        results_df: Historical results DataFrame (optional)

    Returns:
        dict with keys 'baseline', 'ml', 'llm', each containing:
            {
                'time': float or None,
                'confidence': str or None,
                'explanation': str or None,
                'error': str or None
            }

    Example:
        >>> preds = get_all_predictions("John Smith", "WP", 300, 5, "SB")
        >>> if preds['ml']['time']:
        ...     print(f"ML: {preds['ml']['time']:.1f}s")
        >>> if preds['llm']['time']:
        ...     print(f"LLM: {preds['llm']['time']:.1f}s")
    """
    predictions = {
        'baseline': {'time': None, 'confidence': None, 'explanation': None, 'error': None},
        'ml': {'time': None, 'confidence': None, 'explanation': None, 'error': None},
        'llm': {'time': None, 'confidence': None, 'explanation': None, 'error': None}
    }

    # Load results once
    if results_df is None:
        results_df = load_results_df()

    # 1. Get baseline prediction (statistical baseline without quality adjustment)
    historical_times, data_source = get_competitor_historical_times_flexible(
        competitor_name, species, event_code, results_df
    )

    if len(historical_times) >= 3:
        weights = [1.5, 1.3, 1.1, 1.0, 0.9, 0.8]
        weighted_times = [t * w for t, w in zip(historical_times[:6], weights[:len(historical_times)])]
        baseline = sum(weighted_times) / sum(weights[:len(historical_times)])
        confidence = "HIGH"
        explanation = f"Statistical baseline ({data_source})"
    elif len(historical_times) > 0:
        baseline = statistics.mean(historical_times)
        confidence = "MEDIUM"
        explanation = f"Limited history ({data_source})"
    else:
        baseline, baseline_source = get_event_baseline_flexible(species, diameter, event_code, results_df)
        if baseline:
            confidence = "LOW"
            explanation = f"Event baseline ({baseline_source})"
        else:
            # Ultimate fallback based on diameter
            if diameter >= 350:
                baseline = 60.0
            elif diameter >= 300:
                baseline = 45.0
            elif diameter >= 250:
                baseline = 35.0
            else:
                baseline = 30.0
            confidence = "LOW"
            explanation = "Default estimate (no history)"

    predictions['baseline']['time'] = baseline
    predictions['baseline']['confidence'] = confidence
    predictions['baseline']['explanation'] = explanation

    # 2. Get ML prediction
    ml_time, ml_conf, ml_expl = predict_time_ml(
        competitor_name, species, diameter, quality, event_code, results_df
    )

    if ml_time is not None:
        predictions['ml']['time'] = ml_time
        predictions['ml']['confidence'] = ml_conf
        predictions['ml']['explanation'] = ml_expl
    else:
        predictions['ml']['error'] = ml_expl if ml_expl else "ML prediction unavailable"

    # 3. Get LLM prediction
    llm_time, llm_conf, llm_expl = predict_competitor_time_with_ai(
        competitor_name, species, diameter, quality, event_code, results_df
    )

    if llm_time is not None:
        predictions['llm']['time'] = llm_time
        predictions['llm']['confidence'] = llm_conf
        predictions['llm']['explanation'] = llm_expl
    else:
        predictions['llm']['error'] = "LLM prediction failed"

    return predictions


def select_best_prediction(
    all_predictions: Dict[str, Dict[str, Optional[str]]]
) -> Tuple[float, str, str, str]:
    """
    Select the best prediction from available methods.

    Priority: ML > LLM > Baseline

    Args:
        all_predictions: dict from get_all_predictions()

    Returns:
        tuple: (predicted_time, method_name, confidence, explanation)

    Example:
        >>> preds = get_all_predictions("John Smith", "WP", 300, 5, "SB")
        >>> time, method, conf, exp = select_best_prediction(preds)
        >>> print(f"Using {method}: {time:.1f}s ({conf})")
    """
    # Priority 1: ML (if available and valid)
    if all_predictions['ml']['time'] is not None:
        return (
            all_predictions['ml']['time'],
            'ML',
            all_predictions['ml']['confidence'],
            all_predictions['ml']['explanation']
        )

    # Priority 2: LLM (if available and valid)
    if all_predictions['llm']['time'] is not None:
        return (
            all_predictions['llm']['time'],
            'LLM',
            all_predictions['llm']['confidence'],
            all_predictions['llm']['explanation']
        )

    # Priority 3: Baseline (always available)
    return (
        all_predictions['baseline']['time'],
        'Baseline',
        all_predictions['baseline']['confidence'],
        all_predictions['baseline']['explanation']
    )


def generate_prediction_analysis_llm(
    all_competitors_predictions: List[Dict],
    wood_selection: Dict
) -> str:
    """
    Use LLM to analyze differences between ML and LLM predictions across all competitors.

    Args:
        all_competitors_predictions: list of dicts with competitor predictions
            [
                {
                    'name': str,
                    'predictions': {
                        'baseline': {...},
                        'ml': {...},
                        'llm': {...}
                    }
                },
                ...
            ]
        wood_selection: dict with wood characteristics
            {
                'species': str,
                'size_mm': float,
                'quality': int,
                'event': str
            }

    Returns:
        str: Natural language analysis of prediction differences

    Example:
        >>> analysis = generate_prediction_analysis_llm(predictions, wood_info)
        >>> print(analysis)
    """
    if not call_ollama("test", model="qwen2.5:7b"):
        return "LLM analysis unavailable (Ollama not running)"

    # Build concise summary for LLM
    summary_lines = []
    for comp_pred in all_competitors_predictions[:10]:  # Limit to 10 for prompt size
        name = comp_pred['name'][:20]  # Truncate long names
        baseline = comp_pred['predictions']['baseline']['time']
        ml = comp_pred['predictions']['ml']['time']
        llm = comp_pred['predictions']['llm']['time']

        if ml and llm:
            ml_llm_diff = ml - llm
            summary_lines.append(f"{name}: Baseline={baseline:.1f}s, ML={ml:.1f}s, LLM={llm:.1f}s (diff={ml_llm_diff:+.1f}s)")
        elif ml:
            summary_lines.append(f"{name}: Baseline={baseline:.1f}s, ML={ml:.1f}s, LLM=N/A")
        elif llm:
            summary_lines.append(f"{name}: Baseline={baseline:.1f}s, ML=N/A, LLM={llm:.1f}s")

    summary_text = "\n".join(summary_lines)

    prompt = f"""You are analyzing handicap predictions for a woodchopping competition.

WOOD CHARACTERISTICS:
- Species: {wood_selection.get('species', 'Unknown')}
- Diameter: {wood_selection.get('size_mm', 0)}mm
- Quality: {wood_selection.get('quality', 5)}/10
- Event: {wood_selection.get('event', 'Unknown')}

PREDICTION COMPARISON (Baseline vs ML vs LLM):
{summary_text}

Analyze these predictions and provide a brief 3-4 sentence analysis covering:
1. Overall agreement or divergence between methods
2. Possible reasons for any significant differences
3. Which method appears most reliable for this scenario
4. Any recommendations for the judge

Keep your response concise and practical."""

    response = call_ollama(prompt, model="qwen2.5:7b")

    if response:
        return response
    else:
        return "Unable to generate LLM analysis at this time."


def display_dual_predictions(
    handicap_results: List[Dict],
    wood_selection: Dict
) -> None:
    """
    Display handicap marks with all three prediction methods side-by-side.

    Args:
        handicap_results: List of dicts from calculate_ai_enhanced_handicaps()
            [
                {
                    'name': str,
                    'mark': int,
                    'predictions': {
                        'baseline': {...},
                        'ml': {...},
                        'llm': {...}
                    },
                    'method_used': str
                },
                ...
            ]
        wood_selection: Dict with wood characteristics

    Shows:
        - Competitor name
        - Handicap mark
        - Baseline prediction
        - ML prediction
        - LLM prediction
        - Method used for marks
        - Summary of methods
        - Optional AI analysis of differences
    """
    if not handicap_results:
        print("No handicap results to display")
        return

    # Sort by mark (ascending)
    sorted_results = sorted(handicap_results, key=lambda x: x['mark'])

    # Build header
    print("\n" + "=" * 110)
    wood_info = f"{wood_selection.get('species', 'Unknown')}, {wood_selection.get('size_mm', 0)}mm, Quality: {wood_selection.get('quality', 5)}"
    print(f"HANDICAP MARKS - {wood_info}")
    print("=" * 110)

    # Column headers
    print(f"\n{'Competitor Name':<35} {'Mark':>4}  {'Baseline':>9}  {'ML Model':>9}  {'LLM Model':>9}  {'Used':<8}")
    print("-" * 110)

    # Count methods available
    ml_available_count = 0
    llm_available_count = 0
    method_counts = {'Baseline': 0, 'ML': 0, 'LLM': 0}

    # Display each competitor
    for comp in sorted_results:
        name = comp['name'][:35]
        mark = comp['mark']

        # Get predictions
        baseline_time = comp['predictions']['baseline']['time']
        ml_time = comp['predictions']['ml']['time']
        llm_time = comp['predictions']['llm']['time']

        # Format predictions (show "N/A" if None)
        baseline_str = f"{baseline_time:.1f}s" if baseline_time else "N/A"
        ml_str = f"{ml_time:.1f}s" if ml_time else "N/A"
        llm_str = f"{llm_time:.1f}s" if llm_time else "N/A"

        # Track which method was used
        method_used = comp.get('method_used', 'Unknown')
        method_counts[method_used] = method_counts.get(method_used, 0) + 1

        # Count availability
        if ml_time is not None:
            ml_available_count += 1
        if llm_time is not None:
            llm_available_count += 1

        print(f"{name:<35} {mark:4d}  {baseline_str:>9}  {ml_str:>9}  {llm_str:>9}  {method_used:<8}")

    print("=" * 110)

    # Display prediction methods summary
    print("\nPrediction Methods Summary:")
    print(f"  • Baseline: Statistical calculation (always available)")

    if ml_available_count > 0:
        ml_status = "HIGH" if _model_training_data_size >= 80 else "MEDIUM" if _model_training_data_size >= 50 else "LOW"
        print(f"  • ML Model: XGBoost trained on {_model_training_data_size} records [CONFIDENCE: {ml_status}] - Available for {ml_available_count}/{len(sorted_results)} competitors")
    else:
        print(f"  • ML Model: Not available (insufficient training data)")

    if llm_available_count > 0:
        print(f"  • LLM Model: Ollama qwen2.5:7b AI prediction - Available for {llm_available_count}/{len(sorted_results)} competitors")
    else:
        print(f"  • LLM Model: Not available (Ollama not running or prediction failed)")

    # Show which method was primarily used
    primary_method = max(method_counts, key=method_counts.get)
    print(f"\nMarks calculated using: {primary_method} predictions")
    print(f"(Priority: ML > LLM > Baseline - most accurate method available)")

    # Offer AI analysis
    print("\n" + "=" * 110)
    analyze = input("\nPress Enter to see AI analysis of prediction differences (or 'n' to skip): ").strip().lower()

    if analyze != 'n':
        print("\n" + "=" * 110)
        print("AI ANALYSIS OF PREDICTIONS")
        print("=" * 110)
        print("\nAnalyzing prediction differences...")

        analysis = generate_prediction_analysis_llm(handicap_results, wood_selection)

        # Word wrap the analysis for better readability
        wrapped_lines = textwrap.wrap(analysis, width=106)
        for line in wrapped_lines:
            print(line)

        print("\n" + "=" * 110)
