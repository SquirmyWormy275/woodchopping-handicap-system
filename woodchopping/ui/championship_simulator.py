"""
Championship race simulator - fun predictive tool for race outcome analysis.

This module provides a standalone championship event simulator that uses the existing
prediction and Monte Carlo simulation systems to predict race outcomes without
calculating handicap marks. All competitors start together (Mark 3) in championship
format - fastest raw time wins.

Key Features:
    - Reuses existing prediction engine (baseline, ML, LLM)
    - Runs 2 million Monte Carlo simulations for statistical confidence
    - Displays individual competitor statistics (time variations, consistency)
    - AI-powered race analysis focusing on matchups and competitive dynamics
    - View-only (no tournament state changes or Excel writes)
"""

from typing import Dict, List
import sys

from woodchopping.ui.wood_ui import wood_menu, select_event_code
from woodchopping.ui.competitor_ui import select_all_event_competitors
from woodchopping.predictions.prediction_aggregator import (
    get_all_predictions,
    select_best_prediction
)
from woodchopping.simulation.monte_carlo import run_monte_carlo_simulation
from woodchopping.simulation.visualization import (
    visualize_simulation_results,
    generate_simulation_summary
)
from woodchopping.simulation.fairness import get_championship_race_analysis
from woodchopping.data import load_results_df


def run_championship_simulator(comp_df):
    """
    Interactive championship race simulator.

    Provides a complete workflow for simulating championship-format races where
    all competitors start together (Mark 3) and fastest time wins. Uses AI-enhanced
    predictions and Monte Carlo simulation to predict race outcomes, win probabilities,
    and competitive dynamics.

    Workflow:
        1. Wood configuration (species, diameter, quality)
        2. Event type selection (SB/UH)
        3. Competitor selection (no enforced limits for fun scenarios)
        4. Generate predictions for all competitors (all receive Mark 3)
        5. Run Monte Carlo simulation (2 million races)
        6. Display championship results table with win rates
        7. Display visualization (bar chart)
        8. Display individual competitor statistics
        9. AI race outcome analysis

    Args:
        comp_df: Competitor master roster DataFrame

    Returns:
        None (view-only feature, no state changes)

    Example:
        >>> from woodchopping.data import load_competitor_df
        >>> comp_df = load_competitor_df()
        >>> run_championship_simulator(comp_df)
        # User interacts with prompts to configure race and view predictions
    """
    print("\n" + "=" * 70)
    print("  CHAMPIONSHIP RACE SIMULATOR")
    print("  Predict race outcomes without handicaps - all start together!")
    print("=" * 70)

    # Initialize wood selection
    wood_selection = {'species': None, 'size_mm': None, 'quality': None, 'event': None}

    # Step 1: Configure wood characteristics
    print("\n[STEP 1/4] Configure Wood Characteristics")
    wood_selection = wood_menu(wood_selection)

    # Step 2: Select event type (SB/UH)
    print("\n[STEP 2/4] Select Event Type")
    wood_selection = select_event_code(wood_selection)

    if not wood_selection.get('event'):
        print("\n[ERROR] Event type is required. Returning to menu.")
        input("\nPress Enter to continue...")
        return

    # Step 3: Select competitors
    print("\n[STEP 3/4] Select Competitors")
    print("Note: No competitor limit for championship simulator - have fun!")
    selected_df = select_all_event_competitors(
        comp_df,
        max_competitors=None,  # No limit
        require_confirmation=True
    )

    if selected_df.empty:
        print("\n[ERROR] No competitors selected. Returning to menu.")
        input("\nPress Enter to continue...")
        return

    # Step 4: Generate predictions
    print("\n[STEP 4/4] Generating Predictions...")
    predictions = _generate_championship_predictions(
        selected_df,
        wood_selection
    )

    if not predictions:
        print("\n[ERROR] Prediction generation failed. Returning to menu.")
        input("\nPress Enter to continue...")
        return

    # Display championship results table
    _display_championship_results(predictions, wood_selection)

    # Run Monte Carlo simulation
    print(f"\n[SIMULATION] Running 2,000,000 race simulations...")
    analysis = run_monte_carlo_simulation(predictions, num_simulations=2_000_000)

    # Display simulation summary
    summary = generate_simulation_summary(analysis)
    print("\n" + summary)

    # Display visualization
    visualize_simulation_results(analysis)

    # Display individual competitor statistics
    _display_individual_competitor_stats(analysis, predictions)

    # AI race analysis
    show_ai = input("\nView AI race analysis? (y/n): ").strip().lower()
    if show_ai == 'y':
        _display_race_analysis(analysis, predictions)

    print("\n" + "=" * 70)
    print("  CHAMPIONSHIP SIMULATOR COMPLETE")
    print("=" * 70)
    input("\nPress Enter to return to main menu...")


def _generate_championship_predictions(selected_df, wood_selection: Dict) -> List[Dict]:
    """
    Generate predictions for all competitors with Mark 3 (championship format).

    Uses the standard prediction pipeline (baseline, ML, LLM) but assigns
    Mark 3 to all competitors since championship format has no handicaps.

    Args:
        selected_df: DataFrame of selected competitors
        wood_selection: Wood characteristics dict

    Returns:
        List of prediction dicts sorted by predicted time (fastest first)
    """
    results_df = load_results_df()
    predictions = []

    total = len(selected_df)
    print(f"\nGenerating predictions for {total} competitors...")

    for idx, (_, row) in enumerate(selected_df.iterrows(), 1):
        comp_name = row['competitor_name']

        # Progress indicator
        sys.stdout.write(f"\r  Progress: {idx}/{total} ({comp_name[:30]}...)")
        sys.stdout.flush()

        # Get all 3 prediction methods
        all_preds = get_all_predictions(
            comp_name,
            wood_selection['species'],
            wood_selection['size_mm'],
            wood_selection['quality'],
            wood_selection['event'],
            results_df,
            tournament_results=None  # No tournament weighting for mock events
        )

        # Select best prediction
        pred_time, method, confidence, explanation = select_best_prediction(all_preds)

        predictions.append({
            'name': comp_name,
            'predicted_time': pred_time,
            'mark': 3,  # Championship: everyone starts together
            'method_used': method,
            'confidence': confidence,
            'predictions': all_preds
        })

    print("\n")  # New line after progress

    # Sort by predicted time (fastest first for championship)
    predictions.sort(key=lambda x: x['predicted_time'])

    return predictions


def _display_championship_results(predictions: List[Dict], wood_selection: Dict):
    """
    Display championship race predictions in table format.

    Shows predicted times and prediction methods before Monte Carlo simulation.
    After simulation, win rates and podium percentages will be shown.

    Args:
        predictions: List of prediction dicts
        wood_selection: Wood characteristics for header display
    """
    from woodchopping.data import get_species_name_from_code

    species_name = get_species_name_from_code(wood_selection['species'])

    print("\n" + "=" * 70)
    print("  CHAMPIONSHIP RACE PREDICTIONS")
    print("=" * 70)
    print(f"\nWood: {species_name}, {wood_selection['size_mm']}mm, Quality {wood_selection['quality']}")
    print(f"Event: {wood_selection['event']}")
    print(f"Competitors: {len(predictions)}")
    print("\n" + "-" * 70)
    print(f"{'#':<4} {'Competitor':<30} {'Pred. Time':<12} {'Method':<8}")
    print("-" * 70)

    for idx, pred in enumerate(predictions, 1):
        print(f"{idx:<4} {pred['name']:<30} {pred['predicted_time']:>6.1f}s      {pred['method_used']:<8}")

    print("-" * 70)
    print("\nNote: All competitors start together (Mark 3) - fastest time wins!")


def _display_individual_competitor_stats(analysis: Dict, predictions: List[Dict]):
    """
    Display detailed per-competitor statistics from Monte Carlo simulation.

    Shows time variations, consistency ratings, and performance ranges for each
    competitor based on 2 million simulated races.

    Args:
        analysis: Monte Carlo simulation analysis dict
        predictions: List of prediction dicts
    """
    print("\n" + "=" * 70)
    print("  INDIVIDUAL PERFORMANCE ANALYSIS")
    print(f"  (Based on {analysis['num_simulations']:,} simulations)")
    print("=" * 70)

    competitor_stats = analysis['competitor_time_stats']
    winner_pcts = analysis['winner_percentages']
    avg_positions = analysis['avg_finish_positions']

    for pred in predictions:
        name = pred['name']
        stats = competitor_stats[name]

        print(f"\nCompetitor: {name}")
        print("  " + "-" * 66)
        print(f"  Predicted Time:     {pred['predicted_time']:>6.1f}s")
        print(f"  Average Sim Time:   {stats['mean']:>6.2f}s  (+/-{stats['std_dev']:.2f}s std dev)")
        print(f"  Time Range:         {stats['min']:>6.1f}s - {stats['max']:.1f}s")
        print(f"  25th Percentile:    {stats['p25']:>6.1f}s  |  75th Percentile: {stats['p75']:.1f}s")
        print(f"  Win Rate:           {winner_pcts[name]:>6.2f}%")
        print(f"  Avg Finish Pos:     {avg_positions[name]:>6.2f}")
        print(f"  Consistency Rating: {stats['consistency_rating']}")

    print("\n" + "=" * 70)


def _display_race_analysis(analysis: Dict, predictions: List[Dict]):
    """
    Display AI-generated race outcome analysis.

    Uses LLM to provide engaging commentary on race dynamics, likely winners,
    key matchups, and competitive narratives.

    Args:
        analysis: Monte Carlo simulation analysis dict
        predictions: List of prediction dicts
    """
    import textwrap

    print("\n" + "=" * 70)
    print("  AI RACE ANALYSIS")
    print("=" * 70)

    try:
        # Get championship race analysis from AI
        race_analysis = get_championship_race_analysis(analysis, predictions)

        # Display with word wrapping
        print()
        for line in race_analysis.split('\n'):
            if len(line) <= 100:
                print(line)
            else:
                wrapped = textwrap.fill(line, width=100)
                print(wrapped)

    except Exception as e:
        print(f"\n[ERROR] AI analysis failed: {e}")
        print("Continuing without AI analysis...")
        print("\nStatistical Summary:")
        print(f"  Most likely winner: {max(analysis['winner_percentages'].items(), key=lambda x: x[1])[0]}")
        print(f"  Win probability: {max(analysis['winner_percentages'].values()):.1f}%")

    print("\n" + "=" * 70)
