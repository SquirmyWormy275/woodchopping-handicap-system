"""
Monte Carlo simulation for handicap fairness testing.

This module implements race simulation with absolute performance variation
to validate that handicap marks create equal winning probabilities for all competitors.

Key Innovation:
    Absolute variance (+/- 3 seconds for ALL competitors) rather than proportional variance.
    This is critical for fairness because real-world factors (wood grain, technique wobble,
    fatigue) affect all skill levels equally in absolute seconds, not proportionally.

Performance Variation Model:
    - Normal distribution centered on predicted time
    - Standard deviation = 3 seconds (PERFORMANCE_VARIANCE_SECONDS)
    - ~68% of actual times within +/- 3s of predicted
    - ~95% of actual times within +/- 6s of predicted
    - Minimum time floor prevents unreasonably fast times (50% of predicted)
"""

from typing import List, Dict, Any, Optional
import numpy as np

from config import rules, sim_config, display


def simulate_single_race(competitors_with_marks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Simulate a single race with absolute performance variation.

    Each competitor's actual time is sampled from a normal distribution centered
    on their predicted time with +/- 3 second standard deviation (absolute variance).
    The finish time accounts for their handicap mark (delayed start).

    Args:
        competitors_with_marks: List of competitor dictionaries containing:
            - name: Competitor name
            - mark: Handicap mark (seconds delay from front marker)
            - predicted_time: Predicted cutting time

    Returns:
        List of finish results sorted by finish time (fastest first):
            - name: Competitor name
            - mark: Handicap mark
            - actual_time: Simulated actual cutting time
            - finish_time: Total time from race start (delay + cutting time)
            - predicted_time: Original predicted time

    Algorithm:
        1. For each competitor:
           - Sample actual_time ~ Normal(predicted_time, 3 seconds)
           - Enforce minimum time = 50% of predicted (prevent unrealistic fast times)
           - Calculate finish_time = (mark - 3) + actual_time
        2. Sort by finish_time to determine race winner and positions

    Example:
        >>> competitors = [
        ...     {'name': 'Alice', 'mark': 3, 'predicted_time': 60},
        ...     {'name': 'Bob', 'mark': 25, 'predicted_time': 38}
        ... ]
        >>> results = simulate_single_race(competitors)
        >>> winner = results[0]['name']
    """
    finish_results = []

    for comp in competitors_with_marks:
        # Apply absolute variance (+/- 3 seconds for all competitors - critical for fairness)
        actual_time = np.random.normal(
            comp['predicted_time'],
            rules.PERFORMANCE_VARIANCE_SECONDS
        )

        # Prevent unreasonably fast times
        actual_time = max(actual_time, comp['predicted_time'] * 0.5)

        # Calculate finish time accounting for handicap
        start_delay = comp['mark'] - rules.MIN_MARK_SECONDS  # Front marker starts immediately
        finish_time = start_delay + actual_time

        finish_results.append({
            'name': comp['name'],
            'mark': comp['mark'],
            'actual_time': actual_time,
            'finish_time': finish_time,
            'predicted_time': comp['predicted_time']
        })

    # Sort by finish time
    finish_results.sort(key=lambda x: x['finish_time'])

    return finish_results


def run_monte_carlo_simulation(
    competitors_with_marks: List[Dict[str, Any]],
    num_simulations: Optional[int] = None
) -> Dict[str, Any]:
    """
    Run Monte Carlo simulation to assess handicap fairness.

    Simulates thousands of races to determine if all competitors have equal
    probability of winning. Tracks win rates, finish positions, and spread statistics.

    Args:
        competitors_with_marks: List of competitor dicts with marks and predicted times
        num_simulations: Number of race simulations to run (defaults to config value,
                        typically 1,000,000 for statistical significance)

    Returns:
        Dictionary containing comprehensive simulation statistics:
            - num_simulations: Number of races simulated
            - finish_spreads: List of finish time spreads for all races
            - avg_spread: Average finish time spread (seconds)
            - median_spread: Median finish time spread
            - min_spread: Tightest finish observed
            - max_spread: Widest finish observed
            - tight_finish_prob: Probability of finish within 10 seconds
            - very_tight_finish_prob: Probability of finish within 5 seconds
            - winner_counts: Dict mapping competitor name to number of wins
            - winner_percentages: Dict mapping competitor name to win percentage
            - podium_counts: Dict mapping competitor name to top-3 finishes
            - podium_percentages: Dict mapping competitor name to top-3 percentage
            - avg_finish_positions: Dict mapping competitor name to average position
            - front_marker_name: Name of slowest predicted (starts first)
            - back_marker_name: Name of fastest predicted (starts last)
            - front_marker_wins: Number of wins for front marker
            - back_marker_wins: Number of wins for back marker
            - competitors: Original competitors list

    Statistical Significance:
        With 1M simulations, margin of error is extremely small (<0.1%).
        Even 1-2% win rate differences are statistically meaningful.

    Fairness Metrics:
        - Ideal win rate = 100% / num_competitors
        - Excellent handicaps: all within +/- 1.5% of ideal
        - Good handicaps: all within +/- 5% of ideal
        - Poor handicaps: spread > 16%

    Example:
        >>> analysis = run_monte_carlo_simulation(competitors)
        >>> print(f"Avg spread: {analysis['avg_spread']:.1f}s")
        >>> for name, pct in analysis['winner_percentages'].items():
        ...     print(f"{name}: {pct:.1f}% win rate")
    """
    if num_simulations is None:
        num_simulations = sim_config.NUM_SIMULATIONS

    print("\n" + "=" * display.SEPARATOR_LENGTH)
    print(f"RUNNING MONTE CARLO SIMULATION ({num_simulations:,} races)")
    print("=" * display.SEPARATOR_LENGTH)
    print(f"Simulating races with +/-{rules.PERFORMANCE_VARIANCE_SECONDS} second absolute performance variation...")

    # Track statistics
    finish_spreads = []
    winner_counts = {comp['name']: 0 for comp in competitors_with_marks}
    podium_counts = {comp['name']: 0 for comp in competitors_with_marks}  # Top 3
    finish_position_sums = {comp['name']: 0 for comp in competitors_with_marks}
    # Track individual finish times for each competitor (for per-competitor statistics)
    competitor_finish_times = {comp['name']: [] for comp in competitors_with_marks}

    # Track front marker (slowest predicted, starts first)
    front_marker_name = max(competitors_with_marks, key=lambda x: x['predicted_time'])['name']
    back_marker_name = min(competitors_with_marks, key=lambda x: x['predicted_time'])['name']

    # Run simulations
    for i in range(num_simulations):
        if (i + 1) % 50000 == 0:
            print(f"  Completed {i + 1}/{num_simulations} simulations...")

        race_results = simulate_single_race(competitors_with_marks)

        # Calculate finish spread
        spread = race_results[-1]['finish_time'] - race_results[0]['finish_time']
        finish_spreads.append(spread)

        # Track winner
        winner_counts[race_results[0]['name']] += 1

        # Track podium (top 3)
        for j in range(min(3, len(race_results))):
            podium_counts[race_results[j]['name']] += 1

        # Track average finish positions
        for pos, result in enumerate(race_results, 1):
            finish_position_sums[result['name']] += pos

        # Track individual finish times for per-competitor statistics
        for result in race_results:
            competitor_finish_times[result['name']].append(result['finish_time'])

    # Calculate statistics
    avg_finish_positions = {name: pos_sum / num_simulations
                           for name, pos_sum in finish_position_sums.items()}

    # Calculate per-competitor time statistics
    competitor_time_stats = {}
    for name, times in competitor_finish_times.items():
        times_array = np.array(times)
        competitor_time_stats[name] = {
            'mean': np.mean(times_array),
            'std_dev': np.std(times_array),
            'min': np.min(times_array),
            'max': np.max(times_array),
            'p25': np.percentile(times_array, 25),
            'p50': np.percentile(times_array, 50),  # median
            'p75': np.percentile(times_array, 75),
            'consistency_rating': _calculate_consistency_rating(np.std(times_array))
        }

    analysis = {
        'num_simulations': num_simulations,
        'finish_spreads': finish_spreads,
        'avg_spread': np.mean(finish_spreads),
        'median_spread': np.median(finish_spreads),
        'min_spread': np.min(finish_spreads),
        'max_spread': np.max(finish_spreads),
        'tight_finish_prob': sum(1 for s in finish_spreads if s < 10) / num_simulations,
        'very_tight_finish_prob': sum(1 for s in finish_spreads if s < 5) / num_simulations,
        'winner_counts': winner_counts,
        'winner_percentages': {name: (count / num_simulations * 100)
                              for name, count in winner_counts.items()},
        'podium_counts': podium_counts,
        'podium_percentages': {name: (count / num_simulations * 100)
                              for name, count in podium_counts.items()},
        'avg_finish_positions': avg_finish_positions,
        'front_marker_name': front_marker_name,
        'back_marker_name': back_marker_name,
        'front_marker_wins': winner_counts[front_marker_name],
        'back_marker_wins': winner_counts[back_marker_name],
        'competitors': competitors_with_marks,
        'competitor_time_stats': competitor_time_stats  # Individual competitor statistics
    }

    return analysis


def _calculate_consistency_rating(std_dev: float) -> str:
    """
    Rate competitor consistency based on finish time standard deviation.

    This rating indicates how predictable a competitor's performance is across
    thousands of simulated races. Lower standard deviation means more consistent
    (predictable) performance.

    Args:
        std_dev: Standard deviation of finish times across simulations (seconds)

    Returns:
        Consistency rating string with interpretation

    Ratings:
        Very High: std_dev <= 2.5s (very predictable performance)
            - Competitor consistently finishes within a narrow time window
            - Prediction accuracy is excellent
        High: std_dev <= 3.0s (predictable, close to ±3s variance model)
            - Expected variance matches the simulation model
            - Normal consistency for the sport
        Moderate: std_dev <= 3.5s (slightly above expected variance)
            - Slightly more variable than model predicts
            - May indicate prediction uncertainty or technique variation
        Low: std_dev > 3.5s (high variability, unpredictable outcomes)
            - Wide range of possible finish times
            - Predictions less reliable, more upset potential

    Note:
        The ±3 second variance model assumes all competitors have equal
        absolute variance. Standard deviations close to 3.0s validate this model.
        Deviations significantly above 3.0s may indicate prediction errors or
        genuine performance inconsistency.

    Example:
        >>> rating = _calculate_consistency_rating(2.8)
        >>> print(rating)
        'High (expected variance)'
    """
    if std_dev <= 2.5:
        return "Very High (low variance)"
    elif std_dev <= 3.0:
        return "High (expected variance)"
    elif std_dev <= 3.5:
        return "Moderate (above expected)"
    else:
        return "Low (high variance)"
