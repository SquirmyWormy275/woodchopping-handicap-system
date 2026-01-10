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
    - Per-competitor std-dev when available (clamped), otherwise 3 seconds
    - Shared heat-level variance applied to all competitors
    - Minimum time floor prevents unreasonably fast times (50% of predicted)
"""

from typing import List, Dict, Any, Optional
import numpy as np

from config import rules, sim_config, display


def _get_competitor_variance_seconds(comp: Dict[str, Any]) -> float:
    """Return per-competitor variance (std-dev) with reasonable bounds."""
    variance = comp.get('performance_std_dev')
    if variance is None:
        variance = rules.PERFORMANCE_VARIANCE_SECONDS

    try:
        variance = float(variance)
        if np.isnan(variance):
            variance = rules.PERFORMANCE_VARIANCE_SECONDS
    except (TypeError, ValueError):
        variance = rules.PERFORMANCE_VARIANCE_SECONDS

    variance = max(sim_config.MIN_COMPETITOR_STD_SECONDS, min(sim_config.MAX_COMPETITOR_STD_SECONDS, variance))
    return variance


def simulate_single_race(competitors_with_marks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Simulate a single race with performance variation.

    Each competitor's actual time is sampled from a normal distribution centered
    on their predicted time with per-competitor std-dev when available (fallback +/-3s).
    A shared heat-level variance is applied to all competitors. The finish time
    accounts for their handicap mark (delayed start).

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
           - Sample actual_time ~ Normal(predicted_time + heat_delta, competitor_std)
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

    # Shared heat conditions (wind, grain pattern, moisture) affect everyone
    heat_delta = np.random.normal(0.0, sim_config.HEAT_VARIANCE_SECONDS)

    for comp in competitors_with_marks:
        # Per-competitor variance with shared heat effect
        variance_seconds = _get_competitor_variance_seconds(comp)
        actual_time = np.random.normal(
            comp['predicted_time'] + heat_delta,
            variance_seconds
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
    num_simulations: Optional[int] = None,
    track_finish_orders: bool = False,
    track_podium_margins: bool = False,
    show_live_leaders: bool = False,
    progress_interval: int = 50000
) -> Dict[str, Any]:
    """
    Run Monte Carlo simulation to assess handicap fairness.

    Simulates thousands of races to determine if all competitors have equal
    probability of winning. Tracks win rates, finish positions, and spread statistics.

    Args:
        competitors_with_marks: List of competitor dicts with marks and predicted times
        num_simulations: Number of race simulations to run (defaults to config value,
                        typically 1,000,000 for statistical significance)
        track_finish_orders: Track most common finish order (full order for <=8 competitors)
        track_podium_margins: Track average podium margins and photo-finish rate
        show_live_leaders: Print interim leader updates during long runs
        progress_interval: Simulation count interval for progress updates

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
    print(f"Simulating races with per-competitor variance (fallback +/-{rules.PERFORMANCE_VARIANCE_SECONDS}s) and +/-{sim_config.HEAT_VARIANCE_SECONDS:.1f}s heat variance...")

    # Track statistics
    finish_spreads = []
    winner_counts = {comp['name']: 0 for comp in competitors_with_marks}
    podium_counts = {comp['name']: 0 for comp in competitors_with_marks}  # Top 3
    finish_position_sums = {comp['name']: 0 for comp in competitors_with_marks}
    # Track individual finish times for each competitor (for per-competitor statistics)
    competitor_finish_times = {comp['name']: [] for comp in competitors_with_marks}
    order_counts = {} if track_finish_orders else None
    order_scope = "podium" if track_finish_orders and len(competitors_with_marks) > 8 else "full"
    margin_12_sum = 0.0
    margin_23_sum = 0.0
    margin_12_count = 0
    margin_23_count = 0
    photo_finish_count = 0
    photo_finish_threshold = 0.25

    # Track front marker (slowest predicted, starts first)
    front_marker_name = max(competitors_with_marks, key=lambda x: x['predicted_time'])['name']
    back_marker_name = min(competitors_with_marks, key=lambda x: x['predicted_time'])['name']

    # Run simulations
    for i in range(num_simulations):
        if progress_interval and (i + 1) % progress_interval == 0:
            print(f"  Completed {i + 1}/{num_simulations} simulations...")
            if show_live_leaders:
                leaders = sorted(winner_counts.items(), key=lambda x: x[1], reverse=True)[:3]
                leader_str = ", ".join([f"{name} {count / (i + 1) * 100:.1f}%" for name, count in leaders])
                print(f"    Leaders: {leader_str}")

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

        # Track podium margins
        if track_podium_margins and len(race_results) >= 2:
            margin_12 = race_results[1]['finish_time'] - race_results[0]['finish_time']
            margin_12_sum += margin_12
            margin_12_count += 1
            if margin_12 <= photo_finish_threshold:
                photo_finish_count += 1
            if len(race_results) >= 3:
                margin_23 = race_results[2]['finish_time'] - race_results[1]['finish_time']
                margin_23_sum += margin_23
                margin_23_count += 1

        # Track most common finish order (full order for <=8; podium order for >8)
        if order_counts is not None:
            if order_scope == "full":
                order_key = tuple(result['name'] for result in race_results)
            else:
                order_key = tuple(result['name'] for result in race_results[:3])
            order_counts[order_key] = order_counts.get(order_key, 0) + 1

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

    competitor_variances = {
        comp['name']: _get_competitor_variance_seconds(comp)
        for comp in competitors_with_marks
    }

    most_common_order = None
    most_common_order_pct = None
    if order_counts:
        most_common_order = max(order_counts.items(), key=lambda x: x[1])[0]
        most_common_order_pct = (order_counts[most_common_order] / num_simulations) * 100.0

    avg_margin_12 = (margin_12_sum / margin_12_count) if margin_12_count else None
    avg_margin_23 = (margin_23_sum / margin_23_count) if margin_23_count else None
    photo_finish_pct = (photo_finish_count / margin_12_count * 100.0) if margin_12_count else None

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
        'competitor_time_stats': competitor_time_stats,  # Individual competitor statistics
        'heat_variance_seconds': sim_config.HEAT_VARIANCE_SECONDS,
        'competitor_variances': competitor_variances,
        'most_common_order': most_common_order,
        'most_common_order_pct': most_common_order_pct,
        'most_common_order_scope': order_scope if order_counts is not None else None,
        'avg_podium_margin_12': avg_margin_12,
        'avg_podium_margin_23': avg_margin_23,
        'photo_finish_pct': photo_finish_pct,
        'photo_finish_threshold': photo_finish_threshold
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
        High: std_dev <= 3.0s (predictable, close to default variance model)
            - Expected variance matches the simulation model
            - Normal consistency for the sport
        Moderate: std_dev <= 3.5s (slightly above expected variance)
            - Slightly more variable than model predicts
            - May indicate prediction uncertainty or technique variation
        Low: std_dev > 3.5s (high variability, unpredictable outcomes)
            - Wide range of possible finish times
            - Predictions less reliable, more upset potential

    Note:
        The default ±3 second variance model assumes all competitors have equal
        absolute variance. When per-competitor variance is provided, expected
        std-dev may shift accordingly.

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
