"""
AI-powered fairness assessment for handicap marks.

This module uses LLM analysis to evaluate Monte Carlo simulation results and provide
expert assessment of handicap quality, pattern diagnosis, and adjustment recommendations.
"""

from typing import Dict, Any, List, Optional
import textwrap
import numpy as np

from config import sim_config, llm_config
from woodchopping.predictions.llm import call_ollama
from woodchopping.simulation.monte_carlo import run_monte_carlo_simulation
from woodchopping.simulation.visualization import (
    generate_simulation_summary,
    visualize_simulation_results
)


def get_ai_assessment_of_handicaps(analysis: Dict[str, Any]) -> str:
    """
    Use LLM to provide expert assessment of handicap fairness.

    Analyzes Monte Carlo simulation results using an AI model to:
    - Rate overall fairness (Excellent/Very Good/Good/Fair/Poor/Unacceptable)
    - Diagnose systematic bias patterns (front marker advantage, etc.)
    - Assess prediction accuracy and identify problematic competitors
    - Provide specific, actionable recommendations for improvement

    Args:
        analysis: Simulation results dictionary from run_monte_carlo_simulation()

    Returns:
        Formatted assessment text containing:
        - FAIRNESS RATING: Overall quality rating
        - STATISTICAL ANALYSIS: Interpretation of win rate spreads and finish times
        - PATTERN DIAGNOSIS: Identification of systematic biases
        - PREDICTION ACCURACY: Assessment of time prediction quality
        - RECOMMENDATIONS: Specific actions to improve fairness

    Fairness Rating Scale:
        EXCELLENT (spread <= 3%): All win rates within +/- 1.5% of ideal
        VERY GOOD (spread <= 6%): All win rates within +/- 3% of ideal
        GOOD (spread <= 10%): Acceptable for competition
        FAIR (spread <= 16%): Noticeable imbalance, adjustments recommended
        POOR (spread > 16%): Significant bias requiring recalibration
        UNACCEPTABLE: Any competitor >2x or <0.5x ideal win rate

    Common Diagnostic Patterns:
        1. Front Marker Advantage: Soft wood bias in predictions
        2. Back Marker Advantage: Hard wood bias in predictions
        3. Middle Compression: Predictions too conservative at extremes
        4. Experience Bias: Better predictions for experienced competitors
        5. Species Miscalibration: Systematic bias across all competitors

    Example:
        >>> analysis = run_monte_carlo_simulation(competitors)
        >>> assessment = get_ai_assessment_of_handicaps(analysis)
        >>> print(assessment)
        FAIRNESS RATING: VERY GOOD

        STATISTICAL ANALYSIS: With 4 competitors, ideal win rate is 25.0% each...
        ...

    Note:
        If Ollama is unavailable, returns a simplified fallback assessment
        based on statistical thresholds.
    """
    # Calculate fairness metrics
    max_win_rate = max(analysis['winner_percentages'].values())
    min_win_rate = min(analysis['winner_percentages'].values())
    win_rate_spread = max_win_rate - min_win_rate
    ideal_win_rate = 100.0 / len(analysis['competitors'])

    # Calculate per-competitor deviations
    win_rate_deviations = {}
    for name, pct in analysis['winner_percentages'].items():
        deviation = pct - ideal_win_rate
        win_rate_deviations[name] = deviation

    # Identify extremes
    most_favored = max(win_rate_deviations, key=win_rate_deviations.get)
    most_disadvantaged = min(win_rate_deviations, key=win_rate_deviations.get)

    # Format data for prompt
    winner_data = "\n".join([f"  - {name}: {pct:.2f}% win rate (deviation: {win_rate_deviations[name]:+.2f}%)"
                            for name, pct in sorted(analysis['winner_percentages'].items(),
                                                   key=lambda x: x[1], reverse=True)])

    competitor_details = "\n".join([f"  - {comp['name']}: {comp['predicted_time']:.1f}s predicted +/- Mark {comp['mark']}"
                                    for comp in sorted(analysis['competitors'],
                                                      key=lambda x: x['predicted_time'], reverse=True)])

    win_rate_std_dev = np.std(list(analysis['winner_percentages'].values()))
    coefficient_of_variation = (win_rate_std_dev / ideal_win_rate) * 100 if ideal_win_rate > 0 else 0

    prompt = f"""You are a master woodchopping handicapper and statistician analyzing the fairness of predicted handicap marks through Monte Carlo simulation.

HANDICAPPING PRINCIPLES

PRIMARY GOAL: Create handicaps where ALL competitors have EQUAL probability of winning.
- In a fair handicap system, skill level should NOT predict victory
- A novice with Mark 3 should win as often as an expert with Mark 25
- The slowest competitor should have the same chance as the fastest

HANDICAPPING MECHANISM:
1. Predict each competitor's raw cutting time
2. Slowest predicted time receives Mark 3 (starts first)
3. Faster predicted times receive higher marks (delayed starts)
4. If predictions are perfect, everyone finishes simultaneously
5. Natural variation (+/-3s) creates competitive spread

QUALITY FACTORS IN PREDICTIONS:
- Wood species (hardness variations)
- Block diameter (volume to cut)
- Wood quality rating (0-10 scale, affects cutting speed)
- Historical competitor performance

SIMULATION METHODOLOGY

WHAT WE TESTED:
- Simulated {analysis['num_simulations']:,} races with {len(analysis['competitors'])} competitors
- Applied +/-3 second ABSOLUTE performance variation (realistic race conditions)
- Variation represents: technique consistency, wood grain, fatigue, environmental conditions

WHY ABSOLUTE VARIANCE (+/-3s for everyone):
- Real factors affect all skill levels equally in absolute seconds
- Wood grain knot costs 2s for novice AND expert (not proportional to skill)
- Technique wobble affects everyone by similar absolute time
- This is a CRITICAL breakthrough in fair handicapping

STATISTICAL SIGNIFICANCE:
- With {analysis['num_simulations']:,} simulations, margin of error is extremely small
- Patterns in results are REAL, not random noise
- Even 1-2% win rate differences are statistically meaningful

SIMULATION RESULTS

COMPETITOR PREDICTIONS AND MARKS:
{competitor_details}

IDEAL WIN RATE: {ideal_win_rate:.2f}% per competitor
(Perfect handicapping means all competitors win exactly {ideal_win_rate:.2f}% of races)

ACTUAL WIN RATES:
{winner_data}

STATISTICAL MEASURES:
- Win Rate Spread: {win_rate_spread:.2f}% (maximum minus minimum)
- Standard Deviation: {win_rate_std_dev:.2f}%
- Coefficient of Variation: {coefficient_of_variation:.1f}%

FINISH TIME ANALYSIS:
- Average finish spread: {analysis['avg_spread']:.1f} seconds
- Median finish spread: {analysis['median_spread']:.1f} seconds
- Tight finishes (<10s): {analysis['tight_finish_prob']*100:.1f}% of races
- Very tight finishes (<5s): {analysis['very_tight_finish_prob']*100:.1f}% of races

FRONT AND BACK MARKER PERFORMANCE:
- Front Marker (slowest): {analysis['front_marker_name']} - {analysis['front_marker_wins']/analysis['num_simulations']*100:.1f}% wins
- Back Marker (fastest): {analysis['back_marker_name']} - {analysis['back_marker_wins']/analysis['num_simulations']*100:.1f}% wins

PATTERN IDENTIFICATION:
- Most Favored: {most_favored} ({analysis['winner_percentages'][most_favored]:.2f}%, +{win_rate_deviations[most_favored]:.2f}%)
- Most Disadvantaged: {most_disadvantaged} ({analysis['winner_percentages'][most_disadvantaged]:.2f}%, {win_rate_deviations[most_disadvantaged]:.2f}%)

FAIRNESS CRITERIA

RATING SCALE (based on win rate spread):

EXCELLENT (Spread d 3%):
- All win rates within +/-1.5% of ideal ({ideal_win_rate-1.5:.1f}% to {ideal_win_rate+1.5:.1f}%)
- Handicaps are nearly perfect
- Predictions are highly accurate
- No adjustments needed

VERY GOOD (Spread d 6%):
- All win rates within +/-3% of ideal ({ideal_win_rate-3:.1f}% to {ideal_win_rate+3:.1f}%)
- Handicaps are working well
- Minor prediction inaccuracies
- Only minor adjustments if desired

GOOD (Spread d 10%):
- All win rates within +/-5% of ideal ({ideal_win_rate-5:.1f}% to {ideal_win_rate+5:.1f}%)
- Acceptable fairness for competition
- Some prediction bias exists
- Consider adjustments for championship events

FAIR (Spread d 16%):
- Win rates within +/-8% of ideal
- Noticeable imbalance
- Predictions need refinement
- Adjustments recommended

POOR (Spread > 16%):
- Significant imbalance detected
- Predictions are systematically biased
- Handicaps require major adjustment
- Not suitable for fair competition

UNACCEPTABLE (Any competitor >2x or <0.5x ideal):
- Extreme bias detected
- One competitor has double (or half) expected win rate
- Fundamental prediction error
- Complete recalibration required

DIAGNOSTIC PATTERNS

COMMON ISSUES TO IDENTIFY:

1. FRONT MARKER ADVANTAGE (soft wood bias):
   Pattern: Front marker wins >ideal, back marker wins <ideal
   Cause: Predictions underestimate benefit of soft wood to slower competitors
   Fix: Increase quality adjustment for front markers on soft wood

2. BACK MARKER ADVANTAGE (hard wood bias):
   Pattern: Back marker wins >ideal, front marker wins <ideal
   Cause: Predictions underestimate difficulty of hard wood for slower competitors
   Fix: Increase time penalties for front markers on hard wood

3. MIDDLE COMPRESSION:
   Pattern: Extreme competitors (fastest/slowest) win less than middle competitors
   Cause: Predictions too conservative at extremes
   Fix: Increase handicap spread (widen gaps between marks)

4. EXPERIENCE BIAS:
   Pattern: Competitors with more historical data win more often
   Cause: Better predictions for experienced competitors
   Fix: Adjust confidence weighting or baseline calculations

5. SPECIES MISCALIBRATION:
   Pattern: Systematic bias across all competitors
   Cause: Species hardness factor incorrect
   Fix: Adjust species baseline percentage

YOUR ANALYSIS TASK

Provide a comprehensive assessment in the following structure:

1. FAIRNESS RATING: State one of: Excellent / Very Good / Good / Fair / Poor / Unacceptable

2. STATISTICAL ANALYSIS (2-3 sentences):
   - Interpret the win rate spread of {win_rate_spread:.2f}%
   - Comment on finish time spreads (average {analysis['avg_spread']:.1f}s)
   - Assess if variation is appropriate for exciting competition

3. PATTERN DIAGNOSIS (2-3 sentences):
   - Identify which diagnostic pattern (if any) is present
   - Explain WHY this pattern occurred based on competitor times
   - Reference specific competitors showing the bias

4. PREDICTION ACCURACY (1-2 sentences):
   - Are the predictions systematically biased or just slightly off?
   - Is the issue with one competitor or system-wide?

5. RECOMMENDATIONS (2-3 specific actions):
   If EXCELLENT or VERY GOOD: Affirm handicaps are ready for use
   If GOOD: Suggest optional refinements
   If FAIR, POOR, or UNACCEPTABLE: Provide specific adjustment recommendations

   Format recommendations as bullet points:
   " First specific action (include numbers when possible)
   " Second specific action
   " Final recommendation

RESPONSE REQUIREMENTS:
- Keep total response to 8-12 sentences maximum
- Be specific and actionable
- Use technical terms confidently
- Cite actual numbers from the data above
- Base analysis on ACTUAL DATA, not generic observations
- Reference specific competitors, percentages, and patterns you observe

Your Expert Assessment:"""

    response = call_ollama(prompt, num_predict=llm_config.TOKENS_FAIRNESS_ASSESSMENT)

    if response:
        return response
    else:
        # Enhanced fallback assessment
        if win_rate_spread < 3:
            rating = "EXCELLENT"
            assessment = "Handicaps are nearly perfect. Predictions are highly accurate with minimal bias."
        elif win_rate_spread < 6:
            rating = "VERY GOOD"
            assessment = "Handicaps are working very well. Minor prediction variations are within acceptable range."
        elif win_rate_spread < 10:
            rating = "GOOD"
            assessment = "Handicaps are acceptable for competition. Some prediction refinement would improve fairness."
        elif win_rate_spread < 16:
            rating = "FAIR"
            assessment = "Noticeable imbalance detected. Predictions show systematic bias requiring adjustment."
        else:
            rating = "POOR"
            assessment = "Significant imbalance requiring major prediction recalibration."

        front_wins = analysis['front_marker_wins']/analysis['num_simulations']*100
        back_wins = analysis['back_marker_wins']/analysis['num_simulations']*100

        if front_wins > ideal_win_rate + 3:
            pattern = "Front marker advantage detected (soft wood bias likely)."
        elif back_wins > ideal_win_rate + 3:
            pattern = "Back marker advantage detected (hard wood bias likely)."
        else:
            pattern = "No clear front/back marker bias pattern."

        return f"""FAIRNESS RATING: {rating}

STATISTICAL ANALYSIS: With {len(analysis['competitors'])} competitors, ideal win rate is {ideal_win_rate:.1f}% each. Actual spread is {win_rate_spread:.2f}% (from {min_win_rate:.1f}% to {max_win_rate:.1f}%). {assessment} Average finish spread of {analysis['avg_spread']:.1f}s creates exciting competition.

PATTERN DIAGNOSIS: {pattern} {most_favored} is most favored at {analysis['winner_percentages'][most_favored]:.1f}% wins (+{win_rate_deviations[most_favored]:.1f}% above ideal), while {most_disadvantaged} is disadvantaged at {analysis['winner_percentages'][most_disadvantaged]:.1f}% wins ({win_rate_deviations[most_disadvantaged]:.1f}% below ideal).

RECOMMENDATIONS:
- {"Handicaps are ready for competition use - no adjustments needed." if win_rate_spread < 6 else f"Review predictions for {most_favored} and {most_disadvantaged} - time estimates may need adjustment."}
- {"Continue collecting historical data to improve future predictions." if win_rate_spread < 10 else "Consider adjusting quality/species factors in prediction model."}
- {"Monitor real competition results to validate simulation predictions." if win_rate_spread < 16 else "Recalibrate baseline calculations before using these handicaps in competition."}"""


def format_ai_assessment(assessment_text: str, width: int = 100) -> None:
    """
    Format and print AI assessment with intelligent text wrapping.

    Preserves structure while wrapping long lines:
    - Section headers (lines ending with colon or ALL CAPS) stay on single line
    - Bullet points get hanging indent (initial=2 spaces, subsequent=4 spaces)
    - Regular paragraphs wrap at word boundaries
    - Blank lines between sections are maintained

    Args:
        assessment_text: Raw AI assessment text
        width: Maximum line width for wrapping (default 100)

    Example:
        >>> format_ai_assessment(ai_response, width=100)
    """
    paragraphs = assessment_text.split('\n\n')

    for paragraph in paragraphs:
        if not paragraph.strip():
            continue

        lines = paragraph.split('\n')
        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue

            # Check if it's a section header (contains colon, typically uppercase)
            if ':' in stripped and (
                stripped.isupper() or
                stripped.split(':')[0].isupper()
            ):
                # Don't wrap section headers - keep on single line
                print(stripped)
            # Check if it's a bullet point
            elif stripped.startswith(('-', '*', '•', '▪', '◦')):
                # Wrap bullet points with hanging indent
                wrapped = textwrap.wrap(
                    stripped,
                    width=width,
                    initial_indent='  ',
                    subsequent_indent='    '
                )
                for wrapped_line in wrapped:
                    print(wrapped_line)
            else:
                # Regular paragraph text - wrap with no special indent
                wrapped = textwrap.wrap(stripped, width=width)
                for wrapped_line in wrapped:
                    print(wrapped_line)

        print()  # Blank line between paragraphs/sections


def simulate_and_assess_handicaps(
    competitors_with_marks: List[Dict[str, Any]],
    num_simulations: Optional[int] = None
) -> None:
    """
    Run complete simulation and AI assessment workflow.

    This is the main high-level function that:
    1. Runs Monte Carlo simulation
    2. Displays statistical summary
    3. Shows visual win rate chart
    4. Provides AI fairness assessment

    Args:
        competitors_with_marks: List of competitor dicts with marks and predicted times
        num_simulations: Number of race simulations (defaults to config value)

    Displays:
        - Monte Carlo simulation progress
        - Comprehensive statistical summary
        - Visual bar chart of win rates
        - AI-generated fairness assessment with recommendations

    Example:
        >>> competitors = [
        ...     {'name': 'Alice', 'mark': 3, 'predicted_time': 60},
        ...     {'name': 'Bob', 'mark': 25, 'predicted_time': 38}
        ... ]
        >>> simulate_and_assess_handicaps(competitors)

        ======================================================================
        RUNNING MONTE CARLO SIMULATION (1,000,000 races)
        ======================================================================
        ...

    Note:
        This function prints directly to console and does not return a value.
        Use run_monte_carlo_simulation() directly if you need the raw data.
    """
    if not competitors_with_marks or len(competitors_with_marks) < 2:
        print("Need at least 2 competitors to run simulation.")
        return

    # Use config default if not specified
    if num_simulations is None:
        num_simulations = sim_config.NUM_SIMULATIONS

    # Run simulation
    analysis = run_monte_carlo_simulation(competitors_with_marks, num_simulations)

    # Display results
    summary = generate_simulation_summary(analysis)
    print(summary)

    # Visualize
    visualize_simulation_results(analysis)

    # Get AI assessment
    print("\n" + "="*70)
    print("AI HANDICAPPING ASSESSMENT")
    print("="*70)
    print("\nAnalyzing fairness of handicaps...")

    ai_assessment = get_ai_assessment_of_handicaps(analysis)

    # Format and display with intelligent text wrapping
    print("")  # Add spacing
    format_ai_assessment(ai_assessment, width=100)

    print("="*70)
