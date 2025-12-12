"""
AI-Enhanced Time Prediction for Woodchopping Handicap System

This module provides LLM-based time predictions that combine historical data
with AI reasoning to adjust for wood quality and other contextual factors.

Functions:
    predict_competitor_time_with_ai() - Predict time using historical data + LLM quality adjustment
"""

import re
import statistics
from typing import Tuple, Optional
import pandas as pd

# Import config
from config import llm_config

# Import local modules
from woodchopping.data import load_wood_data
from woodchopping.predictions.baseline import (
    get_competitor_historical_times_flexible,
    get_event_baseline_flexible
)
from woodchopping.predictions.llm import call_ollama


def predict_competitor_time_with_ai(
    competitor_name: str,
    species: str,
    diameter: float,
    quality: int,
    event_code: str,
    results_df: pd.DataFrame
) -> Tuple[float, str, str]:
    """
    Predict competitor's time using historical data + LLM reasoning for quality adjustment.

    This function implements a multi-stage prediction process:
    1. Get historical times (with cascading fallback)
    2. Calculate baseline time using weighted average
    3. Use LLM to adjust baseline for wood quality
    4. Fallback to statistical adjustment if LLM unavailable

    Args:
        competitor_name: Competitor's name
        species: Wood species code
        diameter: Diameter in mm
        quality: Wood quality rating (0-10, where 5=average, >5=softer, <5=harder)
        event_code: Event type (SB or UH)
        results_df: Historical results DataFrame

    Returns:
        Tuple of (predicted_time, confidence, explanation)
        - predicted_time: Float predicted time in seconds
        - confidence: "HIGH", "MEDIUM", "LOW", or "VERY LOW"
        - explanation: String describing prediction method and data source

    Example:
        >>> time, conf, exp = predict_competitor_time_with_ai(
        ...     "John Smith", "White Pine", 300, 7, "SB", results_df
        ... )
        >>> print(f"Predicted {time:.1f}s (confidence: {conf})")
        >>> print(f"Based on: {exp}")
    """
    # Step 1: Get historical data
    historical_times, data_source = get_competitor_historical_times_flexible(
        competitor_name, species, event_code, results_df
    )

    # Step 2: Calculate baseline
    if len(historical_times) >= 3:
        weights = [1.5, 1.3, 1.1, 1.0, 0.9, 0.8]
        weighted_times = [t * w for t, w in zip(historical_times[:6], weights[:len(historical_times)])]
        baseline = sum(weighted_times) / sum(weights[:len(historical_times)])
        confidence = "HIGH"
        explanation_source = f"competitor history {data_source}"

    elif len(historical_times) > 0:
        baseline = statistics.mean(historical_times)
        confidence = "MEDIUM"
        explanation_source = f"limited competitor history {data_source}"

    else:
        baseline, baseline_source = get_event_baseline_flexible(species, diameter, event_code, results_df)

        if baseline:
            confidence = "LOW"
            explanation_source = f"{baseline_source} (no competitor history)"
        else:
            if diameter >= 350:
                baseline = 60.0
            elif diameter >= 300:
                baseline = 50.0
            elif diameter >= 275:
                baseline = 45.0
            else:
                baseline = 40.0

            confidence = "VERY LOW"
            explanation_source = "estimated from size (no historical data)"

    # LOAD ACTUAL WOOD DATA FROM EXCEL
    wood_df = load_wood_data()

    # Format wood species data for AI
    wood_data_text = ""
    if wood_df is not None and not wood_df.empty:
        wood_data_text = "\nAVAILABLE WOOD SPECIES DATABASE:\n"
        for _, row in wood_df.iterrows():
            species_name = row.get('species', 'Unknown')
            wood_data_text += f"  - {species_name}"
            if 'hardness_category' in row:
                wood_data_text += f": Category={row.get('hardness_category', 'N/A')}"
            if 'base_adjustment_pct' in row:
                wood_data_text += f", Base Adjustment={row.get('base_adjustment_pct', 0):+.1f}%"
            if 'description' in row:
                wood_data_text += f", Description: {row.get('description', '')}"
            wood_data_text += "\n"

    # Step 3: AI prediction prompt
    prompt = f"""You are a master woodchopping handicapper making precision time predictions for competition.

HANDICAPPING OBJECTIVE

Your prediction must account for wood characteristics and competitor ability to create fair handicaps.
When handicaps are applied, all competitors should finish simultaneously if your predictions are accurate.
This requires deep understanding of how wood properties affect cutting times.

COMPETITOR PROFILE

Name: {competitor_name}
Baseline Time: {baseline:.1f} seconds
Data Source: {explanation_source}
Confidence Level: {confidence}

BASELINE INTERPRETATION:
- This baseline assumes QUALITY 5 wood (average hardness)
- Your task is to adjust this baseline for the ACTUAL quality rating
- Historical data already accounts for competitor's skill level and typical conditions

WOOD SPECIFICATIONS

Species: {species}
Diameter: {diameter:.0f}mm
Quality Rating: {quality}/10
Event Type: {event_code}

WOOD CHARACTERISTICS DATABASE
{wood_data_text}

QUALITY RATING SYSTEM

Quality measures wood condition on a 0-10 scale:

10 = Extremely soft/rotten
   - Wood breaks apart easily
   - Minimal resistance to axe
   - FASTEST possible cutting time
   - Reduces baseline time by approximately 10-15%

9 = Very soft (ideal competition wood)
   - Excellent cutting conditions
   - Clean grain, well-seasoned
   - Reduces baseline time by approximately 7-10%

8 = Soft
   - Good cutting conditions
   - Easy to work with
   - Reduces baseline time by approximately 5-7%

7 = Moderately soft
   - Better than average
   - Noticeable improvement over baseline
   - Reduces baseline time by approximately 3-5%

6 = Slightly soft
   - Marginally better than average
   - Minor improvement
   - Reduces baseline time by approximately 1-3%

5 = AVERAGE HARDNESS (BASELINE REFERENCE POINT)
   - This is what the baseline time assumes
   - NO ADJUSTMENT needed at quality 5
   - Standard competition wood

4 = Slightly hard
   - Marginally tougher than average
   - Minor slowdown
   - Increases baseline time by approximately 1-3%

3 = Moderately hard
   - Noticeably tougher
   - More resistance
   - Increases baseline time by approximately 3-5%

2 = Hard (difficult cutting)
   - Significant resistance
   - Green wood, tough grain
   - Increases baseline time by approximately 5-8%

1 = Very hard
   - Major difficulty
   - Knots, irregular grain
   - Increases baseline time by approximately 8-12%

0 = Extremely hard/barely suitable
   - Maximum difficulty
   - SLOWEST possible cutting time
   - Increases baseline time by approximately 12-15%

CURRENT SITUATION ANALYSIS

Your wood is quality {quality}, which is {abs(quality - 5)} point(s) {"ABOVE" if quality > 5 else "BELOW" if quality < 5 else "AT"} the baseline reference.

{"This wood is SOFTER than baseline - expect FASTER cutting time." if quality > 5 else "This wood is HARDER than baseline - expect SLOWER cutting time." if quality < 5 else "This wood is AVERAGE hardness - baseline time should be accurate."}

CALCULATION METHODOLOGY

Step 1: Start with baseline time: {baseline:.1f}s

Step 2: Apply species base adjustment (if available in database)
- Check species database above for {species}
- Apply the base adjustment percentage if listed

Step 3: Apply quality adjustment
- Calculate deviation from quality 5: {quality} - 5 = {quality - 5}
- Apply adjustment based on quality scale above
- Use the percentage ranges provided (1.5-2.5% per point as guideline)

Step 4: Consider wood physics
- Softer wood (quality >5): Cuts faster, less resistance
- Harder wood (quality <5): Cuts slower, more resistance
- Effect is roughly linear in the middle range (3-7)
- Effect accelerates at extremes (0-2 and 8-10)

Step 5: Validate against typical ranges for {diameter:.0f}mm diameter
- 275mm diameter: 35-50s typical range
- 300mm diameter: 40-55s typical range
- 325mm diameter: 45-60s typical range
- 350mm diameter: 50-70s typical range
- 375mm+ diameter: 60-90s typical range

CRITICAL FACTORS FOR FAIR HANDICAPPING

Front/Back Marker Dynamics:
- Softer wood (quality >5) disproportionately benefits slower competitors (front markers)
  * They gain more time than expected from easier cutting
  * Risk: Front marker finishes before back marker even starts

- Harder wood (quality <5) disproportionately penalizes slower competitors
  * They lose more time than expected from difficult cutting
  * Risk: Back marker wins by excessive margin

Your adjustment must account for this to maintain fair handicapping.

WOOD DENSITY AND SIZE INTERACTION

The {diameter:.0f}mm diameter creates a cutting area of approximately {3.14159 * (diameter/2)**2 / 10000:.2f} square cm.
- Larger diameter = exponentially more wood volume to remove
- Quality affects this proportionally: softer wood on large diameter saves significant time
- This diameter/quality interaction is already partially in baseline, but verify your adjustment makes sense

RESPONSE REQUIREMENT

Calculate the most accurate predicted time for {competitor_name} cutting {species} at quality {quality}.

CRITICAL: Respond with ONLY the predicted time as a decimal number.
- Example: 47.3
- NO units (like "seconds" or "s")
- NO explanations
- NO additional text
- JUST THE NUMBER

Predicted time:"""

    response = call_ollama(prompt)

    if response is None:
        # Fallback: statistical quality adjustment
        quality_adjustment = (5 - quality) * 0.02
        predicted_time = baseline * (1 + quality_adjustment)
        explanation = f"Predicted {predicted_time:.1f}s ({explanation_source}, quality adjusted)"
        return predicted_time, confidence, explanation

    try:
        # Parse LLM response - extract first number
        numbers = re.findall(r'\d+\.?\d*', response)
        if numbers:
            predicted_time = float(numbers[0])

            # Sanity check: prediction should be within 50% of baseline
            if baseline * 0.5 <= predicted_time <= baseline * 1.5:
                explanation = f"Predicted {predicted_time:.1f}s ({explanation_source}, AI quality adjusted)"
                return predicted_time, confidence, explanation
    except:
        pass

    # If LLM parsing fails, return baseline
    explanation = f"Predicted {baseline:.1f}s ({explanation_source})"
    return baseline, confidence, explanation
