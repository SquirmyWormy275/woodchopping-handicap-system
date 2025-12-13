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
    # Step 1: Get historical data WITH TIME-DECAY WEIGHTING
    # Critical for aging competitors: recent performances weighted much higher than peak from years ago
    historical_data, data_source = get_competitor_historical_times_flexible(
        competitor_name, species, event_code, results_df, return_weights=True
    )

    # Step 2: Calculate time-weighted baseline
    if len(historical_data) >= 3:
        # Each item is (time, date, weight) where weight = 0.5^(days_old/730)
        # Recent results (this season) have weight ~1.0
        # Results from 10 years ago have weight ~0.03 (essentially zero)
        weighted_sum = sum(time * weight for time, date, weight in historical_data)
        weight_sum = sum(weight for time, date, weight in historical_data)
        baseline = weighted_sum / weight_sum if weight_sum > 0 else statistics.mean([t for t,d,w in historical_data])

        confidence = "HIGH"
        avg_weight = weight_sum / len(historical_data)
        explanation_source = f"time-weighted history {data_source} ({len(historical_data)} results, avg weight {avg_weight:.2f})"

    elif len(historical_data) > 0:
        # Limited history - still apply time-decay
        weighted_sum = sum(time * weight for time, date, weight in historical_data)
        weight_sum = sum(weight for time, date, weight in historical_data)
        baseline = weighted_sum / weight_sum if weight_sum > 0 else statistics.mean([t for t,d,w in historical_data])

        confidence = "MEDIUM"
        avg_weight = weight_sum / len(historical_data)
        explanation_source = f"limited time-weighted history {data_source} ({len(historical_data)} results, avg weight {avg_weight:.2f})"

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

QUALITY RATING SYSTEM - CRITICAL UNDERSTANDING

Quality measures wood SOFTNESS on a 0-10 scale:
- HIGHER number = SOFTER wood = FASTER cutting = LOWER time
- LOWER number = HARDER wood = SLOWER cutting = HIGHER time

10 = Extremely soft/rotten
   - Wood breaks apart easily, minimal resistance
   - FASTEST possible cutting time
   - MULTIPLY baseline by 0.85-0.90 (reduce by 10-15%)
   - Example: 40s baseline → 34-36s predicted

9 = Very soft (ideal competition wood)
   - Excellent cutting conditions, clean grain
   - MULTIPLY baseline by 0.90-0.93 (reduce by 7-10%)
   - Example: 40s baseline → 36-37s predicted

8 = Soft
   - Good cutting conditions, easy to work with
   - MULTIPLY baseline by 0.93-0.95 (reduce by 5-7%)
   - Example: 40s baseline → 37-38s predicted

7 = Moderately soft
   - Better than average, noticeable improvement
   - MULTIPLY baseline by 0.95-0.97 (reduce by 3-5%)
   - Example: 40s baseline → 38-39s predicted

6 = Slightly soft
   - Marginally better than average
   - MULTIPLY baseline by 0.97-0.99 (reduce by 1-3%)
   - Example: 40s baseline → 39-40s predicted

5 = AVERAGE HARDNESS (BASELINE REFERENCE POINT)
   - This is what the baseline time assumes
   - MULTIPLY baseline by 1.00 (NO ADJUSTMENT)
   - Example: 40s baseline → 40s predicted

4 = Slightly hard
   - Marginally tougher than average
   - MULTIPLY baseline by 1.01-1.03 (increase by 1-3%)
   - Example: 40s baseline → 40-41s predicted

3 = Moderately hard
   - Noticeably tougher, more resistance
   - MULTIPLY baseline by 1.03-1.05 (increase by 3-5%)
   - Example: 40s baseline → 41-42s predicted

2 = Hard (difficult cutting)
   - Significant resistance, green wood, tough grain
   - MULTIPLY baseline by 1.05-1.08 (increase by 5-8%)
   - Example: 40s baseline → 42-43s predicted

1 = Very hard
   - Major difficulty, knots, irregular grain
   - MULTIPLY baseline by 1.08-1.12 (increase by 8-12%)
   - Example: 40s baseline → 43-45s predicted

0 = Extremely hard/barely suitable
   - Maximum difficulty
   - SLOWEST possible cutting time
   - MULTIPLY baseline by 1.12-1.15 (increase by 12-15%)
   - Example: 40s baseline → 45-46s predicted

CURRENT SITUATION ANALYSIS

Baseline time: {baseline:.1f}s (assumes quality 5 wood)
Your wood quality: {quality}/10

Quality deviation: {quality - 5:+d} points from baseline reference

CRITICAL CALCULATION DIRECTION:
{"⚠️ Quality " + str(quality) + " > 5 means SOFTER wood → FASTER cutting → LOWER time than baseline" if quality > 5 else "⚠️ Quality " + str(quality) + " < 5 means HARDER wood → SLOWER cutting → HIGHER time than baseline" if quality < 5 else "✓ Quality 5 = baseline assumption → NO ADJUSTMENT needed"}

{"Expected multiplier: 0.95-0.97 (reduce baseline by 3-5%)" if quality == 6 else "Expected multiplier: 0.93-0.95 (reduce baseline by 5-7%)" if quality == 7 else "Expected multiplier: 0.90-0.93 (reduce baseline by 7-10%)" if quality == 8 else "Expected multiplier: 0.85-0.90 (reduce baseline by 10-15%)" if quality >= 9 else "Expected multiplier: 1.01-1.03 (increase baseline by 1-3%)" if quality == 4 else "Expected multiplier: 1.03-1.05 (increase baseline by 3-5%)" if quality == 3 else "Expected multiplier: 1.05-1.08 (increase baseline by 5-8%)" if quality == 2 else "Expected multiplier: 1.08-1.12 (increase baseline by 8-12%)" if quality == 1 else "Expected multiplier: 1.12-1.15 (increase baseline by 12-15%)" if quality == 0 else "Expected multiplier: 1.00 (no change)"}

{"Target range: " + str(round(baseline * 0.95, 1)) + "s - " + str(round(baseline * 0.97, 1)) + "s" if quality == 6 else "Target range: " + str(round(baseline * 0.93, 1)) + "s - " + str(round(baseline * 0.95, 1)) + "s" if quality == 7 else "Target range: " + str(round(baseline * 0.90, 1)) + "s - " + str(round(baseline * 0.93, 1)) + "s" if quality == 8 else "Target range: " + str(round(baseline * 0.85, 1)) + "s - " + str(round(baseline * 0.90, 1)) + "s" if quality >= 9 else "Target range: " + str(round(baseline * 1.01, 1)) + "s - " + str(round(baseline * 1.03, 1)) + "s" if quality == 4 else "Target range: " + str(round(baseline * 1.03, 1)) + "s - " + str(round(baseline * 1.05, 1)) + "s" if quality == 3 else "Target range: " + str(round(baseline * 1.05, 1)) + "s - " + str(round(baseline * 1.08, 1)) + "s" if quality == 2 else "Target range: " + str(round(baseline * 1.08, 1)) + "s - " + str(round(baseline * 1.12, 1)) + "s" if quality == 1 else "Target range: " + str(round(baseline * 1.12, 1)) + "s - " + str(round(baseline * 1.15, 1)) + "s" if quality == 0 else "Target: " + str(round(baseline, 1)) + "s (baseline)"}

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
