# Underhand Prediction Model Issues - Root Cause Analysis

## Executive Summary

The UH prediction model is producing inaccurate handicaps because **most competitors only have historical data from 325mm wood, but the system doesn't properly scale these times when predicting for 275mm wood**. This results in predictions that don't match real-world performance.

## Test Case: 275mm Aspen, Quality 6

### Historical Data Availability

| Competitor | UH Results | Wood Sizes | Issue |
|------------|------------|------------|-------|
| **Erin LaVoie** | 6 results | **275mm only** | ✓ Exact match |
| Cody Labahn | 5 results | **325mm only** | ⚠ No 275mm data |
| David Moses Jr. | 9 results | **325mm only** | ⚠ No 275mm data |
| Eric Hoberg | 4 results | **325mm only** | ⚠ No 275mm data |
| Cole Schlenker | **0 results** | None | ⚠ No UH data at all |

**KEY FINDING:** Only 1 of 5 competitors has historical data in the target wood size (275mm)!

### ML Model Predictions (Currently Selected)

Based on the selection logic (Priority: ML > LLM > Baseline), the system uses:

| Competitor | ML Prediction | Confidence | Issue |
|------------|--------------|------------|-------|
| Cody Labahn | 22.7s | HIGH | Extrapolated from 325mm (23-33s range) |
| **Erin LaVoie** | 23.2s | HIGH | Based on 275mm data (23-31s range) |
| Cole Schlenker | 24.6s | MEDIUM | **No competitor data!** Model guess |
| Eric Hoberg | 28.7s | MEDIUM | Extrapolated from 325mm (27-56s range) |
| David Moses Jr. | 33.8s | HIGH | Based on 325mm (19-38s), recent 29s |

### Resulting Handicap Marks

| Competitor | Prediction | Mark | Start Delay | Problem |
|------------|-----------|------|-------------|---------|
| David Moses Jr. | 33.8s | **Mark 3** | 0s (front) | Slowest predicted, starts first |
| Eric Hoberg | 28.7s | Mark 8 | 5s later | |
| Cole Schlenker | 24.6s | Mark 12 | 9s later | **No data** but getting mid-pack mark |
| Erin LaVoie | 23.2s | Mark 14 | 11s later | |
| Cody Labahn | 22.7s | **Mark 14** | 11s later | Fastest, starts last |

### Why This Contradicts Real-World Observations

**User Report:** "Moses and Cole Schlenker have decidedly beat Erin Lavoie in real life"

**Model Says:**
- Erin (23.2s) should beat Moses (33.8s) by **10+ seconds**
- Moses should be the **slowest** competitor

**Discrepancy:** Moses is predicted 10.6 seconds slower than Erin, but user says he beats her decisively.

## Root Cause Analysis

### 1. **Wood Size Scaling Not Applied**

The ML model learns from ALL historical data (both 275mm and 325mm), but doesn't explicitly understand that:

**275mm wood = Faster times than 325mm wood**

The 50mm difference (275mm vs 325mm) is approximately **18% smaller diameter**.

Empirical evidence from Erin Cramsey's data:
- 275mm: 40-85s range
- 250mm: 53s

This shows smaller wood = faster times, but the relationship is complex and non-linear.

**Problem:** When Moses's 325mm times (19-38s) are used to predict 275mm performance, the model doesn't apply sufficient scaling. His most recent 29s time in 325mm should translate to something faster in 275mm (perhaps 24-26s), making him competitive with Erin.

### 2. **Cole Schlenker - No Data Fallback**

Cole has **ZERO UH historical data**, yet the ML model predicts 24.6s with MEDIUM confidence.

**How is this possible?**
- The ML model learns patterns across all competitors
- It looks at features like:
  - Average competitor performance
  - Wood characteristics (Janka hardness, specific gravity)
  - Event type
  - Diameter

For Cole, it's essentially using:
- Event baseline for UH
- Wood properties for Aspen
- 275mm diameter patterns
- **NO competitor-specific performance data**

This is better than nothing, but still unreliable. The 24.6s prediction could be wildly wrong (could be 20s or 40s in reality).

### 3. **Moses Time-Decay Weighting**

Moses's historical times span 2015-2025:
- **Best:** 19s (2018) - 7 years ago
- **Recent:** 29s (2025) - current
- **Time-decay weight:** 0.19 (very low - old data heavily discounted)

The model correctly identifies Moses has **slowed significantly** over time. A 10-second decline (19s → 29s) is substantial.

**BUT:** User says Moses is still beating Erin. This suggests either:
1. Moses has more recent better performances not in the database
2. The 29s time in 325mm translates to faster times in 275mm
3. The wood size adjustment factor is critical

### 4. **Erin's Data Appears Optimal But May Not Be**

Erin has 6 results in exactly 275mm:
- 2025: **23s**
- 2024: **24s, 26s**
- 2023: 31s
- 2022: 31s

Recent times (23-26s) are excellent. Time-decay avg weight: 0.54

The model predicts 23.2s, which seems reasonable.

**But why does Moses beat her?**

Possible explanations:
1. **Competition context:** Moses might perform better under pressure
2. **Unreported results:** More recent races not in database
3. **Wood variation:** Quality/species variations between competitions
4. **Technique:** Moses might have better technique for certain wood characteristics

## The Fundamental Problem: DATA SPARSITY

### UH Results by Diameter

Querying the database:
- **325mm UH:** 30+ results (Moses, Cody, Eric, Angus, etc.)
- **275mm UH:** 11 results (only Erin LaVoie and Erin Cramsey)
- **Other sizes:** Scattered

**Critical Gap:** 73% of UH data is from 325mm, but competitions also run 275mm events.

### Why This Breaks Predictions

1. **Insufficient training data** for 275mm-specific patterns
2. **No cross-size calibration** - model can't learn "how much faster is 275mm vs 325mm?"
3. **Overconfidence** - ML reports "HIGH" confidence even when extrapolating across wood sizes

## Solutions Required

### Immediate Fixes

#### 1. **Implement Diameter-Based Scaling Factor**

When historical data is from a different wood size, apply adjustment:

```python
# Simplified linear scaling (needs empirical validation)
def scale_time_for_diameter(time_seconds, from_diameter, to_diameter):
    """
    Scale chopping time based on diameter difference.

    Assumption: Time scales roughly with diameter^2 (proportional to wood volume)
    This needs to be validated against real competition data.
    """
    if from_diameter == to_diameter:
        return time_seconds

    # Diameter ratio
    ratio = to_diameter / from_diameter

    # Time scales with square of diameter (more wood to chop)
    # This is a simplified model - real relationship may be different
    scaled_time = time_seconds * (ratio ** 1.5)  # Exponent needs tuning

    return scaled_time

# Example:
# Moses: 29s in 325mm → predict for 275mm
# scaled = 29 * (275/325)^1.5 = 29 * 0.845^1.5 ≈ 29 * 0.777 ≈ 22.5s
```

**Validation needed:** Analyze competitors who have times in BOTH 275mm and 325mm to determine actual scaling relationship.

#### 2. **Confidence Adjustment for Wood Size Mismatch**

```python
def adjust_confidence_for_size_mismatch(confidence, hist_diameter, pred_diameter):
    """
    Reduce confidence when extrapolating across wood sizes.
    """
    if hist_diameter == pred_diameter:
        return confidence

    diameter_diff = abs(hist_diameter - pred_diameter)

    if diameter_diff > 50:  # e.g., 275mm vs 350mm
        if confidence == "HIGH":
            return "MEDIUM"
        elif confidence == "MEDIUM":
            return "LOW"
    elif diameter_diff > 25:  # e.g., 275mm vs 325mm
        if confidence == "HIGH":
            return "MEDIUM"

    return confidence
```

#### 3. **Warning Flags for Judges**

Add explicit warnings to handicap output:

```
⚠ Cody Labahn: Prediction based on 325mm data, no 275mm history
⚠ Cole Schlenker: NO UH DATA - prediction based on event baseline only
⚠ David Moses Jr.: Time-decay weight 0.19 (old data) + no 275mm history
```

#### 4. **Manual Override UI**

Allow judges to adjust predictions based on recent observations:

```
Competitor: David Moses Jr.
  ML Prediction: 33.8s (HIGH confidence)
  Based on: 325mm data (9 results), recent time 29s

  Judge Override: [  ] seconds  [Reason: ________________]
```

### Medium-Term Improvements

#### 1. **Learn Diameter Scaling from Historical Data**

Find all competitors with results in multiple wood sizes:

```sql
SELECT competitor, diameter1, avg_time1, diameter2, avg_time2
FROM (
    SELECT competitor_name,
           FIRST_VALUE(size_mm) OVER (PARTITION BY competitor_name ORDER BY size_mm) as diameter1,
           FIRST_VALUE(size_mm) OVER (PARTITION BY competitor_name ORDER BY size_mm DESC) as diameter2
    FROM results
    WHERE event = 'UH'
)
```

Calculate empirical scaling factor:
```python
scaling_factor = avg_time2 / avg_time1
diameter_ratio = diameter2 / diameter1
# Fit curve to find best exponent
```

#### 2. **Species-Specific Scaling**

Different wood species may have different scaling relationships:
- Softwoods (Pine) might scale linearly
- Hardwoods (Oak) might scale exponentially

Build separate models for each species.

#### 3. **ML Model Features Enhancement**

Add explicit features to ML model:
- `diameter_match`: Boolean (1 if historical size == prediction size, else 0)
- `diameter_ratio`: float (hist_diameter / pred_diameter)
- `has_exact_size_history`: Boolean

This helps the model learn when to trust extrapolations.

#### 4. **Cross-Validation by Wood Size**

When training ML model, stratify by wood size:
- Train on 325mm → test on 325mm: Measure accuracy
- Train on 325mm → test on 275mm: Measure scaling error
- Adjust model to minimize cross-size prediction error

### Long-Term Solutions

#### 1. **Collect More 275mm UH Data**

**Action:** Run competitions with 275mm UH events and record results

**Target:** Get at least 3 results per competitor in 275mm UH

This is the most reliable solution but requires time.

#### 2. **Physics-Based Model**

Develop a theoretical model based on:
- Wood volume (π × r² × length)
- Janka hardness
- Axe efficiency
- Chopper power output

This could provide better extrapolation when data is sparse.

#### 3. **Competitor Profiling**

Track per-competitor scaling factors:
- Some competitors might adapt better to larger/smaller wood
- Build "preferred size" profiles
- Account for age/strength changes over time

## Recommended Immediate Action

**For the current competition (275mm Aspen, Quality 6):**

1. **Apply manual diameter scaling** to Moses, Cody, and Eric's predictions:
   ```
   Moses: 33.8s → ~25-27s (scaled from 325mm)
   Cody: 22.7s → ~20-22s (scaled from 325mm)
   Eric: 28.7s → ~23-25s (scaled from 325mm)
   ```

2. **Flag Cole's prediction as unreliable** - suggest placing him in an early heat to gather actual data

3. **Warn judges** that 4 of 5 competitors have no 275mm history

4. **Collect results** from this competition to improve future 275mm predictions

## Conclusion

The UH prediction model isn't fundamentally broken - it's making the best predictions it can with available data. The core issue is **data sparsity** (lack of 275mm UH results) combined with **no diameter scaling logic**.

Fixing this requires:
1. Short-term: Manual adjustments and judge warnings
2. Medium-term: Diameter scaling algorithms
3. Long-term: More comprehensive data collection

The user's observation that Moses beats Erin is likely accurate, and the model is wrong because it's extrapolating Moses's 325mm times to 275mm without proper scaling.
