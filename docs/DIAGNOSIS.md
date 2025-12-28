# UH Prediction Model Diagnosis - 275mm Aspen

## The Problem

The current handicap system is producing **inverted results** for underhand events. Competitors with the best historical times are receiving the worst handicap marks (starting last), while competitors with no data are given the best marks (starting first).

## Current (INCORRECT) Predictions

Testing configuration: 275mm Aspen, Quality 6

| Competitor | Predicted Time | Mark | Position | Has UH Data? |
|------------|---------------|------|----------|--------------|
| Erin LaVoie | 25.5s | 21 | **LAST** | YES (6 results in 275mm) |
| Cody Labahn | 27.4s | 19 | 4th | YES (5 results in 325mm) |
| David Moses Jr. | 29.8s | 17 | 3rd | YES (9 results in 325mm) |
| Eric Hoberg | 31.2s | 16 | 2nd | YES (4 results in 325mm) |
| Cole Schlenker | 43.8s | **3** | **FIRST** | **NO - NO UH DATA** |

## Why This Is Wrong

### 1. Erin LaVoie Should Be FASTEST (Front Marker), Not Slowest

**Historical Data:**
- 6 UH results **all in 275mm** (exact size being predicted)
- Recent times: **23s** (2025), **24s** (2024), **26s** (2024)
- Older times: 31s (2022-2023)

**Current Prediction:** 25.5s → Mark 21 (starts last, 18 seconds behind front marker)

**Problem:** The system correctly calculates a baseline of 25.5s, but assigns the HIGHEST mark number to the fastest predicted time. **This is backwards!**

In handicapping:
- **Lower mark = Faster competitor = Starts earlier (front marker)**
- **Higher mark = Slower competitor = Starts later (back marker)**

### 2. Cody Labahn - Wood Size Mismatch

**Historical Data:**
- 5 UH results **all in 325mm** (50mm LARGER than test configuration)
- Times range: 23-33s in 325mm

**Current Prediction:** 27.4s

**Problem:** All of Cody's data is from 325mm wood. The system is applying these times to 275mm predictions without proper adjustment. Smaller wood = faster times, so Cody should be predicted faster than 27.4s.

### 3. David Moses Jr. - Aging Competitor Data Not Properly Weighted

**Historical Data:**
- 9 UH results in 325mm
- Most recent: **29s** (2025)
- Historical best: 19s (2018, 7 years ago)
- Time decay weight: **0.19** (very old data)

**Current Prediction:** 29.8s

**Problem:** David's most recent time is 29s (10 seconds slower than his 2018 peak). The time-decay weighting shows average weight of only 0.19, meaning most data is very old. Current prediction of 29.8s seems reasonable for a aging competitor, but applying 325mm data to 275mm without size adjustment is still problematic.

### 4. Cole Schlenker - No Data Fallback Issues

**Historical Data:**
- **ZERO UH results**

**Current Predictions:**
- LLM (selected): 43.8s (confidence: LOW)
- ML model: 24.6s (confidence: MEDIUM)

**Problem:** The system selects the LLM prediction (43.8s) which is based on event baseline with no competitor-specific data. The ML model predicts 24.6s with HIGHER confidence, but isn't being used. The 43.8s default makes Cole the front marker despite having no actual performance history.

## Root Causes

### 1. **Inverted Mark Assignment Logic**

The mark calculation appears to be backwards:
```
mark = max(3, round(slowest_time - their_time + 3))
```

This gives:
- Slowest time → Mark 3 (front marker) ✗ WRONG
- Fastest time → Highest mark (back marker) ✗ WRONG

Should be:
- Fastest time → Mark 3 (front marker) ✓
- Slowest time → Highest mark (back marker) ✓

### 2. **Insufficient Wood Size Adjustment**

When historical data is from different wood sizes (325mm vs 275mm), the prediction system doesn't adequately adjust for the size difference. The system notes "on various wood types" but doesn't properly scale times based on diameter.

### 3. **LLM vs ML Prediction Selection**

The system always prefers LLM predictions over ML predictions, even when:
- LLM has LOW confidence
- ML has MEDIUM or HIGH confidence
- Competitor has no relevant historical data (fallback to generic baseline)

Cole Schlenker's case shows this clearly:
- LLM: 43.8s (LOW confidence, event baseline only)
- ML: 24.6s (MEDIUM confidence, learned from similar competitors)
- **LLM is selected despite lower confidence**

### 4. **Time-Decay Weighting Issues**

While time-decay weighting is implemented, it may not be aggressive enough for aging competitors. David Moses Jr.'s average weight of 0.19 suggests very old data, but the system still gives it significant influence.

## Real-World Performance (User Observation)

User states:
- "Moses and Cole Schlenker have decidedly beat Erin Lavoie in real life"
- "There is no way Erin Lavoie should be marked as high as Cody Labahn"

This confirms:
1. Erin should be a faster/front marker (lower mark number)
2. Moses, despite being older, is still competitive
3. Current predictions don't match observed performance

## Recommendations

### Immediate Fixes Required

1. **Fix mark assignment logic** - Fastest predicted time should get Mark 3 (front), slowest gets highest mark (back)

2. **Improve wood size adjustment** - When using 325mm data to predict 275mm times:
   - Apply diameter-based scaling factor
   - Give preference to exact-size matches
   - Warn judges when predictions are based on different wood sizes

3. **Revise prediction selection logic**:
   - Don't automatically prefer LLM over ML
   - Weight by confidence level
   - For competitors with no data, prefer ML model (learned from similar competitors) over event baseline

4. **Enhanced warnings for judges**:
   - Flag when predictions are based on different wood sizes
   - Flag when competitor has no relevant data
   - Show confidence levels prominently
   - Allow manual override

### Long-term Improvements

1. **Better diameter scaling model** - Learn diameter-to-time relationship from historical data

2. **Cross-validation against real results** - Track prediction accuracy and adjust models

3. **Species-specific models** - Different wood types may require different prediction approaches

4. **Manual override UI** - Allow judges to adjust predictions based on recent observations
