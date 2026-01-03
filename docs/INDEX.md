# Documentation Index

Complete guide to all documentation in the Woodchopping Handicap System.

---

## Start Here

### 1. **[SYSTEM_STATUS.md](SYSTEM_STATUS.md)** - System Overview
**What**: Comprehensive status report of the entire system
**When to Read**: First time exploring the system, checking current capabilities
**Contents**:
- Overall system health
- Feature implementation status
- Test results
- Recent improvements
- Known limitations

### 2. **[ReadMe.md](ReadMe.md)** - User Manual
**What**: Detailed user manual and function reference
**When to Read**: Learning how to use the program
**Contents**:
- Program workflow
- Menu options
- Function dictionary
- Data requirements
- AAA rules compliance

### 3. **[CLAUDE.md](../CLAUDE.md)** - Project Architecture (Root Directory)
**What**: AI assistant guidelines and project architecture
**When to Read**: Understanding codebase structure, contributing to the project
**Contents**:
- Project overview
- Core components
- Architectural patterns
- Development workflow

---

## Technical Deep-Dives

### ML Model & Predictions

#### **[ML_AUDIT_REPORT.md](ML_AUDIT_REPORT.md)** - ML Model Audit
**Status**: Updated Dec 24, 2025 (time-decay issue resolved)
**Contents**:
- ML model architecture (XGBoost)
- Feature engineering (6 features)
- Time-decay weighting implementation
- Wood characteristics usage
- Diameter scaling analysis
- Handicap calculation validation
- Test results and fairness metrics

#### **[TIME_DECAY_CONSISTENCY_UPDATE.md](TIME_DECAY_CONSISTENCY_UPDATE.md)** - Time-Decay Implementation
**Date**: Dec 24, 2025
**Status**: IMPLEMENTED
**Contents**:
- Problem identified (ML used simple mean)
- Impact example (David Moses Jr. 3.6s improvement)
- Implementation details (lines changed)
- Testing results (0.3-0.8s spreads)
- Before/after comparisons

#### **[SCALING_IMPROVEMENTS.md](SCALING_IMPROVEMENTS.md)** - Diameter Scaling
**Date**: Dec 23, 2025
**Status**: IMPLEMENTED
**Contents**:
- Test case analysis (275mm Aspen)
- Before/after prediction comparison
- Scaling formula (exponent 1.4)
- Real-world validation
- Files modified

---

## Problem Diagnosis

### **[UH_PREDICTION_ISSUES.md](UH_PREDICTION_ISSUES.md)** - Original UH Problem
**Date**: Dec 23, 2025
**Problem**: UH predictions wildly inaccurate for cross-diameter cases
**Root Cause**: No diameter scaling implemented
**Resolution**: Diameter scaling module created

### **[DIAGNOSIS.md](DIAGNOSIS.md)** - Initial Investigation
**Date**: Dec 23, 2025
**Contents**:
- Historical data analysis
- Competitor-by-competitor breakdown
- Prediction method comparison
- Issue identification

---

## Feature Documentation

### **[NewFeatures.md](NewFeatures.md)** - Planned Enhancements
**Contents**:
- V3 roadmap items
- Future model upgrades (qwen2.5:37b)
- Tournament persistence
- Time-weighted historical data
- 3-Board Jigger support

### **[HANDICAP_SYSTEM_EXPLAINED.md](HANDICAP_SYSTEM_EXPLAINED.md)** - System Explanation
**Contents**:
- How handicaps work
- AAA rules
- Prediction methodology
- Fairness objectives

---

## Development History

### **[MODULE_REFACTORING_COMPLETE.md](MODULE_REFACTORING_COMPLETE.md)** - Modular Refactor
**Date**: Early Dec 2025
**Changes**: Transitioned from monolithic FunctionsLibrary.py to modular `woodchopping/` package

### **[REFACTORING_COMPLETE.md](REFACTORING_COMPLETE.md)** - Refactoring Summary
### **[REFACTORING_SUMMARY.md](REFACTORING_SUMMARY.md)** - Refactoring Details

---

## Documentation by Purpose

### If you want to...

**...understand how the system works overall**
→ Start with [SYSTEM_STATUS.md](SYSTEM_STATUS.md)

**...learn how to use the program**
→ Read [ReadMe.md](ReadMe.md)

**...understand the ML model**
→ Read [ML_AUDIT_REPORT.md](ML_AUDIT_REPORT.md)

**...understand time-decay weighting**
→ Read [TIME_DECAY_CONSISTENCY_UPDATE.md](TIME_DECAY_CONSISTENCY_UPDATE.md)

**...understand diameter scaling**
→ Read [SCALING_IMPROVEMENTS.md](SCALING_IMPROVEMENTS.md)

**...see how a specific problem was solved**
→ Read [UH_PREDICTION_ISSUES.md](UH_PREDICTION_ISSUES.md) or [DIAGNOSIS.md](DIAGNOSIS.md)

**...contribute code**
→ Read [CLAUDE.md](../CLAUDE.md) for architecture guidelines

**...check what features are planned**
→ Read [NewFeatures.md](NewFeatures.md)

---

## Document Status Legend

- **[CURRENT]**: Up-to-date, reflects current system state
- **[RESOLVED]**: Problem documented and fixed
- **[HISTORICAL]**: Documents past state or development process
- **[PLANNED]**: Future enhancements not yet implemented

---

## Quick Reference

### System Metrics (Dec 24, 2025)

```
Test Results:
- UH (275mm Aspen): 0.8s spread [EXCELLENT]
- SB (300mm EWP): 0.3s spread [EXCELLENT]

ML Model Performance:
- SB: MAE 2.55s, R² 0.989 (69 training records)
- UH: MAE 2.35s, R² 0.878 (35 training records)

Feature Importance:
- competitor_avg_time_by_event: 73-82% (dominant)
- Experience: 5-11%
- Wood properties: 10-14%
- Diameter: 1-5%

Time-Decay:
- Half-life: 730 days (2 years)
- Applied to: ALL prediction methods ✓
- Consistency: FULLY CONSISTENT ✓
```

---

## Version Control

This documentation set reflects **Version 4.4** (December 2025)

**Major Changes in V4.4**:
- Complete modular architecture migration (eliminated FunctionsLibrary.py)
- Tournament result weighting (97% same-wood optimization)
- Time-decay weighting made consistent across all methods
- Wood quality adjustments added to ML and baseline
- Diameter scaling implemented and validated
- Comprehensive documentation overhaul

**Last Documentation Update**: December 28, 2025

---

## Contributing

When adding new documentation:
1. Place technical docs in `docs/` directory
2. Update this INDEX.md with new entries
3. Update [SYSTEM_STATUS.md](SYSTEM_STATUS.md) if system capabilities change
4. Update [README.md](../README.md) in root if user-facing changes

---

**Maintained by**: Alex Kaper
**AI Assistant**: Claude (Anthropic)
