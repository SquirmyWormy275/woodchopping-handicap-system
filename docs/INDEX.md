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

## Technical Documentation

### **[ML_AUDIT_REPORT.md](ML_AUDIT_REPORT.md)** - ML Model Audit
**Status**: Current (Jan 2026)
**Contents**:
- ML model architecture (XGBoost)
- Feature engineering (6 features)
- Time-decay weighting implementation
- Wood characteristics usage
- Diameter scaling analysis
- Handicap calculation validation
- Test results and fairness metrics

### **[HANDICAP_SYSTEM_EXPLAINED.md](HANDICAP_SYSTEM_EXPLAINED.md)** - System Explanation
**Contents**:
- How handicaps work
- AAA rules
- Prediction methodology
- Fairness objectives

### **[NewFeatures.md](NewFeatures.md)** - Planned Enhancements
**Contents**:
- V3 roadmap items
- Future model upgrades (qwen2.5:37b)
- Tournament persistence
- Time-weighted historical data
- 3-Board Jigger support

### **[CHECK_MY_WORK_FEATURE.md](CHECK_MY_WORK_FEATURE.md)** - Check My Work Validation
**Contents**:
- Judge validation system
- Cross-validation features
- Manual override capabilities

### **[QAA_INTERPOLATION_IMPLEMENTATION.md](QAA_INTERPOLATION_IMPLEMENTATION.md)** - QAA Interpolation
**Contents**:
- Quality/Age/Axe scaling logic
- Interpolation algorithms
- Implementation details

---

## Documentation by Purpose

### If you want to...

**...understand how the system works overall**
→ Start with [SYSTEM_STATUS.md](SYSTEM_STATUS.md)

**...learn how to use the program**
→ Read [ReadMe.md](ReadMe.md)

**...understand the ML model**
→ Read [ML_AUDIT_REPORT.md](ML_AUDIT_REPORT.md)

**...contribute code**
→ Read [CLAUDE.md](../CLAUDE.md) for architecture guidelines

**...check what features are planned**
→ Read [NewFeatures.md](NewFeatures.md)

---

## Document Status Legend

- **[CURRENT]**: Up-to-date, reflects current system state
- **[PLANNED]**: Future enhancements not yet implemented

---

## Quick Reference

### System Metrics (V5.0 - Jan 2026)

```
Architecture:
- Fully modular package structure
- Clean separation of concerns
- Production-ready code organization

ML Model Performance:
- SB: MAE 2.55s, R² 0.989
- UH: MAE 2.35s, R² 0.878

Fairness Metrics:
- Target win rate spread: < 2%
- Absolute variance: ±3s for all competitors
- Time-decay half-life: 730 days (2 years)

Feature Importance:
- competitor_avg_time_by_event: 73-82% (dominant)
- Experience: 5-11%
- Wood properties: 10-14%
- Diameter: 1-5%
```

---

## Version Control

This documentation set reflects **Version 5.0** (January 2026)

**Major Features in V5.0**:
- Complete modular architecture
- Multi-event tournament management
- Championship race simulator
- Enhanced Monte Carlo statistics
- Tournament result weighting (97%)
- Diameter scaling with validation
- Comprehensive judge education system

**Last Documentation Update**: January 4, 2026

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
