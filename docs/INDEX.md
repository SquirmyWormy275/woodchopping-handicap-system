# Documentation Index

Complete guide to all documentation in the Woodchopping Handicap System.

---

## Project Structure

```
woodchopping-handicap-system/
├── MainProgramV5_2.py          # Main entry point
├── config.py                   # Configuration settings
├── explanation_system_functions.py  # Judge education wizard
├── woodchopping.xlsx           # Main data file
├── README.md                   # User guide
├── CLAUDE.md                   # Development guidelines
│
├── woodchopping/               # Core package (predictions, handicaps, simulation, ui)
├── tests/                      # Test suite
│   └── validation/             # Validation/backtesting tests
├── scripts/                    # Utility scripts (analysis, data processing)
├── data/                       # Data outputs
│   ├── results/                # CSV results, analysis outputs
│   └── backups/                # Excel backups
├── docs/                       # Documentation (you are here)
│   └── archive/                # Historical implementation reports
├── reference/                  # Competition rules, QAA PDFs
└── saves/                      # Tournament state files (not tracked in git)
```

---

## Start Here

### 1. **[SYSTEM_STATUS.md](SYSTEM_STATUS.md)** - System Overview
**What**: Comprehensive status report of the entire system
**When to Read**: First time exploring the system, checking current capabilities

### 2. **[../README.md](../README.md)** - User Guide (Root Directory)
**What**: User manual and quick start guide
**When to Read**: Learning how to use the program

### 3. **[../CLAUDE.md](../CLAUDE.md)** - Project Architecture (Root Directory)
**What**: AI assistant guidelines and project architecture
**When to Read**: Understanding codebase structure, contributing to the project

---

## Technical Documentation

### **[ML_AUDIT_REPORT.md](ML_AUDIT_REPORT.md)** - ML Model Audit
ML model architecture, feature engineering, performance metrics

### **[HANDICAP_SYSTEM_EXPLAINED.md](HANDICAP_SYSTEM_EXPLAINED.md)** - System Explanation
How handicaps work, AAA rules, prediction methodology

### **[PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)** - Codebase Architecture
Module organization, file purposes, key functions

### **[CHECK_MY_WORK_FEATURE.md](CHECK_MY_WORK_FEATURE.md)** - Check My Work Validation
Judge validation system, cross-validation features

### **[QAA_INTERPOLATION_IMPLEMENTATION.md](QAA_INTERPOLATION_IMPLEMENTATION.md)** - QAA Interpolation
Quality/Age/Axe scaling logic, implementation details

---

## LLM & Prompt Engineering

### **[PROMPT_ENGINEERING_GUIDELINES.md](PROMPT_ENGINEERING_GUIDELINES.md)** - Prompt Guidelines
Core principles, STRATHEX-specific guidelines, testing procedures

### **[PROMPT_CHANGELOG.md](PROMPT_CHANGELOG.md)** - Prompt Version History
LLM prompt changes, rationale, versioning

---

## Archived Documentation

Historical implementation reports are in [archive/](archive/):
- BASELINE_V2_IMPLEMENTATION_SUMMARY.md
- BASELINE_V2_VALIDATION_REPORT.md
- LLM_PROMPT_AUDIT_2026.md
- ML_REDESIGN_IMPLEMENTATION_REPORT.md
- OPTION_5_10_IMPLEMENTATION.md
- PROMPT_UPDATE_SUMMARY_2026-01-12.md
- Tournament and Personnel Changes.md
- V5.2_UI_IMPROVEMENTS.md

---

## Quick Reference

### System Metrics (V5.2 - Jan 2026)

```
ML Model Performance:
- SB: MAE 2.55s, R² 0.989
- UH: MAE 2.35s, R² 0.878

Fairness Metrics:
- Target win rate spread: < 2%
- Absolute variance: ±3s for all competitors
- Time-decay half-life: 730 days (2 years)
```

---

## Contributing

When adding new documentation:
1. Place technical docs in `docs/` directory
2. Update this INDEX.md with new entries
3. Update [SYSTEM_STATUS.md](SYSTEM_STATUS.md) if system capabilities change
4. Update [README.md](../README.md) in root if user-facing changes
5. Archive outdated implementation reports in `docs/archive/`

---

**Maintained by**: Alex Kaper
**AI Assistant**: Claude (Anthropic)
**Last Updated**: January 2026
