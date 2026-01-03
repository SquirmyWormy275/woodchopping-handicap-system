# Project Structure - Quick Reference

**Last Updated**: December 24, 2025

---

## Root Directory (Clean!)

```
woodchopping-handicap-system/
│
├── README.md                      # ← START HERE: Project overview
├── CLAUDE.md                      # AI assistant & architecture guide
├── MainProgramV4_4.py             # ← RUN THIS: Main program
├── config.py                      # System configuration
├── woodchopping.xlsx              # Historical results database
├── tournament_state.json          # Saved tournament state
│
├── FunctionsLibrary.py            # Legacy functions (being phased out)
├── explanation_system_functions.py # Legacy explanations
│
├── woodchopping/                  # Main package (modular code)
│   ├── data/                      #   Data loading & validation
│   ├── handicaps/                 #   Handicap calculation
│   ├── predictions/               #   Baseline, ML, LLM predictions
│   ├── simulation/                #   Monte Carlo fairness
│   └── ui/                        #   User interface
│
├── docs/                          # All documentation (organized!)
├── tests/                         # Test scripts
├── archive/                       # Old/deprecated files
└── scripts/                       # Utility scripts
```

---

## What's Where

### Want to USE the program?
```
1. Open: MainProgramV4_4.py
2. Read: docs/ReadMe.md (user manual)
```

### Want to UNDERSTAND the system?
```
1. Read: README.md (project overview)
2. Read: docs/SYSTEM_STATUS.md (detailed status)
3. Read: docs/INDEX.md (documentation guide)
```

### Want to SEE how it works?
```
1. Run: tests/test_both_events.py
2. Review: docs/ML_AUDIT_REPORT.md
```

### Want to MODIFY the code?
```
1. Read: CLAUDE.md (architecture)
2. Explore: woodchopping/ package
3. Reference: docs/MODULE_REFACTORING_COMPLETE.md
```

---

## Documentation Organization

All documentation now in `docs/`:

### Essential Reading
- **INDEX.md** - Documentation index (start here!)
- **SYSTEM_STATUS.md** - Current system capabilities
- **ReadMe.md** - User manual & function reference

### Technical Documentation
- **ML_AUDIT_REPORT.md** - ML model audit & validation
- **TIME_DECAY_CONSISTENCY_UPDATE.md** - Time-decay implementation
- **SCALING_IMPROVEMENTS.md** - Diameter scaling analysis

### Problem Solving
- **UH_PREDICTION_ISSUES.md** - Original UH problem diagnosis
- **DIAGNOSIS.md** - Initial investigation

### Project Info
- **HANDICAP_SYSTEM_EXPLAINED.md** - How handicaps work
- **NewFeatures.md** - Planned enhancements
- **MODULE_REFACTORING_COMPLETE.md** - Modular refactor history

---

## Testing Organization

All tests now in `tests/`:

### Main Tests
- **test_both_events.py** - Comprehensive SB & UH validation
- **test_uh_predictions.py** - UH-specific prediction tests

**Run tests**:
```bash
cd tests
python test_both_events.py
```

---

## Code Organization (woodchopping/ package)

```
woodchopping/
│
├── data/
│   ├── __init__.py                # Data exports
│   └── excel_io.py                # Excel loading & validation
│
├── handicaps/
│   ├── __init__.py
│   └── calculator.py              # Handicap calculation logic
│
├── predictions/
│   ├── __init__.py
│   ├── baseline.py                # Statistical predictions + time-decay
│   ├── ml_model.py                # XGBoost training & prediction
│   ├── llm.py                     # Ollama API integration
│   ├── ai_predictor.py            # LLM prediction logic
│   ├── diameter_scaling.py        # Diameter scaling calculations
│   └── prediction_aggregator.py   # Prediction selection logic
│
├── simulation/
│   ├── __init__.py
│   └── fairness.py                # Monte Carlo validation
│
└── ui/
    ├── __init__.py
    ├── wood_ui.py                 # Wood configuration UI
    ├── competitor_ui.py           # Competitor selection UI
    ├── handicap_ui.py             # Handicap display UI
    └── tournament_ui.py           # Tournament management UI
```

---

## File Naming Conventions

### Python Files
- `MainProgramV4_4.py` - Main entry point (CamelCase + version)
- `module_name.py` - Modules (snake_case)
- `test_feature.py` - Tests (test_ prefix)

### Documentation Files
- `README.md` - Main project overview (all caps)
- `CLAUDE.md` - AI assistant instructions (all caps)
- `FEATURE_NAME.md` - Feature docs (all caps)
- `ReadMe.md` - User manual (legacy CamelCase)

---

## Recent Cleanup (Dec 24, 2025)

### Moved to `docs/`:
- DIAGNOSIS.md
- ML_AUDIT_REPORT.md
- SCALING_IMPROVEMENTS.md
- SYSTEM_STATUS.md
- TIME_DECAY_CONSISTENCY_UPDATE.md
- UH_PREDICTION_ISSUES.md

### Moved to `tests/`:
- test_both_events.py
- test_uh_predictions.py

### Created:
- README.md (root directory overview)
- docs/INDEX.md (documentation index)
- PROJECT_STRUCTURE.md (this file)

### Result:
**Root directory**: 12 essential files (was 18+ scattered files)
**Docs directory**: 13 organized documents (was 6)
**Tests directory**: 2 test scripts (new organization)

---

## Quick Navigation

```bash
# View project structure
ls -R

# Read main README
cat README.md

# Browse documentation
cd docs
ls
cat INDEX.md

# Run tests
cd tests
python test_both_events.py

# Start program
python MainProgramV4_4.py
```

---

## Notes

- All markdown files use `.md` extension
- Legacy files kept in `archive/` directory
- `__pycache__/` auto-generated by Python (can ignore)
- `tournament_state.json` auto-saved by program

---

**Project Status**: Clean, Organized, Production Ready ✓
