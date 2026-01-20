# Woodchopping Handicap System (STRATHEX)

**Version**: 5.2
**Status**: Production Ready
**Last Updated**: January 19, 2026

A data-driven handicap calculation system for woodchopping competitions that combines historical performance analysis, machine learning (XGBoost), and AI-enhanced predictions to create fair, competitive handicaps.

---

## Quick Start

### Running the Program

```bash
# Start the main tournament management program
python MainProgramV5_2.py
```

### Prerequisites

- **Python**: 3.13+
- **Ollama**: Running locally with `qwen2.5:7b` model (for AI predictions)
- **Data File**: `woodchopping.xlsx` in project root

### Required Python Packages

```bash
pip install pandas openpyxl xgboost scikit-learn requests
```

---

## Project Structure

```
woodchopping-handicap-system/
│
├── MainProgramV5_2.py          # Main tournament management interface
├── explanation_system_functions.py  # STRATHEX educational guide for judges
├── config.py                   # System configuration settings
├── woodchopping.xlsx           # Historical results database
├── tournament_state.json       # Saved tournament state
│
├── woodchopping/               # Main package (modular architecture)
│   ├── data/                   # Data loading and validation
│   ├── handicaps/              # Handicap calculation
│   ├── predictions/            # Prediction methods (Baseline, ML, LLM)
│   ├── simulation/             # Monte Carlo fairness validation
│   └── ui/                     # User interface modules (includes bracket_ui.py)
│
├── docs/                       # Documentation
│   ├── ReadMe.md               # Detailed user manual
│   ├── SYSTEM_STATUS.md        # Current system status report
│   ├── ML_AUDIT_REPORT.md      # ML model audit and validation
│   ├── CLAUDE.md               # AI assistant project instructions
│   └── ... (other documentation)
│
├── tests/                      # Test scripts
│   ├── test_both_events.py     # Comprehensive SB & UH testing
│   └── test_uh_predictions.py  # UH-specific prediction tests
│
├── archive/                    # Archived/deprecated files
└── scripts/                    # Utility scripts
```

---

## Core Features

### 1. Multiple Prediction Methods

- **Baseline**: Statistical time-weighted average with diameter scaling
- **ML (XGBoost)**: Separate models for Standing Block (SB) and Underhand (UH)
- **LLM (Ollama)**: AI-enhanced predictions with wood quality reasoning

All methods use **consistent exponential time-decay weighting** (2-year half-life) to prioritize recent performances over historical peaks.

### 2. Intelligent Prediction Selection

```
Priority Logic:
1. Baseline (scaled) if diameter scaling applied with confidence ≥ MEDIUM
2. ML prediction (best for exact diameter matches)
3. LLM prediction (general fallback)
4. Baseline (statistical fallback)
```

### 3. Advanced Features

- **Diameter Scaling**: Physics-based scaling for cross-diameter predictions
- **Wood Quality Adjustment**: ±2% per quality point (0-10 scale)
- **Time-Decay Weighting**: Recent performances weighted higher than old results
- **Monte Carlo Simulation**: Validate handicap fairness with 2 million iterations, individual competitor statistics tracking
- **Multi-Round Tournaments**: Heats → Semi-finals → Finals
- **Multi-Event Tournaments**: Design complete tournament days with multiple independent events (e.g., "225mm SB", "300mm UH", "275mm SB"), sequential results entry, and final tournament summary
- **Championship Race Simulator (NEW V5.0)**: Fun predictive tool for simulating equal-start championship races with AI-powered race analysis, win probability predictions, and competitive matchup insights
- **Prize Money/Payout System (NEW V5.0)**: Comprehensive payout tracking with configurable dollar amounts per placement (1-10 places), final results display with earnings, tournament-wide earnings summary showing total winnings per competitor across all events
- **Bracket Tournaments (NEW V5.0)**: Head-to-head elimination brackets with AI-powered seeding
  - **Single Elimination**: Traditional knockout format with automatic bye placement for non-power-of-2 competitor counts
  - **Double Elimination**: Winners bracket, losers bracket (second-chance), and grand finals
  - **Smart Seeding**: AI predictions determine bracket seeds (fastest = Seed 1, slowest = lowest seed)
  - **Automatic Advancement**: Match results automatically advance winners and drop losers to appropriate brackets
  - **Visual Displays**: ASCII bracket tree visualization and HTML export for sharing
  - **Flexible Configuration**: Reconfigure wood characteristics before bracket generation, locked after creation

### 4. Fairness Metrics

- **Target**: < 1.0s finish time spread (Excellent)
- **Current Performance**: 0.3s - 0.8s spread in testing
- **AAA Rules Compliant**: 3s minimum mark, 180s maximum time

---

## Event Types

- **SB**: Standing Block
- **UH**: Underhand
- **Future**: 3-Board Jigger (pending more training data)

---

## Testing

```bash
# Run comprehensive tests for both SB and UH
python tests/test_both_events.py

# Run UH-specific prediction tests
python tests/test_uh_predictions.py
```

**Test Results** (Dec 24, 2025):
- UH: 0.8s spread [EXCELLENT]
- SB: 0.3s spread [EXCELLENT]

---

## Documentation

### Essential Reading

1. **[SYSTEM_STATUS.md](docs/SYSTEM_STATUS.md)** - Complete system overview and status
2. **[ReadMe.md](docs/ReadMe.md)** - Detailed user manual and function reference
3. **[CLAUDE.md](CLAUDE.md)** - Project architecture and AI assistant guidelines

### Technical Documentation

4. **[ML_AUDIT_REPORT.md](docs/ML_AUDIT_REPORT.md)** - ML model audit and validation
5. **[TIME_DECAY_CONSISTENCY_UPDATE.md](docs/TIME_DECAY_CONSISTENCY_UPDATE.md)** - Time-decay implementation details
6. **[SCALING_IMPROVEMENTS.md](docs/SCALING_IMPROVEMENTS.md)** - Diameter scaling before/after analysis

### Problem Diagnosis

7. **[UH_PREDICTION_ISSUES.md](docs/UH_PREDICTION_ISSUES.md)** - Original UH prediction problem
8. **[DIAGNOSIS.md](docs/DIAGNOSIS.md)** - Initial investigation notes

---

## Key Concepts

### Time-Decay Weighting

**Formula**: `weight = 0.5^(days_old / 730)`

Recent performances are weighted much higher than old results. Critical for aging competitors:

```
Example - Moses (7-year span):
- 2018 peaks (19-22s): weight 0.06 (3%)
- 2023 results (27-28s): weight 0.50 (50%)
- 2025 current (29s): weight 1.00 (100%)

Old system: 24.2s average (inflated by old peaks)
New system: 27.8s average (reflects current ability)
Improvement: 3.6 seconds more accurate!
```

### Diameter Scaling

**Formula**: `time_scaled = time_original × (diameter_target / diameter_original)^1.4`

Allows predictions when historical data diameter ≠ target diameter:

```
Example - Cody (325mm → 275mm):
- Historical: 27s in 325mm
- Scaled: 27 × (275/325)^1.4 = 22.6s
- Direct prediction (no scaling): 27.4s
- Improvement: 4.8 seconds more accurate!
```

### Wood Quality Scale

```
10 = Extremely soft → FAST cutting (baseline × 0.90)
5  = Average → NO adjustment (baseline × 1.00)
0  = Extremely hard → SLOW cutting (baseline × 1.10)

Adjustment: ±2% per quality point from average
```

---

## Data Requirements

### Excel File Structure

**Sheets Required**:
1. `Competitor`: CompetitorID, Name, Country, State/Province, Gender
2. `wood`: Species, Janka Hardness, Specific Gravity, etc.
3. `Results`: CompetitorID, Event, Time (seconds), Size (mm), Species Code, Quality, HeatID, Date

### Minimum Data for Predictions

- **New Competitor**: 3+ historical results
- **ML Training**: 50+ records per event (UH needs more data)
- **Baseline**: Works with any amount of data (cascading fallback)

---

## Production Use

**Recommended for**:
- Missoula Pro-Am
- Mason County Western Qualifier
- Any AAA-sanctioned woodchopping events

**Current Status**: ✓ Production Ready

**Validation**:
- All prediction methods tested and validated
- Fairness metrics excellent (< 1s spread)
- AAA rules compliance verified
- Time-decay weighting consistent across all methods

---

## Support

**Issues or Questions?**
- Check documentation in `docs/` directory
- Review test outputs in `tests/` directory
- Consult `SYSTEM_STATUS.md` for current system state

**Recent Updates**:
- Dec 24, 2025: Time-decay consistency implemented across all prediction methods
- Dec 24, 2025: Wood quality adjustments added to ML and baseline
- Dec 23, 2025: Diameter scaling implemented and validated

---

## Version History

- **V4.4** (Dec 2025): Complete modular migration, tournament result weighting, documentation overhaul
- **V4.3** (Dec 2025): Time-decay consistency, wood quality integration
- **V4.2** (Dec 2025): Added judges approval for handicaps
- **V4.0** (Dec 2025): Modular architecture, separate SB/UH models
- **V3.0** (Nov 2025): Multi-round tournament system

---

**License**: Academic Project
**Author**: Alex Kaper
**AI Assistant**: Claude (Anthropic)
