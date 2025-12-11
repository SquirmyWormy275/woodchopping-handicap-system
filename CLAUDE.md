# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

STRATHEX (Woodchopping Handicap Calculator) is a data-driven system that calculates fair handicap marks for woodchopping competitions. It combines historical performance analysis, AI-enhanced prediction modeling (via Ollama), and Monte Carlo simulation validation to create handicaps that give all competitors equal probability of winning, regardless of skill level.

The system implements a critical innovation: absolute variance modeling (±3 seconds for all competitors) rather than proportional variance, ensuring true fairness as real-world factors affect competitors equally in absolute terms.

## Running the Program

```bash
# Run main program
python MainProgram.py

# Ensure Ollama is running locally for AI predictions
# Model required: qwen2.5:7b (optimized for mathematical reasoning)
```

**Prerequisites:**
- Python 3.13.3
- Ollama running locally with qwen2.5:7b model
- `woodchopping.xlsx` in the same directory as scripts

## Architecture

### Core Components

**MainProgram.py** - Tournament management interface
- Multi-round tournament system (heats → semis → finals)
- Main menu loop with 14 options covering tournament setup, handicap calculation, heat management, and personnel management
- Tournament state management (`tournament_state` dict) tracks event configuration, rounds, competitors, and handicap results
- Legacy single-heat mode preserved for backward compatibility via `heat_assignment_df`

**FunctionsLibrary.py** - Function library (~2500+ lines)
- All business logic: handicap calculation, AI predictions, Monte Carlo simulation, Excel I/O, tournament round generation
- See [ReadMe.md](ReadMe.md) lines 176-211 for complete function dictionary

**woodchopping.xlsx** - Data persistence
- `Competitor` sheet: CompetitorID, Name, Country, State/Province, Gender
- `wood` sheet: Species data with multipliers
- `Results` sheet: Historical performance data (CompetitorID, Event, Time, Species, Diameter, Quality, HeatID, Date)

### Key Architectural Patterns

**Bidirectional ID/Name Mapping**
- Competitors stored by ID in Excel, referenced by name in UI
- `get_competitor_id_name_mapping()` returns both `id_to_name` and `name_to_id` dicts
- Critical for `append_results_to_excel()` and competitor lookups

**Cascading Fallback Logic for Predictions**
- Level 1: Exact match (competitor + species + event)
- Level 2: Competitor + event (any species)
- Level 3: Event baseline (all competitors, that event)
- Implemented in `get_competitor_historical_times_flexible()` and `get_event_baseline_flexible()`

**Absolute Variance Modeling**
- All competitors get ±3 second performance variation in simulations
- Proven fairer than proportional variance (31% vs 6.7% win rate spread in testing)
- Implemented in `simulate_single_race()`

**Tournament State Management**
```python
tournament_state = {
    'event_name': str,
    'num_stands': int,                      # Available chopping stands
    'format': str,                          # 'heats_to_finals' or 'heats_to_semis_to_finals'
    'all_competitors': list,                # Competitor names
    'all_competitors_df': DataFrame,
    'rounds': list,                         # List of round objects (heats/semis/finals)
    'capacity_info': dict,                  # From calculate_tournament_scenarios()
    'handicap_results_all': list            # [{name, mark, predicted_time, ...}, ...]
}
```

Each round object structure:
```python
{
    'round_name': str,                     # "Heat 1", "Semi-Final A", etc.
    'round_type': str,                     # 'heat', 'semi', 'final'
    'competitors': list,                   # Competitor names in this round
    'handicap_results': list,              # Handicap data for these competitors
    'num_to_advance': int,                 # Top N advance to next round
    'status': str,                         # 'pending', 'in_progress', 'completed'
    'results': dict,                       # {competitor_name: actual_time, ...}
    'advancers': list                      # Names of competitors advancing
}
```

**AI Integration**
- Uses Ollama API (localhost:11434) with qwen2.5:7b model
- `call_ollama(prompt, model)` handles all AI requests
- Two AI use cases:
  1. Time prediction with quality adjustments (`predict_competitor_time_with_ai()`)
  2. Fairness assessment of Monte Carlo results (`get_ai_assessment_of_handicaps()`)
- Graceful fallback if Ollama unavailable

## Development Workflow

### Typical Judge Workflow
1. Configure wood characteristics (species, diameter, quality)
2. Configure tournament (stands, format, event name)
3. Select all competitors from master roster
4. Calculate handicaps for all competitors
5. View handicaps + optional Monte Carlo fairness analysis
6. Generate initial heats with balanced skill distribution
7. Record heat results and select advancers (repeat for all heats)
8. Generate next round (semis/finals)
9. Repeat steps 7-8 until tournament complete

### Adding New Functionality

When adding features to FunctionsLibrary.py:
- Follow existing naming conventions (snake_case functions)
- Add function to dictionary in [ReadMe.md](ReadMe.md) lines 176-211
- Excel operations should use openpyxl, not pandas to_excel (for atomic appends)
- All UI functions should return updated state objects (e.g., `wood_selection` dict, DataFrames)
- Use `format_wood(wood_selection)` to display current wood config in menus

### Monte Carlo Simulation

Default: 250,000 race iterations per simulation
- `run_monte_carlo_simulation()` returns analysis dict with win rates, finish positions, spreads
- `visualize_simulation_results()` creates text-based bar charts
- `get_ai_assessment_of_handicaps()` rates fairness: Excellent/Very Good/Good/Fair/Poor/Unacceptable
- Fairness threshold: win rate spread < 2% = Excellent

### AAA Competition Rules Compliance

Hard constraints enforced in code:
- 3-second minimum mark
- 180-second maximum time limit
- Marks rounded up to whole seconds
- Slowest predicted time gets Mark 3 (front marker)
- Fastest predicted time gets highest mark (back marker)

## Data Model

**Competitor Entry Requirements:**
- Minimum 3 historical times required for new competitors
- Each time requires: Event (SB/UH), Time (seconds), Species, Diameter (mm), Quality (0-10), optional Date

**Wood Quality Scale:**
- 0-3: Soft/rotten wood (faster times)
- 4-7: Average firmness for species
- 8-10: Above average firmness (slower times)

**Event Codes:**
- SB: Standing Block
- UH: Underhand
- (Future: 3-Board Jigger planned but not yet implemented)

## Known Future Enhancements

See [NewFeatures.md](NewFeatures.md) for V3 roadmap:
- Upgrade to qwen2.5:37b model
- Re-add manual formula calculations for comparison vs AI
- Tournament persistence (save/load state) - partially implemented
- Time-weighted historical data (weight recent performances more heavily)
- 3-Board Jigger event support (blocked by limited training data)

## Important Context

**Why Absolute Variance:**
Testing revealed proportional variance (e.g., ±5% of predicted time) creates biased results. A 30-second chopper with ±5% = ±1.5s range, while a 60-second chopper gets ±3s range. This gives faster choppers unfair advantage. Absolute ±3s variance proved fair across all skill levels because real-world consistency factors (technique, wood grain, equipment) affect everyone equally.

**Handicap Mark Convention:**
Lower marks = front markers (better competitors, start earlier)
Higher marks = back markers (slower competitors, start later with delay)
All competitors should theoretically finish simultaneously in perfect system.

**Project Origin:**
Academic project for woodchopping competition management. Currently being considered for use at Missoula Pro-Am and Mason County Western Qualifier. Tested against Australian handicapping data (150+ years institutional knowledge).
