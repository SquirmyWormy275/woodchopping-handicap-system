"""Multi-event tournament management UI functions.

This module handles multi-event tournament operations including:
- Tournament creation and metadata management
- Event addition, viewing, and removal
- Batch operations across all events
- Sequential results entry workflow
- Tournament summary generation
- Multi-event state persistence
"""

import json
import copy
import sys
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import pandas as pd

# Import existing functions for reuse
from woodchopping.ui.wood_ui import wood_menu, format_wood, select_event_code
from woodchopping.ui.competitor_ui import select_all_event_competitors
from woodchopping.ui.tournament_ui import (
    calculate_tournament_scenarios,
    distribute_competitors_into_heats,
    select_heat_advancers,
    generate_next_round
)
from woodchopping.handicaps import calculate_ai_enhanced_handicaps
from woodchopping.data import load_results_df, append_results_to_excel
from woodchopping.ui.adjustment_tracking import log_handicap_adjustment


def create_multi_event_tournament() -> Dict:
    """Create a new multi-event tournament structure.

    Prompts judge for:
    - Tournament name
    - Tournament date (optional, defaults to today)

    Returns:
        dict: Initialized multi_event_tournament_state
    """
    print(f"\n{'='*70}")
    print(f"  CREATE NEW MULTI-EVENT TOURNAMENT")
    print(f"{'='*70}")

    # Prompt for tournament name
    tournament_name = input("\nTournament name (e.g., 'Missoula Pro-Am 2026'): ").strip()
    if not tournament_name:
        tournament_name = "Unnamed Tournament"

    # Prompt for date (optional)
    date_input = input("Tournament date (YYYY-MM-DD, or press Enter for today): ").strip()
    if date_input:
        tournament_date = date_input
    else:
        tournament_date = datetime.now().strftime("%Y-%m-%d")

    # Initialize tournament state
    tournament_state = {
        'tournament_mode': 'multi_event',
        'tournament_name': tournament_name,
        'tournament_date': tournament_date,
        'created_at': datetime.now().isoformat(),
        'total_events': 0,
        'events_completed': 0,
        'current_event_index': 0,
        'events': []
    }

    print(f"\n✓ Tournament '{tournament_name}' created for {tournament_date}")
    print(f"✓ You can now add events to this tournament")

    return tournament_state


def save_multi_event_tournament(tournament_state: Dict, filename: str = "multi_tournament_state.json") -> None:
    """Save multi-event tournament state to JSON file.

    Handles DataFrame serialization and NumPy type conversion for all events and rounds.

    Args:
        tournament_state: Multi-event tournament state dictionary
        filename: Output filename (default: multi_tournament_state.json)
    """
    import numpy as np

    class NumpyEncoder(json.JSONEncoder):
        """Custom JSON encoder for NumPy types."""
        def default(self, obj):
            if isinstance(obj, (np.integer, np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)

    try:
        # Deep copy to avoid mutating original
        state_copy = copy.deepcopy(tournament_state)

        # Convert DataFrames to dict records for JSON serialization
        for event in state_copy.get('events', []):
            # Convert event-level DataFrame
            if 'all_competitors_df' in event and isinstance(event['all_competitors_df'], pd.DataFrame):
                event['all_competitors_df'] = event['all_competitors_df'].to_dict('records')

            # Convert round-level DataFrames
            for round_obj in event.get('rounds', []):
                if 'competitors_df' in round_obj and isinstance(round_obj['competitors_df'], pd.DataFrame):
                    round_obj['competitors_df'] = round_obj['competitors_df'].to_dict('records')

        # Write to JSON with custom encoder for NumPy types
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(state_copy, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)

        print(f"\n✓ Tournament state saved to {filename}")

    except Exception as e:
        print(f"\n⚠ Error saving tournament state: {e}")


def load_multi_event_tournament(filename: str = "multi_tournament_state.json") -> Optional[Dict]:
    """Load multi-event tournament state from JSON file.

    Reconstructs DataFrames from dict records.

    Args:
        filename: Input filename (default: multi_tournament_state.json)

    Returns:
        dict: Loaded tournament state, or None if load failed
    """
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            tournament_state = json.load(f)

        # Reconstruct DataFrames
        for event in tournament_state.get('events', []):
            # Reconstruct event-level DataFrame
            if 'all_competitors_df' in event and isinstance(event['all_competitors_df'], list):
                event['all_competitors_df'] = pd.DataFrame(event['all_competitors_df'])

            # Reconstruct round-level DataFrames
            for round_obj in event.get('rounds', []):
                if 'competitors_df' in round_obj and isinstance(round_obj['competitors_df'], list):
                    round_obj['competitors_df'] = pd.DataFrame(round_obj['competitors_df'])

            # Backward compatibility: add event_type for legacy tournaments
            if 'event_type' not in event:
                event['event_type'] = 'handicap'

            # Backward compatibility: add payout_config for legacy tournaments (V4.5)
            if 'payout_config' not in event:
                event['payout_config'] = None

        print(f"\n✓ Tournament state loaded from {filename}")
        print(f"✓ Tournament: {tournament_state.get('tournament_name', 'Unknown')}")
        print(f"✓ Events: {tournament_state.get('total_events', 0)}")

        return tournament_state

    except FileNotFoundError:
        print(f"\n⚠ Tournament file '{filename}' not found")
        return None
    except Exception as e:
        print(f"\n⚠ Error loading tournament state: {e}")
        return None


def auto_save_multi_event(tournament_state: Dict) -> None:
    """Auto-save multi-event tournament state with default filename.

    Args:
        tournament_state: Multi-event tournament state dictionary
    """
    save_multi_event_tournament(tournament_state, "multi_tournament_state.json")


def add_event_to_tournament(tournament_state: Dict, comp_df: pd.DataFrame, results_df: pd.DataFrame) -> Dict:
    """Add a new event to the tournament (wood, format, competitors ONLY).

    Sequential workflow:
    1. Wood characteristics (species, diameter, quality)
    2. Event code (SB/UH)
    3. Auto-generate event name (e.g., "275mm UH", "300mm SB")
    4. Tournament format (stands, heats structure)
    5. Competitor selection
    6. Set event status to 'configured' (NO HANDICAP CALCULATION)

    Handicaps will be calculated later in batch using calculate_all_event_handicaps().

    Args:
        tournament_state: Multi-event tournament state
        comp_df: Full competitor roster
        results_df: Historical results DataFrame (not used in this function anymore)

    Returns:
        dict: Updated tournament_state with new event (status: 'configured')
    """
    print(f"\n{'='*70}")
    print(f"  ADD NEW EVENT TO TOURNAMENT")
    print(f"{'='*70}")
    print(f"Tournament: {tournament_state.get('tournament_name', 'Unknown')}")
    print(f"Current events: {tournament_state.get('total_events', 0)}")
    print(f"{'='*70}")

    # Generate event ID
    event_order = tournament_state['total_events'] + 1
    event_id = f"event_{event_order}"

    # Step 1: Wood characteristics
    print(f"\n{'='*70}")
    print(f"  WOOD CONFIGURATION")
    print(f"{'='*70}")

    wood_selection = {
        "species": None,
        "size_mm": None,
        "quality": None,
        "event": None
    }

    # Reuse wood menu for species, size, quality
    wood_selection = wood_menu(wood_selection)

    # Validate wood configuration
    if not wood_selection.get('species') or not wood_selection.get('size_mm') or wood_selection.get('quality') is None:
        print("\n⚠ Incomplete wood configuration. Cancelling event addition...")
        return tournament_state

    # Step 2: Event code (if not already set by wood menu)
    if not wood_selection.get('event'):
        wood_selection = select_event_code(wood_selection)

    # Auto-generate event name from wood configuration
    event_name = f"{int(wood_selection['size_mm'])}mm {wood_selection['event']}"
    print(f"\n✓ Event name auto-generated: {event_name}")

    # Step 2.5: Event type selection (NEW V5.0 - includes bracket)
    print(f"\n{'='*70}")
    print(f"  EVENT TYPE FOR: {event_name}")
    print(f"{'='*70}")
    print("\n1. Handicap Event (AI-predicted marks for fair competition)")
    print("2. Championship Event (Mark 3 for all - fastest time wins)")
    print("3. Bracket Event (Head-to-head single elimination)")
    print("\nHandicap: Historical data + AI calculates individual marks for fairness")
    print("Championship: Everyone starts together (Mark 3), fastest time wins")
    print("Bracket: Single elimination head-to-head, AI-seeded, 2 stands only")

    event_type_choice = input("\nSelect event type (1, 2, or 3): ").strip()

    if event_type_choice == '1':
        event_type = 'handicap'
        print("\n✓ Handicap event - marks will be calculated in batch later")
    elif event_type_choice == '2':
        event_type = 'championship'
        print("\n✓ Championship event - all competitors will get Mark 3")
    elif event_type_choice == '3':
        event_type = 'bracket'
        print("\n✓ Bracket event - single elimination tournament with AI seeding")
    else:
        print("\n⚠ Invalid choice. Defaulting to Handicap event")
        event_type = 'handicap'

    # Step 3: Tournament format (stands + format)
    print(f"\n{'='*70}")
    print(f"  TOURNAMENT FORMAT FOR: {event_name}")
    print(f"{'='*70}")

    # BRACKET MODE: Force 2 stands, skip format selection
    if event_type == 'bracket':
        num_stands = 2
        print(f"\n✓ Bracket mode requires exactly 2 stands (head-to-head matches)")

        try:
            tentative = int(input("Approximate number of competitors for this event: ").strip())
        except ValueError:
            print("\n⚠ Invalid input. Cancelling event addition...")
            return tournament_state

        event_format = 'bracket'
        capacity_info = {
            'max_competitors': 999,  # No limit for brackets
            'format_description': 'Single elimination bracket'
        }

        print(f"✓ Bracket tournament - supports any number of competitors (auto byes)")

    # REGULAR MODES: User selects stands and format
    else:
        try:
            num_stands = int(input("\nNumber of available stands for this event: ").strip())
            tentative = int(input("Approximate number of competitors for this event: ").strip())
        except ValueError:
            print("\n⚠ Invalid input. Cancelling event addition...")
            return tournament_state

        # Calculate scenarios
        scenarios = calculate_tournament_scenarios(num_stands, tentative)

        # Display scenarios
        print(f"\n{'='*70}")
        print(f"  SCENARIO 1: Single Heat Mode")
        print(f"{'='*70}")
        print(scenarios['single_heat']['description'])

        print(f"\n{'='*70}")
        print(f"  SCENARIO 2: Heats → Finals")
        print(f"{'='*70}")
        print(scenarios['heats_to_finals']['description'])

        print(f"\n{'='*70}")
        print(f"  SCENARIO 3: Heats → Semis → Finals")
        print(f"{'='*70}")
        print(scenarios['heats_to_semis_to_finals']['description'])

        # User selects format
        print(f"\n{'='*70}")
        format_choice = input("Select format (1, 2, or 3): ").strip()

        if format_choice == '1':
            event_format = 'single_heat'
            capacity_info = scenarios['single_heat']
        elif format_choice == '2':
            event_format = 'heats_to_finals'
            capacity_info = scenarios['heats_to_finals']
        elif format_choice == '3':
            event_format = 'heats_to_semis_to_finals'
            capacity_info = scenarios['heats_to_semis_to_finals']
        else:
            print("\n⚠ Invalid choice. Cancelling event addition...")
            return tournament_state

    # Step 5: Competitor selection
    print(f"\n{'='*70}")
    print(f"  SELECT COMPETITORS FOR: {event_name}")
    print(f"{'='*70}")

    max_comp = capacity_info.get('max_competitors')
    selected_df = select_all_event_competitors(comp_df, max_comp)

    if selected_df.empty:
        print("\n⚠ No competitors selected. Cancelling event addition...")
        return tournament_state

    # Step 5.5: Payout configuration (OPTIONAL) - NEW V5.0
    print(f"\n{'='*70}")
    print(f"  PAYOUT CONFIGURATION (OPTIONAL)")
    print(f"{'='*70}")
    print("\nWould you like to configure payouts for this event?")
    print("(You can skip this if it's a non-cash event)")

    configure_payouts = input("\nConfigure payouts? (y/n): ").strip().lower()

    if configure_payouts == 'y':
        from woodchopping.ui.payout_ui import configure_event_payouts
        payout_config = configure_event_payouts()

        if payout_config:
            print(f"\n✓ Payout configuration saved for {event_name}")
        else:
            payout_config = {'enabled': False}
            print(f"\n✓ Payouts skipped for {event_name}")
    else:
        payout_config = {'enabled': False}
        print(f"\n✓ Payouts skipped for {event_name}")

    # Generate setup for Championship and Bracket events immediately
    if event_type == 'championship':
        print(f"\n{'='*70}")
        print(f"  CHAMPIONSHIP EVENT - AUTO-GENERATING MARKS")
        print(f"{'='*70}")
        print(f"\nAll {len(selected_df)} competitors assigned Mark 3")

        # Create handicap results with Mark 3 for everyone
        handicap_results_all = []
        for comp_name in selected_df['competitor_name'].tolist():
            handicap_results_all.append({
                'name': comp_name,
                'predicted_time': 0.0,  # Not used for championship
                'method_used': 'Championship',
                'confidence': 'N/A',
                'explanation': 'Championship event: fastest time wins',
                'predictions': {},
                'mark': 3
            })

        event_status = 'ready'  # Skip 'configured' status
        print("✓ Championship event ready for heat generation (skips batch calculation)")

    elif event_type == 'bracket':
        # Bracket event - generate predictions and bracket structure immediately
        print(f"\n{'='*70}")
        print(f"  BRACKET EVENT - GENERATING PREDICTIONS & BRACKET")
        print(f"{'='*70}")

        from woodchopping.ui.bracket_ui import generate_bracket_seeds, generate_bracket_with_byes

        # Generate predictions for seeding
        predictions = generate_bracket_seeds(
            selected_df,
            wood_selection['species'],
            wood_selection['size_mm'],
            wood_selection['quality'],
            wood_selection['event']
        )

        # Generate bracket structure with byes
        rounds = generate_bracket_with_byes(predictions)

        # Calculate bracket info
        num_competitors = len(predictions)
        total_rounds = len(rounds)
        total_matches = sum(len(r['matches']) for r in rounds)

        print(f"\n✓ Bracket generated successfully!")
        print(f"  Competitors: {num_competitors}")
        print(f"  Total Rounds: {total_rounds}")
        print(f"  Total Matches: {total_matches}")
        print(f"  Seeding: Fastest predicted time = Seed 1")

        # Store bracket data in event object
        handicap_results_all = []  # Not used for brackets
        event_status = 'ready'  # Ready for match entry

    else:
        # Handicap event - will calculate marks in batch later
        handicap_results_all = []
        event_status = 'configured'

    # Create event object
    event_obj = {
        'event_id': event_id,
        'event_name': event_name,
        'event_order': event_order,
        'status': event_status,
        'event_type': event_type,

        # Wood characteristics
        'wood_species': wood_selection['species'],
        'wood_diameter': wood_selection['size_mm'],
        'wood_quality': wood_selection['quality'],
        'event_code': wood_selection['event'],

        # Tournament configuration
        'num_stands': num_stands,
        'format': event_format,
        'capacity_info': capacity_info,

        # Competitors
        'all_competitors': selected_df['competitor_name'].tolist(),
        'all_competitors_df': selected_df,

        # Handicaps (populated for Championship, empty for Handicap/Bracket events)
        'handicap_results_all': handicap_results_all,

        # Rounds (empty for Handicap/Championship, populated for Bracket)
        'rounds': rounds if event_type == 'bracket' else [],

        # Final results (empty until finals complete)
        'final_results': {
            'first_place': None,
            'second_place': None,
            'third_place': None,
            'all_placements': {}
        },

        # Payout configuration (NEW V5.0)
        'payout_config': payout_config
    }

    # Add bracket-specific fields if bracket event
    if event_type == 'bracket':
        event_obj['predictions'] = predictions
        event_obj['num_competitors'] = num_competitors
        event_obj['total_rounds'] = total_rounds
        event_obj['total_matches'] = total_matches
        event_obj['current_round_number'] = 1
        event_obj['completed_matches'] = 0

    # Add event to tournament
    tournament_state['events'].append(event_obj)
    tournament_state['total_events'] += 1

    print(f"\n{'='*70}")
    print(f"  ✓ EVENT ADDED SUCCESSFULLY")
    print(f"{'='*70}")
    print(f"Event name: {event_name}")
    print(f"Event order: {event_order}")
    print(f"Event type: {event_type.upper()}")
    print(f"Wood: {wood_selection['size_mm']}mm {wood_selection['species']} (Quality {wood_selection['quality']})")
    print(f"Event code: {wood_selection['event']}")
    print(f"Format: {event_format}")
    print(f"Competitors: {len(selected_df)}")
    if event_type == 'championship':
        print(f"Marks: All competitors have Mark 3")
        print(f"Status: Ready (Championship events skip handicap calculation)")
    else:
        print(f"Handicaps: NOT YET CALCULATED (use batch calculation)")
        print(f"Status: Configured (ready for handicap calculation)")
    print(f"{'='*70}")

    # Auto-save
    auto_save_multi_event(tournament_state)

    input("\nPress Enter to continue...")

    return tournament_state


def calculate_all_event_handicaps(tournament_state: Dict, results_df: pd.DataFrame) -> Dict:
    """Calculate handicaps for ALL events in the tournament (BATCH OPERATION).

    This is the key function for the batch handicap workflow. It processes all
    configured events and calculates handicaps in one operation.

    Workflow:
    1. Validate all events are in 'configured' state
    2. For each event, calculate handicaps using calculate_ai_enhanced_handicaps()
    3. Display progress for each event
    4. Update event status to 'ready' after calculation
    5. Display final summary
    6. Auto-save tournament state

    Args:
        tournament_state: Multi-event tournament state
        results_df: Historical results DataFrame for handicap calculations

    Returns:
        dict: Updated tournament_state with handicaps calculated for all events
    """
    from woodchopping.handicaps import calculate_ai_enhanced_handicaps

    print(f"\n{'╔' + '═' * 68 + '╗'}")
    title = "BATCH HANDICAP CALCULATION".center(68)
    print(f"{'║'}{title}{'║'}")
    print(f"{'╚' + '═' * 68 + '╝'}")

    # Validation: Check all events are configured
    not_configured = [e for e in tournament_state.get('events', []) if e['status'] == 'pending']
    if not_configured:
        print(f"\n⚠ ERROR: {len(not_configured)} event(s) not configured:")
        for event in not_configured:
            print(f"  - {event['event_name']}: {event['status']}")
        print("\nPlease configure all events before calculating handicaps.")
        input("\nPress Enter to continue...")
        return tournament_state

    if not tournament_state.get('events'):
        print("\n⚠ No events to calculate handicaps for.")
        input("\nPress Enter to continue...")
        return tournament_state

    # Check if handicaps already calculated
    already_calculated = [e for e in tournament_state['events'] if e['handicap_results_all']]
    if already_calculated:
        print(f"\n⚠ WARNING: {len(already_calculated)} event(s) already have handicaps calculated.")
        print("This will RECALCULATE handicaps for all events.")
        confirm = input("\nContinue? (y/n): ").strip().lower()
        if confirm != 'y':
            print("\n✓ Handicap calculation cancelled")
            input("\nPress Enter to continue...")
            return tournament_state

    total_events = len(tournament_state['events'])
    total_competitors = sum(len(e['all_competitors']) for e in tournament_state['events'])

    print(f"\nCalculating handicaps for {total_events} events ({total_competitors} total competitors)...")
    print(f"{'='*70}\n")

    # Calculate handicaps for each event
    for event_idx, event in enumerate(tournament_state['events'], 1):
        # Skip Championship events (already have Mark 3 assigned)
        if event.get('event_type') == 'championship':
            print(f"{'='*70}")
            print(f"  EVENT {event_idx} of {total_events}: {event['event_name']}")
            print(f"{'='*70}")
            print(f"○ Skipping Championship event (marks pre-assigned: all Mark 3)")
            print(f"{'='*70}\n")
            continue

        print(f"{'='*70}")
        print(f"  EVENT {event_idx} of {total_events}: {event['event_name']}")
        print(f"{'='*70}")
        print(f"Wood: {event['wood_diameter']}mm {event['wood_species']} (Quality {event['wood_quality']})")
        print(f"Event: {event['event_code']}")
        print(f"Competitors: {len(event['all_competitors'])}")
        print()

        def show_progress(current, total, comp_name):
            """Display progress for this event"""
            percent = int((current / total) * 100)
            bar_length = 40
            filled = int((bar_length * current) / total)
            bar = '█' * filled + '░' * (bar_length - filled)
            print(f"  [{bar}] {percent:3d}% - {comp_name}")

        # Calculate handicaps for this event
        handicap_results = calculate_ai_enhanced_handicaps(
            event['all_competitors_df'],
            event['wood_species'],
            event['wood_diameter'],
            event['wood_quality'],
            event['event_code'],
            results_df,
            progress_callback=show_progress
        )

        if not handicap_results:
            print(f"\n⚠ Failed to calculate handicaps for {event['event_name']}")
            print("Skipping this event...")
            continue

        # Update event with handicap results
        event['handicap_results_all'] = handicap_results
        event['status'] = 'ready'  # Now ready for heat generation

        print(f"\n✓ Event {event_idx}: Handicaps calculated for {len(handicap_results)} competitors")
        print()

    # Display final summary
    print(f"{'='*70}")
    print(f"  ✓ BATCH HANDICAP CALCULATION COMPLETE")
    print(f"{'='*70}")

    calculated_events = [e for e in tournament_state['events'] if e['status'] == 'ready']
    calculated_competitors = sum(len(e['handicap_results_all']) for e in calculated_events)

    print(f"Events processed: {len(calculated_events)}/{total_events}")
    print(f"Competitors analyzed: {calculated_competitors}")
    print(f"Status: All events ready for heat generation")
    print(f"{'='*70}")

    # Auto-save
    auto_save_multi_event(tournament_state)
    print("\n✓ Tournament state auto-saved")

    input("\nPress Enter to continue...")
    return tournament_state


def analyze_single_event(event: Dict, event_index: int, tournament_state: Dict) -> None:
    """Comprehensive handicap analysis workflow for a single event.

    Provides analysis tools:
    - Display handicap marks
    - Monte Carlo fairness simulation (optional)
    - AI prediction method analysis (optional)
    - Handicap calculation explanation (optional)

    Marks event as 'analysis_completed' when full workflow is done.

    Args:
        event: Event dict to analyze
        event_index: Index in tournament_state['events']
        tournament_state: Multi-event tournament state dict
    """
    from woodchopping.simulation import simulate_and_assess_handicaps
    from woodchopping.predictions.prediction_aggregator import (
        display_comprehensive_prediction_analysis,
        display_handicap_calculation_explanation
    )

    print(f"\n{'╔' + '═' * 68 + '╗'}")
    title = f"HANDICAP ANALYSIS: {event['event_name']}".center(68)
    print(f"{'║'}{title}{'║'}")
    print(f"{'╚' + '═' * 68 + '╝'}")

    # Validation: Check handicaps are calculated
    if event['status'] not in ['ready', 'scheduled', 'in_progress', 'completed']:
        print(f"\n⚠ ERROR: Handicaps not calculated for this event (status: {event['status']})")
        print("\nPlease calculate handicaps first (Option 5).")
        input("\nPress Enter to continue...")
        return

    if not event['handicap_results_all']:
        print("\n⚠ No handicaps calculated for this event.")
        input("\nPress Enter to continue...")
        return

    # Championship events: simple confirmation display (skip analysis)
    if event.get('event_type') == 'championship':
        print(f"\n{'='*70}")
        print(f"  CHAMPIONSHIP EVENT - MARK CONFIRMATION")
        print(f"{'='*70}")
        print(f"Wood: {event['wood_diameter']}mm {event['wood_species']} (Quality {event['wood_quality']})")
        print(f"Event type: {event['event_code']}")
        print(f"\n✓ Championship Event")
        print(f"✓ All {len(event['handicap_results_all'])} competitors have Mark 3")
        print(f"✓ Fastest time wins - no handicap analysis needed")
        print(f"✓ All competitors start simultaneously")
        print(f"{'='*70}")

        # Mark as analysis completed (simple approval)
        event['analysis_completed'] = True
        tournament_state['events'][event_index] = event

        input("\nPress Enter to continue...")
        return

    # PHASE 1: Display handicap marks
    print(f"\n{'='*70}")
    print(f"  HANDICAP MARKS")
    print(f"{'='*70}")
    print(f"Wood: {event['wood_diameter']}mm {event['wood_species']} (Quality {event['wood_quality']})")
    print(f"Event type: {event['event_code']}")
    print(f"{'='*70}\n")

    # Display handicaps sorted by mark
    sorted_results = sorted(event['handicap_results_all'], key=lambda x: x['mark'])

    print(f"{'Competitor':<35s} {'Predicted Time':<15s} {'Mark':<6s} {'Method'}")
    print(f"{'-'*70}")

    for result in sorted_results:
        name = result['name'][:34]
        time = f"{result['predicted_time']:.1f}s"
        mark = f"{result['mark']}"
        method = result.get('method_used', 'Unknown')[:15]

        print(f"{name:<35s} {time:<15s} {mark:<6s} {method}")

    print(f"{'-'*70}")
    print(f"Total competitors: {len(event['handicap_results_all'])}\n")

    # PHASE 2: Monte Carlo fairness simulation (optional)
    print(f"\n{'='*70}")
    run_mc = input("\nRun Monte Carlo fairness simulation? (y/n): ").strip().lower()
    if run_mc == 'y':
        print(f"\n{'╔' + '═' * 68 + '╗'}")
        title = f"MONTE CARLO SIMULATION".center(68)
        print(f"{'║'}{title}{'║'}")
        print(f"{'╚' + '═' * 68 + '╝'}\n")

        simulate_and_assess_handicaps(event['handicap_results_all'])

    # PHASE 3: AI prediction method analysis (optional)
    print(f"\n{'='*70}")
    show_ai = input("\nView detailed AI analysis of prediction methods? (y/n): ").strip().lower()
    if show_ai == 'y':
        print(f"\n{'╔' + '═' * 68 + '╗'}")
        title = f"AI PREDICTION ANALYSIS".center(68)
        print(f"{'║'}{title}{'║'}")
        print(f"{'╚' + '═' * 68 + '╝'}\n")

        # Prepare wood selection dict for this event
        wood_selection = {
            'species': event['wood_species'],
            'size_mm': event['wood_diameter'],
            'quality': event['wood_quality'],
            'event': event['event_code']
        }

        try:
            display_comprehensive_prediction_analysis(
                event['handicap_results_all'],
                wood_selection
            )
        except Exception as e:
            print(f"\n⚠ Error during AI analysis: {e}")
            print("Continuing...")

    # PHASE 4: Explanation of handicap system (optional)
    print(f"\n{'='*70}")
    show_explanation = input("\nView explanation of how handicaps are calculated? (y/n): ").strip().lower()
    if show_explanation == 'y':
        display_handicap_calculation_explanation()

    # Mark as having completed full analysis workflow
    print(f"\n{'='*70}")
    mark_complete = input("\nMark this event's analysis as complete? (y/n): ").strip().lower()
    if mark_complete == 'y':
        event['analysis_completed'] = True
        print("\n✓ Analysis marked as complete")
        print("  You can now manually adjust handicaps for this event in the Approval menu.")
        # Auto-save
        auto_save_multi_event(tournament_state)
        print("✓ Tournament state auto-saved")
    else:
        print("\n⚠ Analysis not marked complete")
        print("  Manual handicap adjustments will not be available for this event.")

    input("\nPress Enter to continue...")


def view_analyze_all_handicaps(tournament_state: Dict) -> None:
    """Event selection menu for handicap analysis.

    Allows judges to select individual events for comprehensive analysis.

    Args:
        tournament_state: Multi-event tournament state dict
    """
    while True:
        print(f"\n{'╔' + '═' * 68 + '╗'}")
        title = "HANDICAP ANALYSIS - EVENT SELECTION".center(68)
        print(f"{'║'}{title}{'║'}")
        print(f"{'╚' + '═' * 68 + '╝'}")

        # Validation: Check handicaps are calculated
        not_ready = [e for e in tournament_state.get('events', []) if e['status'] not in ['ready', 'scheduled', 'in_progress', 'completed']]
        if not_ready:
            print(f"\n⚠ ERROR: {len(not_ready)} event(s) don't have handicaps calculated:")
            for event in not_ready:
                print(f"  - {event['event_name']}: {event['status']}")
            print("\nPlease calculate handicaps first (Option 5).")
            input("\nPress Enter to continue...")
            return

        if not tournament_state.get('events'):
            print("\n⚠ No events in tournament.")
            input("\nPress Enter to continue...")
            return

        # Display events
        print(f"\n{'='*70}")
        print(f"  SELECT EVENT TO ANALYZE")
        print(f"{'='*70}\n")

        for idx, event in enumerate(tournament_state['events'], 1):
            status_icon = "✓" if event.get('analysis_completed', False) else " "
            print(f"{idx}. [{status_icon}] {event['event_name']}")
            print(f"    Wood: {event['wood_diameter']}mm {event['wood_species']} (Quality {event['wood_quality']})")
            print(f"    Competitors: {len(event.get('handicap_results_all', []))}")
            if event.get('analysis_completed', False):
                print(f"    Analysis: COMPLETED")
            else:
                print(f"    Analysis: Not completed")
            print()

        print(f"{len(tournament_state['events']) + 1}. Return to main menu")

        choice = input("\nSelect event to analyze (or return to menu): ").strip()

        if not choice.isdigit():
            print("\n⚠ Invalid choice. Please enter a number.")
            input("\nPress Enter to continue...")
            continue

        choice_num = int(choice)

        if choice_num == len(tournament_state['events']) + 1:
            return

        if choice_num < 1 or choice_num > len(tournament_state['events']):
            print("\n⚠ Invalid choice.")
            input("\nPress Enter to continue...")
            continue

        # Analyze selected event
        event_index = choice_num - 1
        analyze_single_event(tournament_state['events'][event_index], event_index, tournament_state)


def approve_event_handicaps(tournament_state: Dict) -> None:
    """Event selection menu for handicap approval.

    Allows judges to approve handicaps event-by-event.
    Manual adjustments require full analysis to have been completed.

    Args:
        tournament_state: Multi-event tournament state dict
    """
    from woodchopping.ui.handicap_ui import judge_approval

    while True:
        print(f"\n{'╔' + '═' * 68 + '╗'}")
        title = "HANDICAP APPROVAL - EVENT SELECTION".center(68)
        print(f"{'║'}{title}{'║'}")
        print(f"{'╚' + '═' * 68 + '╝'}")

        # Validation: Check handicaps are calculated
        not_ready = [e for e in tournament_state.get('events', []) if e['status'] not in ['ready', 'scheduled', 'in_progress', 'completed']]
        if not_ready:
            print(f"\n⚠ ERROR: {len(not_ready)} event(s) don't have handicaps calculated:")
            for event in not_ready:
                print(f"  - {event['event_name']}: {event['status']}")
            print("\nPlease calculate handicaps first (Option 5).")
            input("\nPress Enter to continue...")
            return

        if not tournament_state.get('events'):
            print("\n⚠ No events in tournament.")
            input("\nPress Enter to continue...")
            return

        # Display events
        print(f"\n{'='*70}")
        print(f"  SELECT EVENT TO APPROVE")
        print(f"{'='*70}\n")

        for idx, event in enumerate(tournament_state['events'], 1):
            approved = all(r.get('approved_by') for r in event.get('handicap_results_all', []))
            analysis_done = event.get('analysis_completed', False)

            if approved:
                status_icon = "✓"
                status_text = "APPROVED"
            else:
                status_icon = " "
                status_text = "Not approved"

            print(f"{idx}. [{status_icon}] {event['event_name']}")
            print(f"    Wood: {event['wood_diameter']}mm {event['wood_species']} (Quality {event['wood_quality']})")
            print(f"    Competitors: {len(event.get('handicap_results_all', []))}")
            print(f"    Status: {status_text}")
            print(f"    Analysis: {'COMPLETED' if analysis_done else 'Not completed (manual adjustments unavailable)'}")
            print()

        print(f"{len(tournament_state['events']) + 1}. Approve ALL events at once")
        print(f"{len(tournament_state['events']) + 2}. Return to main menu")

        choice = input("\nSelect event to approve (or choose option): ").strip()

        if not choice.isdigit():
            print("\n⚠ Invalid choice. Please enter a number.")
            input("\nPress Enter to continue...")
            continue

        choice_num = int(choice)

        # Return to menu
        if choice_num == len(tournament_state['events']) + 2:
            return

        # Approve all events
        if choice_num == len(tournament_state['events']) + 1:
            print(f"\n{'='*70}")
            print(f"  APPROVE ALL EVENTS")
            print(f"{'='*70}")

            confirm = input("\nApprove handicaps for ALL events? (y/n): ").strip().lower()
            if confirm == 'y':
                initials, timestamp = judge_approval()

                if initials:
                    # Mark all events as approved
                    for event in tournament_state['events']:
                        if event['handicap_results_all']:
                            for result in event['handicap_results_all']:
                                result['approved_by'] = initials
                                result['approved_at'] = timestamp

                    print(f"\n✓ All handicaps approved by {initials} at {timestamp}")
                    # Auto-save
                    auto_save_multi_event(tournament_state)
                    print("✓ Tournament state auto-saved")
                else:
                    print("\n⚠ Approval cancelled")

            input("\nPress Enter to continue...")
            continue

        # Individual event approval
        if choice_num < 1 or choice_num > len(tournament_state['events']):
            print("\n⚠ Invalid choice.")
            input("\nPress Enter to continue...")
            continue

        event_index = choice_num - 1
        event = tournament_state['events'][event_index]

        # Show approval menu for this event
        print(f"\n{'╔' + '═' * 68 + '╗'}")
        title = f"APPROVE: {event['event_name']}".center(68)
        print(f"{'║'}{title}{'║'}")
        print(f"{'╚' + '═' * 68 + '╝'}")

        if not event['handicap_results_all']:
            print("\n⚠ No handicaps calculated for this event.")
            input("\nPress Enter to continue...")
            continue

        # Check if already approved
        already_approved = all(r.get('approved_by') for r in event['handicap_results_all'])
        if already_approved:
            approver = event['handicap_results_all'][0].get('approved_by', 'Unknown')
            approved_time = event['handicap_results_all'][0].get('approved_at', 'Unknown')
            print(f"\n✓ This event was already approved by {approver} at {approved_time}")

            reapprove = input("\nRe-approve these handicaps? (y/n): ").strip().lower()
            if reapprove != 'y':
                input("\nPress Enter to continue...")
                continue

        # Show handicaps
        print(f"\n{'='*70}")
        print(f"  HANDICAP MARKS")
        print(f"{'='*70}\n")

        sorted_results = sorted(event['handicap_results_all'], key=lambda x: x['mark'])

        print(f"{'Competitor':<35s} {'Predicted Time':<15s} {'Mark':<6s}")
        print(f"{'-'*70}")

        for result in sorted_results:
            name = result['name'][:34]
            time = f"{result['predicted_time']:.1f}s"
            mark = f"{result['mark']}"

            print(f"{name:<35s} {time:<15s} {mark:<6s}")

        print(f"{'-'*70}")
        print(f"Total competitors: {len(event['handicap_results_all'])}\n")

        # Championship events: simplified approval (no manual adjustments)
        if event.get('event_type') == 'championship':
            print(f"\n{'='*70}")
            print(f"  CHAMPIONSHIP EVENT APPROVAL")
            print(f"{'='*70}")
            print("\nChampionship Event - all competitors have Mark 3")
            print("No manual adjustments available (all marks are identical)")
            print("\n1. Approve marks as assigned")
            print("2. Cancel (return to event selection)")

            approval_choice = input("\nYour choice (1-2): ").strip()

            if approval_choice == '1':
                # Accept championship marks
                initials, timestamp = judge_approval()

                if initials:
                    # Mark this event as approved
                    for result in event['handicap_results_all']:
                        result['approved_by'] = initials
                        result['approved_at'] = timestamp

                    print(f"\n✓ Championship marks approved by {initials} at {timestamp}")
                    # Auto-save
                    auto_save_multi_event(tournament_state)
                    print("✓ Tournament state auto-saved")
                else:
                    print("\n⚠ Approval cancelled")

                input("\nPress Enter to continue...")
            else:
                # Cancel - return to event selection
                continue

            # Skip the rest of the approval flow for Championship events
            continue

        # Handicap events: full approval options
        # Approval options
        print(f"\n{'='*70}")
        print(f"  APPROVAL OPTIONS")
        print(f"{'='*70}")
        print("\n1. Accept handicaps as calculated")
        print("2. Manually adjust handicaps")
        print("3. Cancel (return to event selection)")

        approval_choice = input("\nYour choice (1-3): ").strip()

        if approval_choice == '1':
            # Accept handicaps
            initials, timestamp = judge_approval()

            if initials:
                # Mark this event as approved
                for result in event['handicap_results_all']:
                    result['approved_by'] = initials
                    result['approved_at'] = timestamp

                print(f"\n✓ Handicaps approved by {initials} at {timestamp}")
                # Auto-save
                auto_save_multi_event(tournament_state)
                print("✓ Tournament state auto-saved")
            else:
                print("\n⚠ Approval cancelled")

            input("\nPress Enter to continue...")

        elif approval_choice == '2':
            # Check if analysis was completed
            if not event.get('analysis_completed', False):
                print(f"\n{'='*70}")
                print(f"  ⚠ ANALYSIS REQUIRED FOR MANUAL ADJUSTMENTS")
                print(f"{'='*70}")
                print("\nManual adjustments require full analysis to be completed first.")
                print("This ensures you understand the handicap calculations and fairness metrics")
                print("before making changes.")
                print("\nPlease:")
                print("1. Return to main menu")
                print("2. Select Option 6 (Analyze Handicaps)")
                print("3. Run full analysis for this event (marks, Monte Carlo, AI analysis)")
                print("4. Mark analysis as complete")
                print("5. Return here to make manual adjustments")
                input("\nPress Enter to continue...")
                continue

            # Manual adjustment workflow
            print(f"\n{'='*70}")
            print(f"  MANUAL HANDICAP ADJUSTMENT")
            print(f"{'='*70}")
            print("\n⚠ This feature allows you to override calculated handicaps.")
            print("Use this ONLY if you have specific knowledge about:")
            print("  - Recent injuries or form changes")
            print("  - Equipment issues")
            print("  - Other factors not reflected in historical data")

            confirm = input("\nProceed with manual adjustments? (y/n): ").strip().lower()
            if confirm != 'y':
                print("\n⚠ Manual adjustment cancelled")
                input("\nPress Enter to continue...")
                continue

            # Allow manual mark adjustment
            print("\nEnter competitor name to adjust (or 'done' to finish):")
            while True:
                comp_name = input("\nCompetitor name: ").strip()

                if comp_name.lower() == 'done':
                    break

                # Find competitor
                matching = [r for r in event['handicap_results_all'] if comp_name.lower() in r['name'].lower()]

                if not matching:
                    print(f"\n⚠ No competitor found matching '{comp_name}'")
                    continue

                if len(matching) > 1:
                    print(f"\n⚠ Multiple matches found:")
                    for r in matching:
                        print(f"  - {r['name']}")
                    print("Please be more specific.")
                    continue

                result = matching[0]
                print(f"\nCompetitor: {result['name']}")
                print(f"Current mark: {result['mark']}")
                print(f"Predicted time: {result['predicted_time']:.1f}s")

                new_mark = input("\nEnter new mark (or blank to cancel): ").strip()

                if not new_mark:
                    print("⚠ No change made")
                    continue

                if not new_mark.isdigit() or int(new_mark) < 3:
                    print("⚠ Invalid mark. Must be >= 3.")
                    continue

                # A5: Prompt for reason
                print("\n" + "─" * 70)
                print("Please explain why you're adjusting this handicap (for audit trail):")
                reason = input("Reason: ").strip()
                while not reason:
                    print("⚠ Reason is required for adjustment tracking.")
                    reason = input("Reason: ").strip()

                old_mark = result['mark']
                result['mark'] = int(new_mark)
                result['manually_adjusted'] = True
                result['original_mark'] = old_mark
                result['adjustment_reason'] = reason  # A5: Store reason

                print(f"\n✓ Mark changed from {old_mark} to {new_mark}")
                print(f"  Reason: {reason}")

            # After adjustments, require approval
            print(f"\n{'='*70}")
            print(f"  APPROVE ADJUSTED HANDICAPS")
            print(f"{'='*70}")

            initials, timestamp = judge_approval()

            if initials:
                # A5: Log all manual adjustments to event state
                for result in event['handicap_results_all']:
                    if result.get('manually_adjusted'):
                        log_handicap_adjustment(
                            tournament_state=event,  # Log to event state (part of multi-event tournament)
                            competitor_name=result['name'],
                            original_mark=result.get('original_mark', result['mark']),
                            adjusted_mark=result['mark'],
                            reason=result.get('adjustment_reason', 'No reason provided'),
                            adjustment_type='manual'
                        )

                # Mark this event as approved
                for result in event['handicap_results_all']:
                    result['approved_by'] = initials
                    result['approved_at'] = timestamp

                print(f"\n✓ Adjusted handicaps approved by {initials} at {timestamp}")
                # Auto-save
                auto_save_multi_event(tournament_state)
                print("✓ Tournament state auto-saved")
            else:
                print("\n⚠ Approval cancelled - adjustments saved but not approved")

            input("\nPress Enter to continue...")

        # Choice 3 just returns to event selection (loop continues)


def view_wood_count(tournament_state: Dict) -> None:
    """Display wood requirements breakdown for tournament.

    Shows:
    - Per-event wood requirements (species, diameter, count, event type)
    - Grand total by species (with full species names)
    - Overall grand total blocks needed

    Args:
        tournament_state: Multi-event tournament state
    """
    from collections import defaultdict
    from woodchopping.data.excel_io import get_species_name_from_code

    print(f"\n{'╔' + '═' * 68 + '╗'}")
    title = f"WOOD COUNT: {tournament_state.get('tournament_name', 'Unknown')}".center(68)
    print(f"{'║'}{title}{'║'}")
    print(f"{'╚' + '═' * 68 + '╝'}")

    if not tournament_state.get('events'):
        print("\n⚠ No events added yet. Use 'Add Event to Tournament' to begin.")
        input("\nPress Enter to continue...")
        return

    # Collect wood requirements per event
    print(f"\n{'='*70}")
    print(f"  WOOD REQUIREMENTS BY EVENT")
    print(f"{'='*70}\n")
    print(f"{'Event':<30s} {'Species':<20s} {'Diameter':<10s} {'Blocks':<10s}")
    print(f"{'-'*70}")

    species_totals = defaultdict(int)
    size_species_totals = defaultdict(int)  # Track size/species combinations
    grand_total = 0

    for event in tournament_state['events']:
        # Get event type indicator
        event_type = event.get('event_type', 'handicap')
        type_indicator = 'CHMP' if event_type == 'championship' else 'HC'

        # Format event name with type indicator
        event_name_raw = event['event_name'][:24]
        event_name = f"{event_name_raw} ({type_indicator})"[:29]

        # Convert species code to full name
        species_code = event['wood_species']
        species_name = get_species_name_from_code(species_code)[:19]
        diameter = f"{int(event['wood_diameter'])}mm"
        blocks = event['capacity_info'].get('total_blocks', 0)

        print(f"{event_name:<30s} {species_name:<20s} {diameter:<10s} {blocks:<10d}")

        # Track totals by species code (for species total section)
        species_totals[species_code] += blocks

        # Track totals by size/species combination (diameter_mm, species_code)
        size_species_key = (int(event['wood_diameter']), species_code)
        size_species_totals[size_species_key] += blocks

        grand_total += blocks

    print(f"{'-'*70}")

    # Breakdown by size/species combination
    print(f"\n{'='*70}")
    print(f"  BREAKDOWN BY SIZE & SPECIES")
    print(f"{'='*70}\n")
    print(f"{'Size/Species':<50s} {'Blocks':<10s}")
    print(f"{'-'*70}")

    # Sort by diameter first, then by species name
    sorted_size_species = sorted(size_species_totals.items(), key=lambda x: (x[0][0], get_species_name_from_code(x[0][1])))

    for (diameter, species_code), blocks in sorted_size_species:
        species_name = get_species_name_from_code(species_code)
        size_species_label = f"{diameter}mm {species_name}"
        print(f"{size_species_label:<50s} {blocks:<10d}")

    print(f"{'-'*70}")

    # Grand total by species (with full names)
    print(f"\n{'='*70}")
    print(f"  GRAND TOTAL BY SPECIES")
    print(f"{'='*70}\n")
    print(f"{'Species':<40s} {'Total Blocks':<10s}")
    print(f"{'-'*70}")

    for species_code in sorted(species_totals.keys(), key=lambda x: get_species_name_from_code(x)):
        # Convert species code to full name
        species_name = get_species_name_from_code(species_code)
        print(f"{species_name:<40s} {species_totals[species_code]:<10d}")

    print(f"{'-'*70}")

    # Overall grand total
    print(f"\n{'='*70}")
    print(f"  OVERALL GRAND TOTAL")
    print(f"{'='*70}")
    print(f"\nTotal blocks needed: {grand_total}")
    print(f"{'='*70}")

    input("\nPress Enter to continue...")


def view_tournament_schedule(tournament_state: Dict) -> None:
    """Display complete tournament schedule for all events.

    Shows:
    - Event order and names
    - Wood configurations
    - Competitor counts
    - Format details (heats, semis, finals)
    - Total blocks needed per event
    - Overall tournament summary

    Args:
        tournament_state: Multi-event tournament state
    """
    print(f"\n{'╔' + '═' * 68 + '╗'}")
    title = f"TOURNAMENT SCHEDULE: {tournament_state.get('tournament_name', 'Unknown')}".center(68)
    print(f"{'║'}{title}{'║'}")
    print(f"{'╚' + '═' * 68 + '╝'}")

    print(f"\nDate: {tournament_state.get('tournament_date', 'Unknown')}")
    print(f"Total events: {tournament_state.get('total_events', 0)}")
    print(f"Events completed: {tournament_state.get('events_completed', 0)}")

    if not tournament_state.get('events'):
        print("\n⚠ No events added yet. Use 'Add Event to Tournament' to begin.")
        input("\nPress Enter to continue...")
        return

    # Display each event
    for event in tournament_state['events']:
        print(f"\n{'='*70}")
        print(f"  EVENT {event['event_order']}: {event['event_name']}")
        print(f"{'='*70}")
        print(f"Status: {event['status'].upper()}")
        print(f"Wood: {event['wood_diameter']}mm {event['wood_species']} (Quality {event['wood_quality']})")
        print(f"Event type: {event['event_code']}")
        print(f"Stands: {event['num_stands']}")
        print(f"Format: {event['format']}")
        print(f"Competitors: {len(event['all_competitors'])}")
        print(f"Handicaps: {'Calculated' if event['handicap_results_all'] else 'Not calculated'}")

        # Show capacity info
        capacity = event['capacity_info']
        print(f"\nFormat details:")
        print(f"  - Heats: {capacity.get('num_heats', 0)}")
        if capacity.get('num_semis', 0) > 0:
            print(f"  - Semi-finals: {capacity.get('num_semis', 0)}")
        if capacity.get('num_finals', 0) > 0:
            print(f"  - Finals: {capacity.get('num_finals', 0)}")
        print(f"  - Total blocks needed: {capacity.get('total_blocks', 0)}")

        # Show rounds if generated
        if event['rounds']:
            print(f"\nRounds generated: {len(event['rounds'])}")
            for round_obj in event['rounds']:
                status_icon = "✓" if round_obj['status'] == 'completed' else "○" if round_obj['status'] == 'pending' else "⚠"
                print(f"  {status_icon} {round_obj['round_name']}: {len(round_obj['competitors'])} competitors ({round_obj['status']})")

    # Overall summary
    print(f"\n{'='*70}")
    print(f"  TOURNAMENT SUMMARY")
    print(f"{'='*70}")

    total_competitors = sum(len(event['all_competitors']) for event in tournament_state['events'])
    total_blocks = sum(event['capacity_info'].get('total_blocks', 0) for event in tournament_state['events'])

    print(f"Total unique competitors: {total_competitors} (may include duplicates across events)")
    print(f"Total blocks needed: {total_blocks}")

    input("\nPress Enter to continue...")


def remove_event_from_tournament(tournament_state: Dict) -> Dict:
    """Remove an event from tournament with confirmation.

    Prompts judge to select event to remove, confirms deletion,
    and reorders remaining events.

    Args:
        tournament_state: Multi-event tournament state

    Returns:
        dict: Updated tournament_state
    """
    if not tournament_state.get('events'):
        print("\n⚠ No events to remove.")
        input("\nPress Enter to continue...")
        return tournament_state

    print(f"\n{'='*70}")
    print(f"  REMOVE EVENT FROM TOURNAMENT")
    print(f"{'='*70}")

    # Display events
    for i, event in enumerate(tournament_state['events'], 1):
        status_warning = ""
        if event['rounds']:
            completed = sum(1 for r in event['rounds'] if r['status'] == 'completed')
            total = len(event['rounds'])
            if completed > 0:
                status_warning = f" [⚠ HAS RESULTS: {completed}/{total} rounds complete]"

        print(f"  {i}) {event['event_name']} ({event['status']}){status_warning}")

    print(f"\n  0) Cancel")

    # Get selection
    try:
        choice = int(input("\nSelect event to remove (number): ").strip())
        if choice == 0:
            print("Cancelling...")
            return tournament_state

        if choice < 1 or choice > len(tournament_state['events']):
            print("⚠ Invalid selection.")
            input("\nPress Enter to continue...")
            return tournament_state

        event_to_remove = tournament_state['events'][choice - 1]

        # Confirm deletion
        print(f"\n⚠ WARNING: You are about to remove:")
        print(f"  Event: {event_to_remove['event_name']}")
        print(f"  Status: {event_to_remove['status']}")

        if event_to_remove['rounds']:
            completed = sum(1 for r in event_to_remove['rounds'] if r['status'] == 'completed')
            if completed > 0:
                print(f"  ⚠⚠ This event has {completed} completed round(s) with recorded results!")

        confirm = input("\nType 'DELETE' to confirm removal: ").strip()

        if confirm != 'DELETE':
            print("Cancelling...")
            return tournament_state

        # Remove event
        tournament_state['events'].pop(choice - 1)
        tournament_state['total_events'] -= 1

        # Reorder remaining events
        for i, event in enumerate(tournament_state['events'], 1):
            event['event_order'] = i
            event['event_id'] = f"event_{i}"

        print(f"\n✓ Event '{event_to_remove['event_name']}' removed successfully")
        print(f"✓ Remaining events reordered")

        # Auto-save
        auto_save_multi_event(tournament_state)

    except ValueError:
        print("⚠ Invalid input.")

    input("\nPress Enter to continue...")
    return tournament_state


def generate_complete_day_schedule(tournament_state: Dict) -> Dict:
    """Generate initial heats for ALL events in tournament.

    For each event:
    - Generate heats using snake draft distribution
    - Update event status to 'scheduled'
    - Display heat assignments

    IMPORTANT: All events must have handicaps calculated first (status='ready').
    Use 'Calculate Handicaps for All Events' before generating schedule.

    Args:
        tournament_state: Multi-event tournament state

    Returns:
        dict: Updated tournament_state with heats generated for all events
    """
    print(f"\n{'╔' + '═' * 68 + '╗'}")
    title = f"GENERATE COMPLETE DAY SCHEDULE".center(68)
    print(f"{'║'}{title}{'║'}")
    print(f"{'╚' + '═' * 68 + '╝'}")

    # Validation: All events must have handicaps calculated (status='ready')
    not_ready = [e for e in tournament_state.get('events', []) if e['status'] != 'ready']
    if not_ready:
        print(f"\n⚠ ERROR: {len(not_ready)} event(s) do not have handicaps calculated:")
        for event in not_ready:
            status_msg = "handicaps NOT calculated" if event['status'] == 'configured' else event['status']
            print(f"  - {event['event_name']}: {status_msg}")
        print("\n⚠ Please calculate handicaps for ALL events first (use Option 5).")
        print("   Workflow: Configure events → Calculate handicaps → Generate schedule")
        input("\nPress Enter to continue...")
        return tournament_state

    if not tournament_state.get('events'):
        print("\n⚠ No events to generate schedule for.")
        input("\nPress Enter to continue...")
        return tournament_state

    # Generate heats for each event
    for event in tournament_state['events']:
        # Get event type indicator
        event_type = event.get('event_type', 'handicap')
        type_indicator = '(CHMP)' if event_type == 'championship' else '(HCP)'
        event_display_name = f"{event['event_name']} {type_indicator}"

        # Skip if already generated
        if event['rounds']:
            print(f"\n○ {event_display_name}: Heats already generated (skipping)")
            continue

        # Display prominent event banner
        print("\n" + "╔" + "═" * 68 + "╗")
        print("║" + event_display_name.center(68) + "║")
        print("╚" + "═" * 68 + "╝")

        num_competitors = len(event['all_competitors'])
        num_stands = event['num_stands']

        # Check format
        if event['format'] == 'single_heat':
            # Single heat mode
            heats = [{
                'round_name': 'Heat 1',
                'round_type': 'heat',
                'round_number': 1,
                'competitors': event['all_competitors'],
                'competitors_df': event['all_competitors_df'],
                'handicap_results': event['handicap_results_all'],
                'num_to_advance': 0,
                'status': 'pending',
                'actual_results': {},
                'finish_order': {},
                'advancers': []
            }]

            # Display detailed stand assignments
            print(f"\n{'─'*70}")
            print(f"  Heat 1 - Single Heat Mode")
            print(f"{'─'*70}")

            # Display competitors with stand numbers and marks
            for stand_num, comp_name in enumerate(heats[0]['competitors'], 1):
                # Find handicap mark for this competitor
                mark = next((c['mark'] for c in heats[0]['handicap_results'] if c['name'] == comp_name), '?')
                mark_str = str(mark) if isinstance(mark, int) else mark

                # Label backmarker and frontmarker (only for handicap events)
                if event_type == 'championship':
                    label = ""
                elif stand_num == 1:
                    label = " ← Backmarker"
                elif stand_num == len(heats[0]['competitors']):
                    label = " ← Frontmarker"
                else:
                    label = ""

                print(f"  Stand {stand_num:2d}: {comp_name:35s} Mark {mark_str:3s}{label}")

            print(f"{'─'*70}")

        else:
            # Multi-round tournament mode
            # Use optimal stands_per_heat from capacity calculation (may be less than total num_stands)
            from math import ceil
            capacity_info = event.get('capacity_info', {})
            stands_per_heat = capacity_info.get('stands_per_heat', num_stands)  # Fallback to num_stands for old saved states
            num_heats = capacity_info.get('num_heats', ceil(num_competitors / num_stands))

            heats = distribute_competitors_into_heats(
                event['all_competitors_df'],
                event['handicap_results_all'],
                stands_per_heat,  # Use optimal stands per heat, not total available stands
                num_heats
            )

            # Display detailed heat assignments with stand numbers
            for heat in heats:
                print(f"\n{'─'*70}")
                print(f"  {heat['round_name']} - {len(heat['competitors'])} competitors (top {heat['num_to_advance']} advance)")
                print(f"{'─'*70}")

                # Competitors are already sorted by snake draft (backmarker first, frontmarker last)
                for stand_num, comp_name in enumerate(heat['competitors'], 1):
                    # Find handicap mark for this competitor
                    mark = next((c['mark'] for c in heat['handicap_results'] if c['name'] == comp_name), '?')
                    mark_str = str(mark) if isinstance(mark, int) else mark

                    # Label backmarker and frontmarker (only for handicap events)
                    if event_type == 'championship':
                        label = ""
                    elif stand_num == 1:
                        label = " ← Backmarker"
                    elif stand_num == len(heat['competitors']):
                        label = " ← Frontmarker"
                    else:
                        label = ""

                    print(f"  Stand {stand_num:2d}: {comp_name:35s} Mark {mark_str:3s}{label}")

                print(f"{'─'*70}")

        # Update event
        event['rounds'] = heats
        event['status'] = 'scheduled'  # Heats generated, ready for competition

        print(f"✓ {event_display_name}: Heats generated successfully")

    print(f"\n{'='*70}")
    print(f"  ✓ COMPLETE DAY SCHEDULE GENERATED")
    print(f"{'='*70}")
    print(f"All events ready for competition!")

    # Auto-save
    auto_save_multi_event(tournament_state)

    input("\nPress Enter to continue...")
    return tournament_state


def view_all_handicaps_summary(tournament_state: Dict) -> None:
    """Display handicap marks for all events in tabular format.

    Shows one section per event with:
    - Competitor name
    - Predicted time
    - Handicap mark
    - Method used

    Args:
        tournament_state: Multi-event tournament state
    """
    print(f"\n{'╔' + '═' * 68 + '╗'}")
    title = f"HANDICAP MARKS - ALL EVENTS".center(68)
    print(f"{'║'}{title}{'║'}")
    print(f"{'╚' + '═' * 68 + '╝'}")

    if not tournament_state.get('events'):
        print("\n⚠ No events in tournament.")
        input("\nPress Enter to continue...")
        return

    for event in tournament_state['events']:
        print(f"\n{'='*70}")
        print(f"  EVENT {event['event_order']}: {event['event_name']}")
        print(f"{'='*70}")
        print(f"Wood: {event['wood_diameter']}mm {event['wood_species']} (Quality {event['wood_quality']})")
        print(f"Event type: {event['event_code']}")
        print(f"{'='*70}")

        if not event['handicap_results_all']:
            print("\n⚠ No handicaps calculated for this event.")
            continue

        # Display handicaps sorted by mark
        sorted_results = sorted(event['handicap_results_all'], key=lambda x: x['mark'])

        print(f"\n{'Competitor':<35s} {'Time':<10s} {'Mark':<6s} {'Method'}")
        print(f"{'-'*70}")

        for result in sorted_results:
            name = result['name'][:34]
            time = f"{result['predicted_time']:.1f}s"
            mark = f"{result['mark']}"
            method = result.get('method_used', 'Unknown')[:15]

            print(f"{name:<35s} {time:<10s} {mark:<6s} {method}")

        print(f"{'-'*70}")
        print(f"Total competitors: {len(event['handicap_results_all'])}")

    input("\nPress Enter to continue...")


def get_next_incomplete_round(tournament_state: Dict) -> Tuple[Optional[int], Optional[Dict], Optional[Dict]]:
    """Find next round that needs results entry.

    Searches events in order for the first pending or in-progress round.

    Args:
        tournament_state: Multi-event tournament state

    Returns:
        tuple: (event_index, event_obj, round_obj) or (None, None, None) if all complete
    """
    for event_idx, event in enumerate(tournament_state.get('events', [])):
        for round_obj in event.get('rounds', []):
            if round_obj['status'] in ['pending', 'in_progress']:
                return (event_idx, event, round_obj)

    return (None, None, None)


def display_event_progress(event_obj: Dict, current_round: Dict) -> None:
    """Display current event context and progress.

    Shows:
    - Event name and order
    - Round status (heats, semis, finals)
    - Current round being worked on
    - Next action required

    Args:
        event_obj: Event object
        current_round: Current round object being worked on
    """
    print(f"\n{'╔' + '═' * 68 + '╗'}")
    title = f"EVENT {event_obj['event_order']}: {event_obj['event_name']}".center(68)
    print(f"{'║'}{title}{'║'}")
    print(f"{'╚' + '═' * 68 + '╝'}")

    # Count rounds by type
    heats = [r for r in event_obj['rounds'] if r['round_type'] == 'heat']
    semis = [r for r in event_obj['rounds'] if r['round_type'] == 'semi']
    finals = [r for r in event_obj['rounds'] if r['round_type'] == 'final']

    # Display status
    def get_status_display(rounds):
        if not rounds:
            return "N/A"
        completed = sum(1 for r in rounds if r['status'] == 'completed')
        total = len(rounds)
        if completed == total:
            return f"✓ Complete ({total}/{total})"
        elif completed > 0:
            return f"⚠ In Progress ({completed}/{total})"
        else:
            return f"○ Pending (0/{total})"

    print(f"\nRound Status:")
    print(f"  Heats: {get_status_display(heats)}")
    if semis:
        print(f"  Semi-finals: {get_status_display(semis)}")
    if finals:
        print(f"  Finals: {get_status_display(finals)}")

    print(f"\nCurrent Round: {current_round['round_name']}")
    print(f"Competitors: {len(current_round['competitors'])}")
    print(f"Status: {current_round['status']}")


def sequential_results_workflow(tournament_state: Dict, wood_selection: Dict, heat_assignment_df: pd.DataFrame) -> Dict:
    """Sequential results entry workflow for all events in tournament.

    Guides judge through recording results for all rounds across all events.
    Navigation:
    - Auto-advance to next incomplete round
    - Show current event context
    - Allow jumping to specific event
    - Track overall progress

    Args:
        tournament_state: Multi-event tournament state
        wood_selection: Wood selection dict (legacy, not used in multi-event)
        heat_assignment_df: Heat assignment DataFrame (legacy, not used in multi-event)

    Returns:
        dict: Updated tournament_state with recorded results
    """
    print(f"\n{'╔' + '═' * 68 + '╗'}")
    title = f"SEQUENTIAL RESULTS ENTRY".center(68)
    print(f"{'║'}{title}{'║'}")
    print(f"{'╚' + '═' * 68 + '╝'}")

    # Validation: All events must have heats generated
    not_ready = [e for e in tournament_state.get('events', []) if e['status'] == 'pending' or e['status'] == 'configured']
    if not_ready:
        print(f"\n⚠ ERROR: {len(not_ready)} event(s) not ready for results:")
        for event in not_ready:
            print(f"  - {event['event_name']}: {event['status']}")
        print("\nPlease generate schedule first (Option 5).")
        input("\nPress Enter to continue...")
        return tournament_state

    while True:
        # Find next incomplete round
        event_idx, event_obj, round_obj = get_next_incomplete_round(tournament_state)

        if event_idx is None:
            print(f"\n{'='*70}")
            print(f"  ✓ ALL EVENTS COMPLETED!")
            print(f"{'='*70}")
            print(f"All rounds across all events have been completed.")
            print(f"You can now generate the final tournament summary (Option 8).")
            input("\nPress Enter to continue...")
            break

        # Display current context
        display_event_progress(event_obj, round_obj)

        # Menu
        print(f"\n{'='*70}")
        print(f"  1) Record Results for {round_obj['round_name']}")
        print(f"  2) Jump to Specific Event")
        print(f"  3) View Tournament Status")
        print(f"  4) Return to Tournament Menu")
        print(f"{'='*70}")

        choice = input("\nYour choice: ").strip()

        if choice == '1':
            # Record results for current round
            print(f"\n{'='*70}")
            print(f"  RECORDING RESULTS: {round_obj['round_name']}")
            print(f"{'='*70}")

            # Prepare wood_selection for this event
            event_wood = {
                'species': event_obj['wood_species'],
                'size_mm': event_obj['wood_diameter'],
                'quality': event_obj['wood_quality'],
                'event': event_obj['event_code']
            }

            # Record results using existing function
            append_results_to_excel(
                heat_assignment_df,  # Legacy param (not used)
                event_wood,
                round_object=round_obj,
                tournament_state=None,  # Pass None to avoid single-event logic
                event_name=event_obj['event_name']  # Pass event name for HeatID
            )

            # Mark round in progress
            round_obj['status'] = 'in_progress'

            # Check if this is a final round
            is_final = round_obj['round_type'] == 'final'

            # Select advancers (if not final)
            if not is_final and event_obj['format'] != 'single_heat':
                advancers = select_heat_advancers(round_obj)
                print(f"\n✓ {round_obj['round_name']} completed")
                print(f"✓ Advancers: {', '.join(advancers)}")

                # Check if we need to generate next round
                current_round_type = round_obj['round_type']

                # Determine if more rounds needed
                all_heats = [r for r in event_obj['rounds'] if r['round_type'] == current_round_type]
                all_heats_complete = all(r['status'] == 'completed' for r in all_heats)

                if all_heats_complete:
                    # All heats/semis of this type complete - generate next round
                    all_advancers = []
                    for heat in all_heats:
                        all_advancers.extend(heat.get('advancers', []))

                    # Determine next round type
                    if current_round_type == 'heat':
                        if event_obj['format'] == 'heats_to_finals':
                            next_type = 'final'
                        else:  # heats_to_semis_to_finals
                            next_type = 'semi'
                    elif current_round_type == 'semi':
                        next_type = 'final'
                    else:
                        next_type = None

                    if next_type:
                        print(f"\n✓ All {current_round_type}s complete")
                        print(f"✓ Generating {next_type} round...")

                        # Generate next round using existing function
                        next_rounds = generate_next_round(
                            {'rounds': event_obj['rounds']},  # Wrap in dict for compatibility
                            all_advancers,
                            next_type,
                            is_championship=(event_obj.get('event_type') == 'championship')
                        )

                        # Add to event rounds
                        event_obj['rounds'].extend(next_rounds)
                        print(f"✓ {len(next_rounds)} {next_type} round(s) generated")

            else:
                # Final round or single heat - mark event as complete
                round_obj['status'] = 'completed'
                print(f"\n✓ {round_obj['round_name']} completed")

                if is_final:
                    event_obj['status'] = 'completed'
                    tournament_state['events_completed'] += 1
                    print(f"✓ {event_obj['event_name']} COMPLETE!")

                    # Extract placements
                    event_obj['final_results'] = extract_event_placements(event_obj)

            # Update event status
            if event_obj['status'] != 'completed':
                event_obj['status'] = 'in_progress'

            # Auto-save
            auto_save_multi_event(tournament_state)

            input("\nPress Enter to continue to next round...")

        elif choice == '2':
            # Jump to specific event
            print(f"\n{'='*70}")
            print(f"  SELECT EVENT")
            print(f"{'='*70}")

            for i, event in enumerate(tournament_state['events'], 1):
                status = event['status']
                print(f"  {i}) {event['event_name']} ({status})")

            print(f"\n  0) Cancel")

            try:
                event_choice = int(input("\nSelect event (number): ").strip())
                if event_choice == 0:
                    continue

                if 1 <= event_choice <= len(tournament_state['events']):
                    tournament_state['current_event_index'] = event_choice - 1
                    print(f"\n✓ Jumped to {tournament_state['events'][event_choice - 1]['event_name']}")
                else:
                    print("⚠ Invalid selection")
            except ValueError:
                print("⚠ Invalid input")

            input("\nPress Enter to continue...")

        elif choice == '3':
            # View tournament status
            view_tournament_schedule(tournament_state)

        elif choice == '4' or choice == '':
            break

        else:
            print("⚠ Invalid selection")

    return tournament_state


def extract_event_placements(event_obj: Dict) -> Dict:
    """Extract top 3 placements from completed event.

    Finds final round and extracts finish order to determine
    1st, 2nd, and 3rd place finishers.

    Args:
        event_obj: Event object with completed finals

    Returns:
        dict: {
            'first_place': name,
            'second_place': name,
            'third_place': name,
            'all_placements': {name: position, ...}
        }
    """
    # Find final round
    final_rounds = [r for r in event_obj.get('rounds', []) if r['round_type'] == 'final']

    if not final_rounds:
        return {
            'first_place': None,
            'second_place': None,
            'third_place': None,
            'all_placements': {}
        }

    final_round = final_rounds[0]
    finish_order = final_round.get('finish_order', {})

    if not finish_order:
        return {
            'first_place': None,
            'second_place': None,
            'third_place': None,
            'all_placements': {}
        }

    # Sort by position (1st, 2nd, 3rd...)
    sorted_placements = sorted(finish_order.items(), key=lambda x: x[1])

    return {
        'first_place': sorted_placements[0][0] if len(sorted_placements) > 0 else None,
        'second_place': sorted_placements[1][0] if len(sorted_placements) > 1 else None,
        'third_place': sorted_placements[2][0] if len(sorted_placements) > 2 else None,
        'all_placements': finish_order
    }


def generate_tournament_summary(tournament_state: Dict) -> None:
    """Generate final tournament summary with top 3 in each event.

    Displays text-based summary showing:
    - Tournament name and date
    - For each event: event name and top 3 finishers with times
    - Tournament statistics

    Args:
        tournament_state: Multi-event tournament state
    """
    print(f"\n{'╔' + '═' * 68 + '╗'}")
    title = f"FINAL TOURNAMENT SUMMARY".center(68)
    print(f"{'║'}{title}{'║'}")
    print(f"{'╚' + '═' * 68 + '╝'}")

    print(f"\nTournament: {tournament_state.get('tournament_name', 'Unknown')}")
    print(f"Date: {tournament_state.get('tournament_date', 'Unknown')}")
    print(f"Total Events: {tournament_state.get('total_events', 0)}")

    # Validation: All events must be completed
    incomplete = [e for e in tournament_state.get('events', []) if e['status'] != 'completed']
    if incomplete:
        print(f"\n⚠ WARNING: {len(incomplete)} event(s) not yet completed:")
        for event in incomplete:
            print(f"  - {event['event_name']}: {event['status']}")
        print("\nShowing results for completed events only.")

    # Display each event's results
    for event in tournament_state.get('events', []):
        print(f"\n{'='*70}")
        print(f"  EVENT {event['event_order']}: {event['event_name']}")
        print(f"{'='*70}")
        print(f"Wood: {event['wood_diameter']}mm {event['wood_species']} (Quality {event['wood_quality']})")
        print(f"Event type: {event['event_code']}")

        if event['status'] != 'completed':
            print(f"\n⚠ Event not completed - no final results available")
            continue

        # Get placements
        results = event.get('final_results', {})

        if not results.get('first_place'):
            print(f"\n⚠ No final results recorded")
            continue

        # Find final round to get times
        final_round = [r for r in event['rounds'] if r['round_type'] == 'final'][0]
        actual_results = final_round.get('actual_results', {})

        print(f"\n{'─'*70}")

        # Display top 3
        if results['first_place']:
            time = actual_results.get(results['first_place'], 'N/A')
            time_str = f"{time:.2f}s" if isinstance(time, (int, float)) else time
            print(f"  🥇 1st Place: {results['first_place']} ({time_str})")

        if results['second_place']:
            time = actual_results.get(results['second_place'], 'N/A')
            time_str = f"{time:.2f}s" if isinstance(time, (int, float)) else time
            print(f"  🥈 2nd Place: {results['second_place']} ({time_str})")

        if results['third_place']:
            time = actual_results.get(results['third_place'], 'N/A')
            time_str = f"{time:.2f}s" if isinstance(time, (int, float)) else time
            print(f"  🥉 3rd Place: {results['third_place']} ({time_str})")

        # Display payouts if configured (NEW V5.0)
        payout_config = event.get('payout_config')
        if payout_config and payout_config.get('enabled'):
            print(f"\n  PAYOUTS:")
            all_placements = results.get('all_placements', {})
            num_places = payout_config.get('num_places', 0)
            payouts = payout_config.get('payouts', {})

            # Display payouts for top 3 (or fewer if fewer paid places)
            for position in range(1, min(4, num_places + 1)):
                if position == 1 and results['first_place']:
                    name = results['first_place']
                elif position == 2 and results['second_place']:
                    name = results['second_place']
                elif position == 3 and results['third_place']:
                    name = results['third_place']
                else:
                    continue

                payout = payouts.get(position, 0)
                from woodchopping.ui.payout_ui import _get_ordinal
                print(f"    {_get_ordinal(position):6s}: ${payout:,.2f}")

            if num_places > 3:
                print(f"    ... plus {num_places - 3} more paid places")

            print(f"\n  Event Purse: ${payout_config['total_purse']:,.2f}")

    # Tournament statistics
    print(f"\n{'='*70}")
    print(f"  TOURNAMENT STATISTICS")
    print(f"{'='*70}")

    completed_events = [e for e in tournament_state['events'] if e['status'] == 'completed']
    total_competitors = sum(len(e['all_competitors']) for e in tournament_state['events'])
    total_rounds = sum(len(e['rounds']) for e in tournament_state['events'])

    print(f"Events completed: {len(completed_events)}/{tournament_state.get('total_events', 0)}")
    print(f"Total competitors: {total_competitors} (may include duplicates across events)")
    print(f"Total rounds run: {total_rounds}")

    # Tournament Earnings Summary (NEW V5.0)
    from woodchopping.ui.payout_ui import calculate_total_earnings, display_tournament_earnings_summary

    competitor_earnings = calculate_total_earnings(tournament_state)

    if competitor_earnings:
        print(f"\n{'='*70}")
        display_tournament_earnings_summary(tournament_state, competitor_earnings)

    print(f"\n{'='*70}")

    input("\nPress Enter to continue...")
