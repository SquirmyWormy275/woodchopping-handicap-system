"""Tournament management UI functions.

This module handles multi-round tournament operations including:
- Tournament scenario calculation
- Heat distribution
- Advancer selection
- Round generation
- Tournament state management
"""

import json
from math import ceil
from typing import Dict, List, Optional
import pandas as pd


def calculate_tournament_scenarios(num_stands: int, tentative_competitors: int) -> Dict:
    """Calculate three tournament format scenarios based on stands and competitor count.

    Args:
        num_stands: Available chopping stands
        tentative_competitors: Judge's estimate of competitor count

    Returns:
        dict: Contains all scenario options with format:
            {
                'single_heat': {scenario details},
                'heats_to_finals': {scenario details},
                'heats_to_semis_to_finals': {scenario details}
            }
    """

    # SCENARIO 0: Single Heat (Training/Testing)
    # Strategy: One heat only, perfect for practice/testing/small casual events
    scenario_0 = {
        'max_competitors': num_stands,
        'num_heats': 1,
        'num_semis': 0,
        'num_finals': 0,
        'advancers_per_heat': 0,  # No one advances, it's just one heat
        'total_blocks': num_stands,
        'description': (
            f"Single heat with up to {num_stands} competitors\n"
            f"  → Perfect for training, testing, or casual events\n"
            f"  → Results can still be saved to build historical data\n"
            f"  → No advancement rounds"
        )
    }

    # Calculate minimum heats needed
    min_heats = ceil(tentative_competitors / num_stands)

    # SCENARIO 1: Heats → Finals
    # Strategy: Use minimum heats, take top N from each to fill finals
    num_heats_s1 = max(min_heats, 2)  # At least 2 heats for a tournament
    max_competitors_s1 = num_heats_s1 * num_stands
    advancers_per_heat_s1 = num_stands // num_heats_s1

    # Ensure we can fill a final
    if advancers_per_heat_s1 * num_heats_s1 < num_stands:
        advancers_per_heat_s1 += 1

    total_blocks_s1 = tentative_competitors + num_stands

    scenario_1 = {
        'max_competitors': max_competitors_s1,
        'num_heats': num_heats_s1,
        'num_semis': 0,
        'num_finals': 1,
        'advancers_per_heat': advancers_per_heat_s1,
        'total_blocks': total_blocks_s1,
        'description': (
            f"{num_heats_s1} heats of {num_stands} (max {max_competitors_s1} competitors)\n"
            f"  → Top {advancers_per_heat_s1} from each heat advance\n"
            f"  → {num_stands}-person Final"
        )
    }

    # SCENARIO 2: Heats → Semis → Finals
    # Strategy: More heats to create semi-final round
    num_heats_s2 = max(min_heats + 2, 4)  # Add 2 more heats for semi tier
    max_competitors_s2 = num_heats_s2 * num_stands

    # Calculate semi-finals
    num_semis = 2
    semi_total = num_semis * num_stands
    advancers_per_heat_s2 = ceil(semi_total / num_heats_s2)
    advancers_per_semi = num_stands // num_semis

    total_blocks_s2 = tentative_competitors + semi_total + num_stands

    scenario_2 = {
        'max_competitors': max_competitors_s2,
        'num_heats': num_heats_s2,
        'num_semis': num_semis,
        'num_finals': 1,
        'advancers_per_heat': advancers_per_heat_s2,
        'advancers_per_semi': advancers_per_semi,
        'semi_total': semi_total,
        'total_blocks': total_blocks_s2,
        'description': (
            f"{num_heats_s2} heats of {num_stands} (max {max_competitors_s2} competitors)\n"
            f"  → Top {advancers_per_heat_s2} from each heat ({semi_total} total)\n"
            f"  → {num_semis} semi-finals of {num_stands}\n"
            f"  → Top {advancers_per_semi} from each semi\n"
            f"  → {num_stands}-person Final"
        )
    }

    return {
        'single_heat': scenario_0,
        'heats_to_finals': scenario_1,
        'heats_to_semis_to_finals': scenario_2
    }


def distribute_competitors_into_heats(all_competitors_df: pd.DataFrame,
                                     handicap_results: List[Dict],
                                     num_stands: int,
                                     num_heats: int) -> List[Dict]:
    """Distribute competitors into balanced heats using snake draft pattern.

    This algorithm ensures fair distribution of skill levels across all heats by:
    1. Sorting competitors by handicap mark (highest = front markers first)
    2. Using snake draft: forward then backward distribution pattern
    3. Automatically handling partial heats with appropriate advancement rules

    Args:
        all_competitors_df: All competitors in tournament
        handicap_results: Results from calculate_ai_enhanced_handicaps()
                         Format: [{'name': ..., 'mark': ..., 'predicted_time': ...}, ...]
        num_stands: Competitors per heat (heat capacity)
        num_heats: Number of heats to create

    Returns:
        list: List of round_object dictionaries, one for each heat
    """

    # Sort competitors by mark (descending: highest mark first = front markers)
    sorted_competitors = sorted(handicap_results, key=lambda x: x['mark'], reverse=True)

    # Initialize empty heats
    heats = []
    for i in range(num_heats):
        heats.append({
            'round_type': 'heat',
            'round_number': i + 1,
            'round_name': f'Heat {i + 1}',
            'competitors': [],
            'competitors_df': pd.DataFrame(),
            'handicap_results': [],
            'actual_results': {},
            'advancers': [],
            'num_to_advance': 2,  # Default, will be adjusted for partial heats
            'status': 'pending'
        })

    # Snake draft distribution
    heat_index = 0
    direction = 1  # 1 = forward, -1 = backward

    for comp in sorted_competitors:
        heats[heat_index]['competitors'].append(comp['name'])
        heats[heat_index]['handicap_results'].append(comp)

        # Move to next heat
        heat_index += direction

        # Reverse direction at ends (snake pattern)
        if heat_index >= num_heats:
            heat_index = num_heats - 1
            direction = -1
        elif heat_index < 0:
            heat_index = 0
            direction = 1

    # Build competitor DataFrames and set advancement rules for each heat
    for heat in heats:
        # Get DataFrame subset for competitors in this heat
        heat['competitors_df'] = all_competitors_df[
            all_competitors_df['competitor_name'].isin(heat['competitors'])
        ].copy()

        # Determine advancement rules based on heat size
        heat_size = len(heat['competitors'])
        heat_capacity = num_stands
        fill_percentage = heat_size / heat_capacity

        # If heat is ≤50% full, only top 1 advances (fairness for partial heats)
        if fill_percentage <= 0.5:
            heat['num_to_advance'] = 1
        else:
            heat['num_to_advance'] = 2

    return heats


def select_heat_advancers(round_object: Dict) -> List[str]:
    """Interactive function for judge to select advancing competitors from a completed heat.

    Args:
        round_object: Round object for completed heat

    Returns:
        list: Names of advancing competitors
    """

    print(f"\n{'='*70}")
    print(f"  {round_object['round_name'].upper()} RESULTS & ADVANCERS")
    print(f"{'='*70}")

    # Use FINISH ORDER to determine results (critical for handicap racing)
    if round_object.get('finish_order'):
        # Sort by finish order (1st, 2nd, 3rd, etc.)
        sorted_by_finish = sorted(
            round_object['finish_order'].items(),
            key=lambda x: x[1]  # Sort by finish position
        )

        # Display results table with finish position, cutting time, and advancement status
        print("\n╔════════════════════════════════════════════════════════════════════╗")
        print(f"║           🏆 {round_object['round_name'].upper()} - FINAL RESULTS 🏆            ║")
        print("╠════════════════════════════════════════════════════════════════════╣")
        print("║  Finish Position based on REAL-TIME completion (handicap included) ║")
        print("╠════════════════════════════════════════════════════════════════════╣")

        advancers = []  # Auto-select based on finish order

        for name, finish_pos in sorted_by_finish:
            cutting_time = round_object['actual_results'].get(name, 0)

            # Determine medal and advancement
            medal = ""
            advance = ""
            if finish_pos == 1:
                medal = "🥇"
            elif finish_pos == 2:
                medal = "🥈"
            elif finish_pos == 3:
                medal = "🥉"
            else:
                medal = "  "

            # Auto-select top finishers for advancement
            if finish_pos <= round_object['num_to_advance']:
                advance = "✓ ADVANCES"
                advancers.append(name)

            # Format line: Position | Name | Cutting Time | Status
            pos_part = f"{medal} {finish_pos:2d}"
            name_part = f"{name[:35]:<35}"
            time_part = f"{cutting_time:6.2f}s"
            status_part = f"{advance:^12}"

            print(f"║  {pos_part} │ {name_part} │ {time_part} │ {status_part} ║")

        print("╚════════════════════════════════════════════════════════════════════╝\n")

        # Display advancers summary
        print(f"✓ Top {round_object['num_to_advance']} finisher(s) automatically advance:")
        for name in advancers:
            finish_pos = round_object['finish_order'][name]
            print(f"  {finish_pos}. {name}")

        # Allow judge to override if needed
        override = input("\nAccept these advancers? (y/n to manually select): ").strip().lower()

        if override == 'y' or override == '':
            # Accept auto-selected advancers
            round_object['advancers'] = advancers
            round_object['status'] = 'completed'
            return advancers
        else:
            # Manual override - let judge pick
            print("\nManual selection mode:")
            advancers = []
            while len(advancers) < round_object['num_to_advance']:
                try:
                    for i, (name, pos) in enumerate(sorted_by_finish, 1):
                        print(f"  {i}) {name} (finished {pos})")

                    choice = input(f"\nSelect competitor {len(advancers)+1} of {round_object['num_to_advance']} (number): ").strip()
                    idx = int(choice) - 1

                    if 0 <= idx < len(sorted_by_finish):
                        selected = sorted_by_finish[idx][0]
                        if selected not in advancers:
                            advancers.append(selected)
                            print(f"  ✓ {selected} selected")
                        else:
                            print("  Already selected. Choose another competitor.")
                    else:
                        print(f"  Invalid number. Enter 1-{len(sorted_by_finish)}")

                except ValueError:
                    print("  Please enter a valid number.")

    else:
        # Fallback: No finish order recorded (legacy mode - use raw times)
        print("⚠ WARNING: No finish order recorded. Using raw cutting times (may be inaccurate for handicap).\n")

        if round_object['actual_results']:
            sorted_results = sorted(
                round_object['actual_results'].items(),
                key=lambda x: x[1]  # Sort by raw time
            )

            print("Competitors (sorted by cutting time):")
            for i, (name, time) in enumerate(sorted_results, 1):
                print(f"  {i}) {name:30s} {time:.2f}s")
        else:
            print("Competitors in heat:")
            for i, name in enumerate(round_object['competitors'], 1):
                print(f"  {i}) {name}")

        # Manual selection
        advancers = []
        while len(advancers) < round_object['num_to_advance']:
            try:
                choice = input(f"\nSelect competitor {len(advancers)+1} of {round_object['num_to_advance']} (number): ").strip()

                if choice == '':
                    print("Selection cannot be blank. Please enter a number.")
                    continue

                idx = int(choice) - 1

                if 0 <= idx < len(round_object['competitors']):
                    selected = round_object['competitors'][idx]
                    if selected not in advancers:
                        advancers.append(selected)
                        print(f"  ✓ {selected} selected")
                    else:
                        print("  Already selected. Choose another competitor.")
                else:
                    print(f"  Invalid number. Enter 1-{len(round_object['competitors'])}")

            except ValueError:
                print("  Please enter a valid number.")

    # Update round object
    round_object['advancers'] = advancers
    round_object['status'] = 'completed'

    return advancers


def generate_next_round(tournament_state: Dict, all_advancers: List[str], next_round_type: str) -> List[Dict]:
    """Generate semi-final or final rounds from advancing competitors.

    Args:
        tournament_state: Global tournament state
        all_advancers: All competitors advancing from previous round
        next_round_type: 'semi' or 'final'

    Returns:
        list: List of round_object dictionaries for next stage
    """

    # Extract handicap results for advancers only
    all_results = []
    for round_obj in tournament_state['rounds']:
        if 'handicap_results' in round_obj and round_obj['handicap_results']:
            all_results.extend(round_obj['handicap_results'])

    # Filter to just advancers
    advancer_results = [r for r in all_results if r['name'] in all_advancers]

    # Determine number of heats for next round
    num_stands = tournament_state['num_stands']

    if next_round_type == 'final':
        # Single final heat
        num_heats = 1
    elif next_round_type == 'semi':
        # Multiple semi-finals based on advancer count
        num_heats = ceil(len(all_advancers) / num_stands)
    else:
        num_heats = ceil(len(all_advancers) / num_stands)

    # Get DataFrame for advancers only
    all_advancers_df = tournament_state['all_competitors_df'][
        tournament_state['all_competitors_df']['competitor_name'].isin(all_advancers)
    ].copy()

    # Use same distribution algorithm (snake draft)
    next_rounds = distribute_competitors_into_heats(
        all_advancers_df,
        advancer_results,
        num_stands,
        num_heats
    )

    # Update round type and names
    for i, round_obj in enumerate(next_rounds):
        round_obj['round_type'] = next_round_type
        if next_round_type == 'final':
            round_obj['round_name'] = 'Final'
            round_obj['round_number'] = 1
        elif next_round_type == 'semi':
            round_obj['round_name'] = f'Semi {i + 1}'
            round_obj['round_number'] = i + 1

    return next_rounds


def view_tournament_status(tournament_state: Dict) -> None:
    """Display visual tournament bracket showing progress and results.

    Args:
        tournament_state: Global tournament state
    """

    print(f"\n{'='*70}")
    print(f"  TOURNAMENT STATUS")
    print(f"{'='*70}")

    if not tournament_state.get('rounds'):
        print("\nNo tournament rounds generated yet.")
        return

    print(f"\nEvent: {tournament_state.get('event_name', 'Unnamed Event')}")
    print(f"Format: {tournament_state.get('format', 'Unknown')}")
    print(f"Stands: {tournament_state.get('num_stands', '?')}")
    print(f"Total Competitors: {len(tournament_state.get('all_competitors', []))}")

    # Group rounds by type
    heats = [r for r in tournament_state['rounds'] if r['round_type'] == 'heat']
    semis = [r for r in tournament_state['rounds'] if r['round_type'] == 'semi']
    finals = [r for r in tournament_state['rounds'] if r['round_type'] == 'final']

    # Display heats
    if heats:
        print(f"\n{'─'*70}")
        print(f"INITIAL HEATS ({len(heats)} total)")
        print(f"{'─'*70}")
        for heat in heats:
            status_icon = "✓" if heat['status'] == 'completed' else "○"
            print(f"{status_icon} {heat['round_name']:15s} - {len(heat['competitors'])} competitors, top {heat['num_to_advance']} advance")
            if heat['status'] == 'completed' and heat.get('advancers'):
                print(f"    Advancers: {', '.join(heat['advancers'])}")

    # Display semis
    if semis:
        print(f"\n{'─'*70}")
        print(f"SEMI-FINALS ({len(semis)} total)")
        print(f"{'─'*70}")
        for semi in semis:
            status_icon = "✓" if semi['status'] == 'completed' else "○"
            print(f"{status_icon} {semi['round_name']:15s} - {len(semi['competitors'])} competitors, top {semi['num_to_advance']} advance")
            if semi['status'] == 'completed' and semi.get('advancers'):
                print(f"    Advancers: {', '.join(semi['advancers'])}")

    # Display finals
    if finals:
        print(f"\n{'─'*70}")
        print(f"FINAL")
        print(f"{'─'*70}")
        for final in finals:
            status_icon = "✓" if final['status'] == 'completed' else "○"
            print(f"{status_icon} {final['round_name']:15s} - {len(final['competitors'])} competitors")
            if final['status'] == 'completed':
                print(f"    Tournament Complete!")

    print(f"\n{'='*70}\n")


def save_tournament_state(tournament_state: Dict, filename: str = "tournament_state.json") -> None:
    """Save tournament state to JSON for crash recovery.

    Args:
        tournament_state: Tournament state to save
        filename: Output filename
    """
    try:
        # Convert DataFrames to dict format for JSON serialization
        state_copy = tournament_state.copy()

        # Convert main competitors DataFrame
        if not state_copy['all_competitors_df'].empty:
            state_copy['all_competitors_df'] = state_copy['all_competitors_df'].to_dict('records')
        else:
            state_copy['all_competitors_df'] = []

        # Convert DataFrames in rounds
        for round_obj in state_copy.get('rounds', []):
            if not round_obj['competitors_df'].empty:
                round_obj['competitors_df'] = round_obj['competitors_df'].to_dict('records')
            else:
                round_obj['competitors_df'] = []

        # Write to file
        with open(filename, 'w') as f:
            json.dump(state_copy, f, indent=2, default=str)

        print(f"Tournament state saved to {filename}")

    except Exception as e:
        print(f"Error saving tournament state: {e}")


def load_tournament_state(filename: str = "tournament_state.json") -> Optional[Dict]:
    """Load tournament state from JSON.

    Args:
        filename: Input filename

    Returns:
        dict: Loaded tournament state, or None if error
    """
    try:
        with open(filename, 'r') as f:
            state = json.load(f)

        # Convert dict records back to DataFrames
        if state.get('all_competitors_df'):
            state['all_competitors_df'] = pd.DataFrame(state['all_competitors_df'])
        else:
            state['all_competitors_df'] = pd.DataFrame()

        # Convert DataFrames in rounds
        for round_obj in state.get('rounds', []):
            if round_obj.get('competitors_df'):
                round_obj['competitors_df'] = pd.DataFrame(round_obj['competitors_df'])
            else:
                round_obj['competitors_df'] = pd.DataFrame()

        print(f"Tournament state loaded from {filename}")
        return state

    except FileNotFoundError:
        print(f"Tournament state file '{filename}' not found.")
        return None
    except Exception as e:
        print(f"Error loading tournament state: {e}")
        return None


def auto_save_state(tournament_state: Dict) -> None:
    """Auto-save tournament state after significant actions.

    Args:
        tournament_state: Tournament state to save
    """
    save_tournament_state(tournament_state, "tournament_state.json")
