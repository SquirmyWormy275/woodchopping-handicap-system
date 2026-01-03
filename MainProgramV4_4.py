# -*- coding: utf-8 -*-
"""
STRATHEX - Woodchopping Handicap Calculator v4.5
Professional Competition System
"""

#Import Pandas, numpy,
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
from math import ceil
from openpyxl import load_workbook
import time
import os

# Fix Windows console encoding for Unicode characters
if sys.platform == 'win32':
    try:
        # Try to set UTF-8 encoding for Windows console
        os.system('chcp 65001 >nul 2>&1')
        # Also set stdout encoding
        if hasattr(sys.stdout, 'reconfigure'):
            sys.stdout.reconfigure(encoding='utf-8')
    except:
        pass  # If it fails, we'll fallback to ASCII banner

# Import functions from modular woodchopping package
from woodchopping.data import (
    load_competitors_df,
    load_results_df,
    append_results_to_excel,
)
from woodchopping.ui import (
    wood_menu,
    select_event_code,
    select_all_event_competitors,
    personnel_management_menu,
)
from woodchopping.ui.handicap_ui import (
    judge_approval,
    manual_adjust_handicaps,
)
from woodchopping.predictions.prediction_aggregator import (
    display_basic_prediction_table,
    display_comprehensive_prediction_analysis,
    display_handicap_calculation_explanation,
)
from woodchopping.ui.tournament_ui import (
    calculate_tournament_scenarios,
    distribute_competitors_into_heats,
    select_heat_advancers,
    generate_next_round,
    view_tournament_status,
    save_tournament_state,
    load_tournament_state,
    auto_save_state,
)
from woodchopping.ui.multi_event_ui import (
    create_multi_event_tournament,
    save_multi_event_tournament,
    load_multi_event_tournament,
    add_event_to_tournament,
    view_wood_count,
    view_tournament_schedule,
    remove_event_from_tournament,
    calculate_all_event_handicaps,
    view_analyze_all_handicaps,
    approve_event_handicaps,
    generate_complete_day_schedule,
    sequential_results_workflow,
    generate_tournament_summary
)
from woodchopping.handicaps import calculate_ai_enhanced_handicaps
from woodchopping.simulation import simulate_and_assess_handicaps

# Keep explanation system (educational tool)
import explanation_system_functions as explain

# STRATHEX Banner
try:
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║                                                                      ║
║    ██████╗████████╗██████╗  █████╗ ████████╗██╗  ██╗███████╗██╗  ██╗ ║
║   ██╔════╝╚══██╔══╝██╔══██╗██╔══██╗╚══██╔══╝██║  ██║██╔════╝╚██╗██╔╝ ║
║   ╚█████╗    ██║   ██████╔╝███████║   ██║   ███████║█████╗   ╚███╔╝  ║
║    ╚═══██╗   ██║   ██╔══██╗██╔══██║   ██║   ██╔══██║██╔══╝   ██╔██╗  ║
║   ██████╔╝   ██║   ██║  ██║██║  ██║   ██║   ██║  ██║███████╗██╔╝╚██╗ ║
║   ╚═════╝    ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝   ╚═╝   ╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝ ║
║                                                                      ║
║              WOODCHOPPING HANDICAP CALCULATOR v4.5                   ║
║                    Professional Competition System                   ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
""")
except UnicodeEncodeError:
    # Fallback to ASCII banner if Unicode fails
    print("""
======================================================================

    STRATHEX - WOODCHOPPING HANDICAP CALCULATOR v4.5
                Professional Competition System

======================================================================
""")
time.sleep(0.8)
#Load Competitor Data from Excel  full roster)
'''Read the xlsx file containing data. 
Sheet "competitors" contains competitor data
Sheet "wood" contains wood species data'''
try:
    comp_df = load_competitors_df()
except Exception as e:
    print(f"Error loading roster from Excel: {e}")
    comp_df = pd.DataFrame(columns=["competitor_name", "competitor_country"])

#Wood Selection Dictionary - initialize with all expected keys
wood_selection = {
    "species": None, 
    "size_mm": None, 
    "quality": None,
    "event": None
}

# Heat assignment state (competitors selected for current heat) - LEGACY for backward compatibility
heat_assignment_df = pd.DataFrame()
heat_assignment_names = []

# Tournament State - NEW multi-round tournament system
tournament_state = {
    'event_name': None,                    # e.g., "SB Championship 2025"
    'num_stands': None,                    # e.g., 8 (number of available chopping stands)
    'tentative_competitors': None,         # User's estimate for planning
    'format': None,                        # "heats_to_finals" or "heats_to_semis_to_finals"
    'all_competitors': [],                 # List of competitor names in event
    'all_competitors_df': pd.DataFrame(),  # DataFrame of all competitors
    'rounds': [],                          # List of Round objects (heats, semis, finals)
    'current_round_index': 0,              # Active round index
    'capacity_info': {},                   # From calculate_tournament_scenarios()
    'handicap_results_all': []             # Handicap results for all competitors
}

# Multi-Event Tournament State - NEW for V4.5
multi_event_tournament_state = {
    'tournament_name': None,               # e.g., "Mason County Western Qualifier 2025"
    'date': None,                          # Tournament date
    'location': None,                      # Venue name
    'events': [],                          # List of event objects (each contains tournament_state)
    'current_event_index': 0,              # Active event being worked on
    'schedule': [],                        # Complete day schedule across all events
    'results': []                          # Final results across all events
}

## Competitor Selection Menu
''' Official will be presented with a list of competitors

    Definitions:
    'Roster'- list of all competitors available in the excel sheet
    'Heat Assignment'- list of competitors selected for the current heat

    1. Select Competitors for Heat from the roster
    2. Add competitors to the roster
    3. View Heat Assignment
    4. Remove competitors from the heat assignment
    5. Return to Main Menu
'''

## Wood Characteristics Menu
''' Official will be presented with a list of wood characteristics
    1. Select Wood Species from the list
    2. Enter Size in mm
    3. Enter wood quality 
    (0 for poor quality, 1-3 for soft, 4-7 for average firmness for species, 8-10 for above average firmness for species)
    4. Return to Main Menu
'''

##Select Event (SB/UH)
''' Official will be presented with two event codes to select from either SB or UH'''

## View Handicap Marks
''' Official will be presented with the calculated handicap marks for each selected competitor in the heat
    1. View Handicap Marks
    2. Return to Main Menu
'''


## Single Event Menu Function
def single_event_menu():
    """Menu for designing and running a single event"""
    global tournament_state, wood_selection, heat_assignment_df, heat_assignment_names, comp_df

    while True:
        print("\n╔════════════════════════════════════════════════════════════════════╗")
        print("║                                                                    ║")
        print("║  ██╗  ██╗ █████╗ ███╗   ██╗██████╗ ██╗ ██████╗ █████╗ ██████╗      ║")
        print("║  ██║  ██║██╔══██╗████╗  ██║██╔══██╗██║██╔════╝██╔══██╗██╔══██╗     ║")
        print("║  ███████║███████║██╔██╗ ██║██║  ██║██║██║     ███████║██████╔╝     ║")
        print("║  ██╔══██║██╔══██║██║╚██╗██║██║  ██║██║██║     ██╔══██║██╔═══╝      ║")
        print("║  ██║  ██║██║  ██║██║ ╚████║██████╔╝██║╚██████╗██║  ██║██║          ║")
        print("║  ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═══╝╚═════╝ ╚═╝ ╚═════╝╚═╝  ╚═╝╚═╝          ║")
        print("║                                                                    ║")
        print("║                      ⚒  Single Event Mode  ⚒                      ║")
        print("║                                                                    ║")
        print("╚════════════════════════════════════════════════════════════════════╝")
        print("\nTOURNAMENT SETUP:")
        print("  1. Configure Wood Characteristics")
        print("  2. Configure Tournament (Stands + Format)")
        print("  3. Select Competitors for Event")
        print("\nHANDICAP ANALYSIS:")
        print("  4. Calculate Handicaps for All Competitors")
        print("  5. View Handicaps & Fairness Analysis")
        print("\nHEAT MANAGEMENT:")
        print("  6. Generate Initial Heats")
        print("  7. Record Heat Results & Select Advancers")
        print("  8. Generate Next Round (Semi/Final)")
        print("  9. View Event Status")
        print("\nSYSTEM:")
        print(" 10. Save Event State")
        print(" 11. Return to Main Menu")
        print("=" * 70)

        menu_choice = input("\nEnter your choice (1-11): ").strip()

        if menu_choice == '1':
            # Configure wood characteristics (same as before)
            wood_selection = wood_menu(wood_selection)

        elif menu_choice == '2':
            # Configure Tournament: stands + format
            print("\n" + "=" * 70)
            print("  CONFIGURE TOURNAMENT")
            print("=" * 70)

            try:
                num_stands = int(input("\nNumber of available stands: ").strip())
                tentative = int(input("Approximate number of competitors: ").strip())

                # Get event code first
                if not wood_selection.get('event'):
                    wood_selection = select_event_code(wood_selection)

                # Calculate scenarios
                scenarios = calculate_tournament_scenarios(num_stands, tentative)

                # Display all three scenarios
                print("\n" + "=" * 70)
                print("  SCENARIO 1: Single Heat Mode")
                print("=" * 70)
                print(scenarios['single_heat']['description'])
                print(f"\nTotal blocks needed: {scenarios['single_heat']['total_blocks']}")

                print("\n" + "=" * 70)
                print("  SCENARIO 2: Heats → Finals")
                print("=" * 70)
                print(scenarios['heats_to_finals']['description'])
                print(f"\nTotal blocks needed: {scenarios['heats_to_finals']['total_blocks']}")

                print("\n" + "=" * 70)
                print("  SCENARIO 3: Heats → Semis → Finals")
                print("=" * 70)
                print(scenarios['heats_to_semis_to_finals']['description'])
                print(f"\nTotal blocks needed: {scenarios['heats_to_semis_to_finals']['total_blocks']}")

                # User selects format
                print("\n" + "=" * 70)
                format_choice = input("Select format (1, 2, or 3): ").strip()

                if format_choice == '1':
                    tournament_state['format'] = 'single_heat'
                    tournament_state['capacity_info'] = scenarios['single_heat']
                elif format_choice == '2':
                    tournament_state['format'] = 'heats_to_finals'
                    tournament_state['capacity_info'] = scenarios['heats_to_finals']
                elif format_choice == '3':
                    tournament_state['format'] = 'heats_to_semis_to_finals'
                    tournament_state['capacity_info'] = scenarios['heats_to_semis_to_finals']
                else:
                    print("Invalid choice. Tournament not configured.")
                    input("\nPress Enter to return to menu...")
                    continue

                tournament_state['num_stands'] = num_stands
                tournament_state['tentative_competitors'] = tentative

                # Prompt for event name
                event_name = input("\nEvent name (e.g., 'SB Championship 2025'): ").strip()
                tournament_state['event_name'] = event_name if event_name else "Unnamed Event"

                print(f"\n✓ Tournament configured: {tournament_state['format']}")
                print(f"✓ Max competitors: {tournament_state['capacity_info']['max_competitors']}")

            except ValueError:
                print("Invalid input. Please enter numbers.")

        elif menu_choice == '3':
            # Select ALL competitors for event
            if not tournament_state.get('num_stands'):
                print("\nERROR: Configure tournament first (Option 2)")
                input("\nPress Enter to return to menu...")
                continue

            max_comp = tournament_state['capacity_info'].get('max_competitors')
            print(f"\nMaximum competitors for this format: {max_comp}")

            selected_df = select_all_event_competitors(comp_df, max_comp)

            if not selected_df.empty:
                tournament_state['all_competitors_df'] = selected_df
                tournament_state['all_competitors'] = selected_df['competitor_name'].tolist()
                print(f"\n✓ {len(tournament_state['all_competitors'])} competitors selected for event")

        elif menu_choice == '4':
            # Calculate handicaps for ALL competitors
            # Comprehensive validation with helpful error messages
            missing = []

            # Check tournament configuration
            if not tournament_state.get('num_stands'):
                missing.append("  ✗ Tournament not configured (use Option 2)")

            # Check competitors
            if not tournament_state.get('all_competitors'):
                missing.append("  ✗ No competitors selected (use Option 3)")

            # Check wood characteristics
            if not wood_selection.get('species'):
                missing.append("  ✗ Wood species not selected (use Option 1)")

            if not wood_selection.get('size_mm'):
                missing.append("  ✗ Wood size (diameter) not set (use Option 1)")

            if wood_selection.get('quality') is None:
                missing.append("  ✗ Wood quality not set (use Option 1)")

            # Check event code
            if not wood_selection.get('event'):
                missing.append("  ✗ Event type not selected (SB/UH - use Option 1 or 3)")

            # If anything is missing, show comprehensive error
            if missing:
                box_width = 68
                print("\n╔" + "═" * box_width + "╗")

                # Center the title
                title = "⚠ CANNOT CALCULATE HANDICAPS ⚠"
                title_line = title.center(box_width)
                print("║" + title_line + "║")

                print("╠" + "═" * box_width + "╣")

                # Header line
                header = "Missing required information:".ljust(box_width)
                print("║" + header + "║")

                # Missing items
                for item in missing:
                    print("║" + item.ljust(box_width) + "║")

                print("╚" + "═" * box_width + "╝")
                print("\nPlease complete the missing items above, then try again.")
                input("\nPress Enter to return to menu...")
                continue

            # Live progress animation
            print("\n╔" + "═" * 70 + "╗")
            print("║" + "   ⏱  HANDICAP CALCULATION IN PROGRESS".ljust(70) + "║")
            print("╠" + "═" * 70 + "╣")
            print("║" + " " * 70 + "║")  # Progress bar line
            print("║" + " " * 70 + "║")  # Competitor name line
            print("║" + " " * 70 + "║")  # Progress info line
            print("╚" + "═" * 70 + "╝")

            def show_progress(current, total, comp_name):
                """Display live progress bar with proper alignment"""
                percent = int((current / total) * 100)
                bar_length = 40
                filled = int((bar_length * current) / total)
                bar = '█' * filled + '░' * (bar_length - filled)

                # Truncate competitor name to fit and ensure it's visible
                max_name_length = 45
                if len(comp_name) > max_name_length:
                    display_name = comp_name[:max_name_length-3] + "..."
                else:
                    display_name = comp_name

                # Format progress info
                progress_info = f"{current}/{total} competitors"

                # Build properly padded lines (70 chars inside the box)
                line1 = f"  [{bar}] {percent:3d}%".ljust(70)
                line2 = f"  ⚒ Analyzing: {display_name}".ljust(70)
                line3 = f"  Progress: {progress_info}".ljust(70)

                # Move cursor up 4 lines (to just after the top border)
                sys.stdout.write('\033[4A')
                # Overwrite the 3 content lines
                sys.stdout.write(f"║{line1}║\n")
                sys.stdout.write(f"║{line2}║\n")
                sys.stdout.write(f"║{line3}║\n")
                sys.stdout.flush()

            # Use existing calculate_ai_enhanced_handicaps function with progress
            results_df = load_results_df()
            handicap_results = calculate_ai_enhanced_handicaps(
                tournament_state['all_competitors_df'],
                wood_selection['species'],
                wood_selection['size_mm'],
                wood_selection['quality'],
                wood_selection['event'],
                results_df,
                progress_callback=show_progress
            )

            # Clear progress lines and show completion
            sys.stdout.write('\033[4A')  # Move cursor up 4 lines to content area
            complete_bar = '█' * 40
            line1 = f"  [{complete_bar}] 100%".ljust(70)
            line2 = f"  ✓ All competitors analyzed successfully!".ljust(70)
            line3 = " " * 70
            sys.stdout.write(f"║{line1}║\n")
            sys.stdout.write(f"║{line2}║\n")
            sys.stdout.write(f"║{line3}║\n")
            sys.stdout.write(f"╚{'═' * 70}╝\n")
            sys.stdout.flush()

            tournament_state['handicap_results_all'] = handicap_results

            # Store wood characteristics in tournament state for recalculation in later rounds
            tournament_state['wood_species'] = wood_selection['species']
            tournament_state['wood_diameter'] = wood_selection['size_mm']
            tournament_state['wood_quality'] = wood_selection['quality']
            tournament_state['event_code'] = wood_selection['event']

            # Success message with axe icon
            print("\n" + "=" * 70)
            print("    ⚒  HANDICAP CALCULATION COMPLETE! ⚒")
            print(f"    ✓  {len(handicap_results)} competitors analyzed")
            print("=" * 70)

        elif menu_choice == '5':
            # View handicaps + comprehensive analysis (NEW 5-PHASE FLOW)
            if not tournament_state.get('handicap_results_all'):
                print("\nERROR: Calculate handicaps first (Option 4)")
                input("\nPress Enter to return to menu...")
                continue

            # PHASE 1: Display initial handicap marks with basic prediction table
            display_basic_prediction_table(
                tournament_state['handicap_results_all'],
                wood_selection
            )

            # PHASE 2: Monte Carlo fairness simulation
            run_mc = input("\nRun Monte Carlo fairness simulation? (y/n): ").strip().lower()
            if run_mc == 'y':
                simulate_and_assess_handicaps(tournament_state['handicap_results_all'])

            # PHASE 3: Comprehensive AI Analysis of Predictions
            show_analysis = input("\nView detailed AI analysis of prediction methods? (y/n): ").strip().lower()
            if show_analysis == 'y':
                try:
                    display_comprehensive_prediction_analysis(
                        tournament_state['handicap_results_all'],
                        wood_selection
                    )
                except Exception as e:
                    print(f"\n⚠ Error during AI analysis: {e}")
                    print("Continuing without AI analysis...")

            # PHASE 4: Optional explanation of handicap calculations
            show_calc = input("\nView explanation of how handicaps are calculated? (y/n): ").strip().lower()
            if show_calc == 'y':
                display_handicap_calculation_explanation()

            # PHASE 5: Judge approval workflow
            print("\n" + "="*70)
            print("  HANDICAP APPROVAL")
            print("="*70)
            print("\n1. Accept handicaps as calculated")
            print("2. Manually adjust individual handicaps")

            approval_choice = input("\nYour choice (1 or 2): ").strip()

            if approval_choice == '2':
                # Manual adjustment workflow
                adjusted_results, initials, timestamp = manual_adjust_handicaps(
                    tournament_state['handicap_results_all'],
                    wood_selection
                )

                if initials:  # Adjustment approved
                    # Update with adjusted marks
                    tournament_state['handicap_results_all'] = adjusted_results
                    # Store approval metadata
                    for result in adjusted_results:
                        result['approved_by'] = initials
                        result['approved_at'] = timestamp
                    print(f"\n✓ Handicaps approved by {initials} at {timestamp}")
                else:
                    print("\n⚠ Adjustment cancelled - handicaps NOT approved")

            elif approval_choice == '1':
                # Accept as calculated - still requires approval
                initials, timestamp = judge_approval()

                if initials:  # Accepted and approved
                    for result in tournament_state['handicap_results_all']:
                        result['approved_by'] = initials
                        result['approved_at'] = timestamp
                    print(f"\n✓ Handicaps approved by {initials} at {timestamp}")
                else:
                    print("\n⚠ Approval cancelled - handicaps NOT approved")
            else:
                print("\n⚠ Invalid choice - handicaps NOT approved")

        elif menu_choice == '6':
            # Generate initial heats
            if not tournament_state.get('handicap_results_all'):
                print("\nERROR: Calculate handicaps first (Option 4)")
                input("\nPress Enter to return to menu...")
                continue

            # Check if heats already exist
            if tournament_state.get('rounds'):
                print("\n" + "=" * 70)
                print("  ⚠ WARNING: HEATS ALREADY EXIST")
                print("=" * 70)
                print(f"\nExisting heats: {len(tournament_state['rounds'])} heat(s)")
                regenerate = input("\nRegenerate heats? This will OVERWRITE existing assignments (y/n): ").strip().lower()
                if regenerate != 'y':
                    print("\n✓ Heat generation cancelled")
                    input("\nPress Enter to return to menu...")
                    continue

            num_competitors = len(tournament_state['all_competitors'])
            num_stands = tournament_state['num_stands']

            # Check if single heat mode
            if tournament_state.get('format') == 'single_heat':
                print(f"\nGenerating single heat for training/testing...")

                # Create single heat with all competitors
                heats = [{
                    'round_name': 'Heat 1',
                    'round_type': 'heat',
                    'competitors': tournament_state['all_competitors'],
                    'handicap_results': tournament_state['handicap_results_all'],
                    'num_to_advance': 0,  # No advancement in single heat mode
                    'status': 'pending',
                    'actual_results': {},
                    'advancers': []
                }]

                tournament_state['rounds'] = heats

                # Display heat assignment
                print(f"\n{'='*70}")
                print(f"  SINGLE HEAT - TRAINING/TESTING MODE")
                print(f"{'='*70}")
                print(f"\nHeat 1 ({len(heats[0]['competitors'])} competitors - No advancement):")
                for i, name in enumerate(heats[0]['competitors'], 1):
                    mark = next((c['mark'] for c in heats[0]['handicap_results'] if c['name'] == name), '?')
                    print(f"  {i}) {name:35s} (Mark {mark})")
                print(f"{'='*70}")
                print("\n✓ Results can be recorded and saved to build historical data")
            else:
                # Multi-round tournament mode
                num_heats = ceil(num_competitors / num_stands)
                print(f"\nGenerating {num_heats} heats with balanced skill distribution...")

                heats = distribute_competitors_into_heats(
                    tournament_state['all_competitors_df'],
                    tournament_state['handicap_results_all'],
                    num_stands,
                    num_heats
                )

                tournament_state['rounds'] = heats

                # Display heat assignments
                print(f"\n{'='*70}")
                print(f"  HEAT ASSIGNMENTS")
                print(f"{'='*70}")
                for heat in heats:
                    print(f"\n{heat['round_name']} ({len(heat['competitors'])} competitors, top {heat['num_to_advance']} advance):")
                    for i, name in enumerate(heat['competitors'], 1):
                        # Find mark for this competitor
                        mark = next((c['mark'] for c in heat['handicap_results'] if c['name'] == name), '?')
                        print(f"  {i}) {name:35s} (Mark {mark})")
                print(f"{'='*70}")

            # Auto-save
            auto_save_state(tournament_state)
            print("\n✓ Tournament state auto-saved")
            input("\nPress Enter to return to menu...")

        elif menu_choice == '7':
            # Record heat results + select advancers
            if not tournament_state.get('rounds'):
                print("\nERROR: Generate heats first (Option 6)")
                input("\nPress Enter to return to menu...")
                continue

            pending = [r for r in tournament_state['rounds'] if r['status'] == 'pending']
            in_progress = [r for r in tournament_state['rounds'] if r['status'] == 'in_progress']

            available = pending + in_progress

            if not available:
                print("\nAll heats in current round completed!")
                print("Use Option 8 to generate next round.")
                input("\nPress Enter to return to menu...")
                continue

            # Select heat to record
            print(f"\n{'='*70}")
            print(f"  SELECT HEAT TO RECORD")
            print(f"{'='*70}")
            for i, heat in enumerate(available, 1):
                status = "⚠ In Progress" if heat['status'] == 'in_progress' else "○ Pending"
                print(f"{i}) {heat['round_name']:15s} - {len(heat['competitors'])} competitors ({status})")

            try:
                heat_choice = int(input("\nSelect heat to record (number): ").strip()) - 1
                selected_heat = available[heat_choice]

                # Record times
                print(f"\n{'='*70}")
                print(f"  RECORDING RESULTS FOR {selected_heat['round_name']}")
                print(f"{'='*70}")
                append_results_to_excel(
                    heat_assignment_df,  # Legacy param (not used)
                    wood_selection,
                    round_object=selected_heat,
                    tournament_state=tournament_state
                )

                # Select advancers (skip for single heat mode)
                if tournament_state.get('format') == 'single_heat':
                    # Single heat mode - no advancement
                    selected_heat['status'] = 'completed'
                    selected_heat['advancers'] = []
                    print(f"\n✓ {selected_heat['round_name']} completed")
                    print("✓ Results saved to historical data for future handicap calculations")
                else:
                    # Multi-round tournament - select advancers
                    advancers = select_heat_advancers(selected_heat)
                    print(f"\n✓ {selected_heat['round_name']} completed")
                    print(f"✓ Advancers: {', '.join(advancers)}")

                # Auto-save
                auto_save_state(tournament_state)

            except (ValueError, IndexError):
                print("Invalid selection.")

        elif menu_choice == '8':
            # Generate next round (semi/final)
            if not tournament_state.get('rounds'):
                print("\nERROR: No tournament rounds exist yet")
                input("\nPress Enter to return to menu...")
                continue

            # Check if all current round heats completed
            current_rounds = tournament_state['rounds']
            incomplete = [r for r in current_rounds if r['status'] != 'completed']

            if incomplete:
                print(f"\nERROR: {len(incomplete)} heat(s) not yet completed:")
                for heat in incomplete:
                    print(f"  - {heat['round_name']}")
                print("\nComplete all heats before generating next round.")
                input("\nPress Enter to return to menu...")
                continue

            # Collect all advancers
            all_advancers = []
            for heat in current_rounds:
                all_advancers.extend(heat.get('advancers', []))

            print(f"\n{len(all_advancers)} competitors advancing to next round:")
            for name in all_advancers:
                print(f"  - {name}")

            # Determine next round type
            current_type = current_rounds[0]['round_type']
            tournament_format = tournament_state.get('format')

            if current_type == 'heat':
                if tournament_format == 'heats_to_finals':
                    next_type = 'final'
                else:  # heats_to_semis_to_finals
                    next_type = 'semi'
            elif current_type == 'semi':
                next_type = 'final'
            else:
                print("\nTournament already has finals generated!")
                input("\nPress Enter to return to menu...")
                continue

            print(f"\nGenerating {next_type} round...")

            # Generate next round (single-event mode always uses handicaps)
            next_rounds = generate_next_round(tournament_state, all_advancers, next_type, is_championship=False)

            # Append to tournament state
            tournament_state['rounds'].extend(next_rounds)

            print(f"\n✓ {len(next_rounds)} {next_type} heat(s) generated")

            # Display
            for round_obj in next_rounds:
                print(f"\n{round_obj['round_name']} ({len(round_obj['competitors'])} competitors):")
                for name in round_obj['competitors']:
                    print(f"  - {name}")

            # Auto-save
            auto_save_state(tournament_state)

        elif menu_choice == '9':
            # View tournament status
            view_tournament_status(tournament_state)

        elif menu_choice == '10':
            # Save tournament state
            save_tournament_state(tournament_state, "tournament_state.json")

        elif menu_choice == '11':
            # Return to main menu
            print("\nReturning to main menu...")
            break

        else:
            print("Invalid selection. Try again.")


## Multi-Event Tournament Menu Function
def multi_event_tournament_menu():
    """Menu for designing and running multi-event tournaments"""
    global multi_event_tournament_state, comp_df

    while True:
        print("\n╔════════════════════════════════════════════════════════════════════╗")
        print("║" + " " * 68 + "║")
        print("║" + "──────────────────────────────".center(68) + "║")
        print("║" + "TOURNAMENT MANAGEMENT SYSTEM".center(68) + "║")
        print("║" + "Multi-Event Handicapping".center(68) + "║")
        print("║" + "──────────────────────────────".center(68) + "║")
        print("║" + "⚒ STRATHEX ⚒".center(68) + "║")
        print("║" + " " * 68 + "║")
        print("╚════════════════════════════════════════════════════════════════════╝")
        print("\nTOURNAMENT SETUP:")
        print("  1. Create New Tournament")
        print("  2. Add Event to Tournament")
        print("  3. Wood Count")
        print("  4. Remove Event from Tournament")
        print("\nHANDICAP CALCULATION:")
        print("  5. Calculate Handicaps for All Events")
        print("  6. View & Analyze Handicaps")
        print("  7. Approve Event Handicaps")
        print("\nTOURNAMENT EXECUTION:")
        print("  8. Generate Complete Day Schedule")
        print("  9. Begin Sequential Results Entry")
        print(" 10. View Tournament Status")
        print("\nTOURNAMENT COMPLETION:")
        print(" 11. Generate Final Tournament Summary")
        print(" 12. Save Tournament State")
        print(" 13. Return to Main Menu")
        print("=" * 70)

        menu_choice = input("\nEnter your choice (1-13): ").strip()

        if menu_choice == '1':
            # Create New Tournament
            multi_event_tournament_state = create_multi_event_tournament()

        elif menu_choice == '2':
            # Add Event to Tournament
            if not multi_event_tournament_state.get('tournament_name'):
                print("\nERROR: Create tournament first (Option 1)")
                input("\nPress Enter to return to menu...")
                continue

            results_df = load_results_df()
            multi_event_tournament_state = add_event_to_tournament(
                multi_event_tournament_state,
                comp_df,
                results_df
            )

        elif menu_choice == '3':
            # Wood Count
            if not multi_event_tournament_state.get('tournament_name'):
                print("\nERROR: Create tournament first (Option 1)")
                input("\nPress Enter to return to menu...")
                continue

            view_wood_count(multi_event_tournament_state)

        elif menu_choice == '4':
            # Remove Event from Tournament
            if not multi_event_tournament_state.get('events'):
                print("\nERROR: No events in tournament to remove")
                input("\nPress Enter to return to menu...")
                continue

            multi_event_tournament_state = remove_event_from_tournament(
                multi_event_tournament_state
            )

        elif menu_choice == '5':
            # Calculate Handicaps for All Events (BATCH)
            if not multi_event_tournament_state.get('events'):
                print("\nERROR: Add events to tournament first (Option 2)")
                input("\nPress Enter to return to menu...")
                continue

            results_df = load_results_df()
            multi_event_tournament_state = calculate_all_event_handicaps(
                multi_event_tournament_state,
                results_df
            )

        elif menu_choice == '6':
            # View & Analyze All Handicaps
            if not multi_event_tournament_state.get('events'):
                print("\nERROR: No events in tournament")
                input("\nPress Enter to return to menu...")
                continue

            view_analyze_all_handicaps(multi_event_tournament_state)

        elif menu_choice == '7':
            # Approve Event Handicaps
            if not multi_event_tournament_state.get('events'):
                print("\nERROR: No events in tournament")
                input("\nPress Enter to return to menu...")
                continue

            approve_event_handicaps(multi_event_tournament_state)

        elif menu_choice == '8':
            # Generate Complete Day Schedule
            if not multi_event_tournament_state.get('events'):
                print("\nERROR: Add events to tournament first (Option 2)")
                input("\nPress Enter to return to menu...")
                continue

            multi_event_tournament_state = generate_complete_day_schedule(
                multi_event_tournament_state
            )

        elif menu_choice == '9':
            # Begin Sequential Results Entry
            if not multi_event_tournament_state.get('events'):
                print("\nERROR: Generate day schedule first (Option 8)")
                input("\nPress Enter to return to menu...")
                continue

            multi_event_tournament_state = sequential_results_workflow(
                multi_event_tournament_state,
                wood_selection,      # Legacy parameter
                heat_assignment_df   # Legacy parameter
            )

        elif menu_choice == '10':
            # View Tournament Status
            if not multi_event_tournament_state.get('tournament_name'):
                print("\nERROR: Create tournament first (Option 1)")
                input("\nPress Enter to return to menu...")
                continue

            view_tournament_schedule(multi_event_tournament_state)

        elif menu_choice == '11':
            # Generate Final Tournament Summary
            if not multi_event_tournament_state.get('events'):
                print("\nERROR: No events completed yet")
                input("\nPress Enter to return to menu...")
                continue

            generate_tournament_summary(multi_event_tournament_state)

        elif menu_choice == '12':
            # Save Tournament State
            if not multi_event_tournament_state.get('tournament_name'):
                print("\nERROR: Create tournament first (Option 1)")
                input("\nPress Enter to return to menu...")
                continue

            filename = input("\nEnter filename (default: multi_tournament_state.json): ").strip()
            if not filename:
                filename = "multi_tournament_state.json"

            save_multi_event_tournament(multi_event_tournament_state, filename)

        elif menu_choice == '13':
            # Return to main menu
            print("\nReturning to main menu...")
            break

        else:
            print("Invalid selection. Try again.")


## Main Menu - Top Level Mode Selection
''' NEW TWO-LEVEL MENU STRUCTURE (V4.5)
Top level: Choose between single event or multi-event tournament
Second level: Detailed menus for each mode
'''
while True:
    print("\n" + "╔" + "═" * 68 + "╗")
    print("║" + " " * 19 + "⚒ STRATHEX MAIN MENU ⚒" + " " * 27 + "║")
    print("╚" + "═" * 68 + "╝")
    print("\nMODE SELECTION:")
    print("  1. Design an Event (Single Event)")
    print("  2. Design a Tournament (Multiple Events)")
    print("\nPERSONNEL & ROSTER:")
    print("  3. Add/Edit/Remove Competitors from Master Roster")
    print("\nLOAD PREVIOUS WORK:")
    print("  4. Load Previous Event/Tournament")
    print("  5. Reload Roster from Excel")
    print("\nSYSTEM:")
    print("  6. How Does This System Work? (Explanation System)")
    print("  7. Exit")
    print("=" * 70)

    menu_choice = input("\nEnter your choice (1-7): ").strip()

    if menu_choice == '1':
        # Design an Event (Single Event Mode)
        single_event_menu()

    elif menu_choice == '2':
        # Design a Tournament (Multiple Events Mode)
        multi_event_tournament_menu()

    elif menu_choice == '3':
        # Personnel management menu
        comp_df = personnel_management_menu(comp_df)

    elif menu_choice == '4':
        # Load Previous Event/Tournament
        print("\n" + "=" * 70)
        print("  LOAD PREVIOUS WORK")
        print("=" * 70)
        print("\n1. Load Single Event State")
        print("2. Load Multi-Event Tournament State")
        print("3. Cancel")

        load_choice = input("\nEnter your choice (1-3): ").strip()

        if load_choice == '1':
            # Load single event
            loaded_state = load_tournament_state("tournament_state.json")
            if loaded_state:
                tournament_state.update(loaded_state)
                print("\n✓ Single event state loaded successfully")
                input("\nPress Enter to return to menu...")
        elif load_choice == '2':
            # Load multi-event tournament
            filename = input("\nEnter filename (default: multi_tournament_state.json): ").strip()
            if not filename:
                filename = "multi_tournament_state.json"

            loaded_multi_state = load_multi_event_tournament(filename)
            if loaded_multi_state:
                multi_event_tournament_state.update(loaded_multi_state)
                print("\n✓ Multi-event tournament state loaded successfully")
                input("\nPress Enter to return to menu...")
        else:
            print("\nLoad cancelled")

    elif menu_choice == '5':
        # Reload roster from Excel
        try:
            comp_df = load_competitors_df()
            print("\n✓ Roster reloaded from Excel")
            input("\nPress Enter to return to menu...")
        except Exception as e:
            print(f"\nFailed to reload roster: {e}")
            input("\nPress Enter to return to menu...")

    elif menu_choice == '6':
        # Launch explanation system
        explain.explanation_menu()

    elif menu_choice == '7' or menu_choice == '':
        # Exit
        save_prompt = input("\nSave current work before exiting? (y/n): ").strip().lower()
        if save_prompt == 'y':
            print("\n" + "=" * 70)
            print("  SAVE OPTIONS")
            print("=" * 70)
            print("\n1. Save Single Event State")
            print("2. Save Multi-Event Tournament State")
            print("3. Skip saving")

            save_choice = input("\nEnter your choice (1-3): ").strip()

            if save_choice == '1':
                auto_save_state(tournament_state)
                print("\n✓ Single event state saved")
            elif save_choice == '2':
                filename = input("\nEnter filename (default: multi_tournament_state.json): ").strip()
                if not filename:
                    filename = "multi_tournament_state.json"
                save_multi_event_tournament(multi_event_tournament_state, filename)

        print("\nGoodbye!")
        break

    else:
        print("Invalid selection. Try again.")
        input("\nPress Enter to return to menu...")