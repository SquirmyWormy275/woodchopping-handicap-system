"""Handicap display and results recording UI functions.

This module handles handicap viewing and heat results operations including:
- Viewing handicap marks with Monte Carlo validation
- Validating heat data completeness
- Recording and saving heat results to Excel
"""

from datetime import datetime
from typing import Dict, Optional, List
import pandas as pd
from openpyxl import load_workbook, Workbook

from woodchopping.data import (
    load_results_df,
    get_competitor_id_name_mapping,
    detect_results_sheet
)
from woodchopping.handicaps import calculate_ai_enhanced_handicaps
from woodchopping.simulation import simulate_and_assess_handicaps
from config import paths


# File/sheet names from config
RESULTS_FILE = paths.EXCEL_FILE
RESULTS_SHEET = paths.RESULTS_SHEET


def view_handicaps_menu(heat_assignment_df: pd.DataFrame, wood_selection: Dict) -> None:
    """View Handicap Marks Menu.

    Official will be presented with the calculated handicap marks for each
    selected competitor in the heat.

    Menu options:
    1. View handicap marks for current heat
    2. Back to Main Menu

    Args:
        heat_assignment_df: Competitors currently assigned to heat
        wood_selection: Dictionary containing wood characteristics
    """

    # Safety check to make sure that either Standing Block or Underhand is selected
    # If not, will default to None so program doesn't crash
    if "event" not in wood_selection:
        wood_selection["event"] = None

    # Menu loop
    while True:
        print("\n--- View Handicap Marks ---")
        print("1) View handicap marks for current heat")
        print("2) Back to Main Menu")
        s = input("Choose an option: ").strip()

        if s == "1":
            if not validate_heat_data(heat_assignment_df, wood_selection):
                continue
            view_handicaps(heat_assignment_df, wood_selection)
            input("\n(Press Enter to return to the View Handicap Marks menu) ")

        elif s == "2" or s == "":
            break

        else:
            print("Invalid selection. Try again.")


def validate_heat_data(heat_assignment_df: pd.DataFrame, wood_selection: Dict) -> bool:
    """Validate that heat data is complete before calculating handicaps.

    Checks specifically for:
    - Competitors in heat assignment
    - Wood species selected
    - Wood size entered
    - Event code selected (SB/UH)

    Args:
        heat_assignment_df: Competitors assigned to heat
        wood_selection: Wood characteristics dictionary

    Returns:
        bool: True if all data is present, False otherwise
    """
    if heat_assignment_df is None or heat_assignment_df.empty:
        print("\nNo competitors in heat assignment. Use Competitor Menu -> Select Competitors for Heat.")
        return False

    if not wood_selection.get("species") or not wood_selection.get("size_mm"):
        print("\nWood selection incomplete. Use Wood Menu to set species and size.")
        return False

    if not wood_selection.get("event"):
        print("\nEvent not selected. Use Wood Menu -> Select event (SB/UH) or Main Menu option 3.")
        return False

    return True


def view_handicaps(heat_assignment_df: pd.DataFrame, wood_selection: Dict) -> None:
    """Calculate and display AI-enhanced handicap marks for the heat.

    This function:
    1. Validates event code
    2. Loads historical results data
    3. Calculates AI-enhanced handicaps
    4. Displays handicap marks with predictions and confidence levels
    5. Optionally runs Monte Carlo simulation for fairness validation

    Args:
        heat_assignment_df: Competitors in heat
        wood_selection: Wood characteristics and event code
    """
    if heat_assignment_df.empty:
        print("No competitors in heat assignment.")
        return

    event_code = wood_selection.get("event")
    if event_code not in ("SB", "UH"):
        print("Invalid or missing event code. Use Wood Menu to select SB or UH.")
        return

    # Load historical results data
    results_df = load_results_df()

    if results_df.empty:
        print("\nNo historical data available. Cannot generate AI predictions.")
        print("Please ensure competitors have historical times entered in the Results sheet.")
        return

    # Get wood selection parameters with defaults
    species = wood_selection.get("species", "Unknown")
    diameter = wood_selection.get("size_mm", 300)
    quality = wood_selection.get("quality", 5)

    # Ensure quality is an integer
    if quality is None:
        quality = 5

    # Calculate AI-enhanced handicaps
    results = calculate_ai_enhanced_handicaps(
        heat_assignment_df,
        species,
        diameter,
        quality,
        event_code,
        results_df
    )

    if not results:
        print("\nUnable to generate handicap marks. Please check historical data.")
        return

    # Display compact results
    print("\n" + "="*70)
    print("CALCULATED HANDICAP MARKS")
    print("="*70)

    for result in results:
        # Pull explanation generated from model
        explanation_text = result['explanation']
        # Find the part after the predicted time
        if '(' in explanation_text:
            data_info = explanation_text[explanation_text.find('(')+1:]
            if ')' in data_info:
                data_info = data_info[:data_info.rfind(')')]
            else:
                data_info = explanation_text
        else:
            data_info = explanation_text

        print(f"{result['name']:25s} Mark {result['mark']:3d}  ({result['predicted_time']:.1f}s predicted) ({data_info}) [Confidence: {result['confidence']}]")

    # Display wood selection
    print("\n" + "="*70)
    print(f"Selected Wood -> Species: {species}, "
          f"Diameter: {diameter} mm, "
          f"Quality: {quality}")
    print(f"Event: {event_code}")
    print("="*70)

    # Offer Monte Carlo simulation
    print("\nWould you like to run a Monte Carlo simulation to validate these handicaps?")
    print("This will simulate 250,000 races to assess fairness.")
    choice = input("Run simulation? (y/n): ").strip().lower()

    if choice == 'y':
        simulate_and_assess_handicaps(results, num_simulations=250000)


def append_results_to_excel(heat_assignment_df: Optional[pd.DataFrame] = None,
                           wood_selection: Optional[Dict] = None,
                           round_object: Optional[Dict] = None,
                           tournament_state: Optional[Dict] = None) -> None:
    """Append heat results to Excel Results sheet.

    This function supports both:
    - NEW TOURNAMENT SYSTEM: Uses round_object and tournament_state
    - LEGACY SINGLE-HEAT SYSTEM: Uses heat_assignment_df

    Collects:
    1. Raw cutting times (for historical data)
    2. Finish order (who finished first in real-time, including handicap)

    Args:
        heat_assignment_df: LEGACY - competitors in heat (for backward compatibility)
        wood_selection: Wood characteristics
        round_object: NEW - Round object from tournament system (optional)
        tournament_state: NEW - Tournament state for context (optional)
    """

    # Determine if using new tournament system or legacy single-heat system
    if round_object is not None:
        # NEW TOURNAMENT SYSTEM
        competitors_list = round_object['competitors']
        round_name = round_object['round_name']
    else:
        # LEGACY SINGLE-HEAT SYSTEM
        if heat_assignment_df is None or heat_assignment_df.empty:
            print("No competitors in heat assignment.")
            return
        competitors_list = heat_assignment_df['competitor_name'].tolist()
        round_name = None

    event_code = wood_selection.get("event")
    if event_code not in ("SB", "UH"):
        print("Event not selected. Use Wood Menu -> Select event (SB/UH) or Main Menu option 3.")
        return

    species = wood_selection.get("species")
    size_mm = wood_selection.get("size_mm")
    quality = wood_selection.get("quality")

    # Generate HeatID
    if round_object and tournament_state:
        # NEW: Use round name and tournament context
        event_name = tournament_state.get('event_name', 'Event')
        heat_id = f"{event_code}-{event_name}-{round_name}".replace(" ", "-")
    else:
        # LEGACY: Prompt for Heat ID
        heat_id = input("Enter a Heat ID (e.g., SB-01-Qual or any short label): ").strip()
        if not heat_id:
            heat_id = f"{event_code}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    # Get ID/name mapping
    _, name_to_id = get_competitor_id_name_mapping()

    rows_to_write = []
    times_collected = {}

    # STEP 1: Collect raw cutting times (for historical data)
    print("\n" + "=" * 70)
    print("STEP 1: RECORD CUTTING TIMES")
    print("=" * 70)
    print("Enter the raw cutting time for each competitor (from their mark to block severed)")
    print("Press Enter to skip a competitor\n")

    for name in competitors_list:
        s = input(f"  Cutting time for {name}: ").strip()

        if s == "":
            continue

        try:
            t = float(s)
            times_collected[name] = t
        except ValueError:
            print("    Invalid time; skipping this entry.")
            continue

    if not times_collected:
        print("\nNo results to record.")
        return

    # STEP 2: Record finish order (WHO FINISHED FIRST in real-time)
    print("\n" + "=" * 70)
    print("STEP 2: RECORD FINISH ORDER")
    print("=" * 70)
    print("Enter the finish position for each competitor (1 = finished first, 2 = second, etc.)")
    print("This is based on when they physically severed their block (handicap delays included)\n")

    finish_order = {}
    for name in times_collected.keys():
        while True:
            pos_str = input(f"  Finish position for {name}: ").strip()
            try:
                position = int(pos_str)
                if position < 1:
                    print("    Position must be 1 or greater")
                    continue
                finish_order[name] = position
                break
            except ValueError:
                print("    Invalid position; please enter a number")

    # Prepare rows for Excel and update round_object
    timestamp = datetime.now().isoformat(timespec="seconds")

    for name, time_val in times_collected.items():
        competitor_id = name_to_id.get(str(name).strip().lower(), name)

        # Include Round and HeatID columns
        rows_to_write.append([competitor_id, event_code, time_val, size_mm, species, timestamp, round_name or "", heat_id])

        # Store in round_object if using tournament system
        if round_object is not None:
            round_object['actual_results'][name] = time_val
            if 'finish_order' not in round_object:
                round_object['finish_order'] = {}
            round_object['finish_order'][name] = finish_order.get(name, 999)

    if not rows_to_write:
        print("No results to write.")
        return

    try:
        # Load or create workbook
        try:
            wb = load_workbook(RESULTS_FILE)
        except Exception:
            wb = Workbook()
            if "Sheet" in wb.sheetnames and RESULTS_SHEET not in wb.sheetnames:
                wb.remove(wb["Sheet"])

        ws = detect_results_sheet(wb)

        # Ensure header exists with Excel column names (NEW: includes Round and HeatID)
        if ws.max_row == 0:
            ws.append(["CompetitorID", "Event", "Time (seconds)", "Size (mm)", "Species Code", "Date", "Round", "HeatID"])

        # Append rows
        for r in rows_to_write:
            ws.append(r)

        wb.save(RESULTS_FILE)
        wb.close()
        print("Results appended to Excel successfully.")

        # NEW: Update round status if using tournament system
        if round_object is not None:
            round_object['status'] = 'in_progress'  # Mark as in progress (not completed until advancers selected)

    except Exception as e:
        print(f"Error appending results: {e}")
