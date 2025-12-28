"""Excel I/O functions for loading and saving woodchopping competition data."""

import pandas as pd
from openpyxl import load_workbook, Workbook
from typing import Tuple, Dict
from datetime import datetime

# Import config
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from config import paths


def get_competitor_id_name_mapping() -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    Load competitor data and return bidirectional ID/name mapping dictionaries.

    Returns:
        Tuple of (id_to_name dict, name_to_id dict)
    """
    try:
        df = pd.read_excel(paths.EXCEL_FILE, sheet_name=paths.COMPETITOR_SHEET)

        if df.empty:
            return {}, {}

        id_to_name = {}
        name_to_id = {}

        for _, row in df.iterrows():
            comp_id = str(row.get('CompetitorID', '')).strip()
            name = str(row.get('Name', '')).strip()

            if comp_id and name:
                id_to_name[comp_id] = name
                name_to_id[name.lower()] = comp_id

        return id_to_name, name_to_id

    except Exception as e:
        print(f"Error loading competitor ID/name mapping: {e}")
        return {}, {}


def load_competitors_df() -> pd.DataFrame:
    """
    Load the competitor roster from Excel into a DataFrame.

    Returns:
        DataFrame with competitor information (name, country, ID, state, gender)
    """
    try:
        df = pd.read_excel(paths.EXCEL_FILE, sheet_name=paths.COMPETITOR_SHEET)

        # Standardize column names
        column_mapping = {
            'Name': 'competitor_name',
            'Country': 'competitor_country',
            'CompetitorID': 'competitor_id',
            'State/Province': 'state_province',
            'Gender': 'gender'
        }
        df = df.rename(columns=column_mapping)

        if df.empty:
            print("No competitors found in Excel. Please add competitors first.")
            return pd.DataFrame(columns=["competitor_name", "competitor_country"])

        print(f"Roster loaded successfully from Excel. Found {len(df)} competitors.")
        return df

    except FileNotFoundError:
        print(f"Excel file '{paths.EXCEL_FILE}' not found. Creating new file.")
        # Create the Excel file with proper sheets
        wb = Workbook()
        if "Sheet" in wb.sheetnames:
            wb.remove(wb["Sheet"])
        ws = wb.create_sheet(paths.COMPETITOR_SHEET)
        ws.append(["CompetitorID", "Name", "Country", "State/Province", "Gender"])
        wb.save(paths.EXCEL_FILE)
        wb.close()
        return pd.DataFrame(columns=["competitor_name", "competitor_country"])
    except Exception as e:
        print(f"Error loading roster from Excel: {e}")
        print(f"Looking for sheet: {paths.COMPETITOR_SHEET}")
        return pd.DataFrame(columns=["competitor_name", "competitor_country"])


def load_wood_data() -> pd.DataFrame:
    """
    Load wood species data from Excel into a DataFrame.

    Returns:
        DataFrame with wood properties (species, multiplier, janka, etc.)
    """
    try:
        df = pd.read_excel(paths.EXCEL_FILE, sheet_name=paths.WOOD_SHEET)
        return df
    except Exception as e:
        print(f"Error loading wood data: {e}")
        return pd.DataFrame(columns=["species", "multiplier"])


# Cache for species name lookups to avoid repeated Excel reads
_species_cache = {}

def get_species_name_from_code(species_code: str) -> str:
    """
    Convert species code (e.g., 'S01') to species name (e.g., 'Eastern White Pine').

    Uses caching to avoid repeated Excel reads.

    Args:
        species_code: Species code like 'S01', 'S02', etc.

    Returns:
        str: Species name, or the code itself if not found
    """
    # Check cache first
    if species_code in _species_cache:
        return _species_cache[species_code]

    try:
        wood_df = pd.read_excel(paths.EXCEL_FILE, sheet_name=paths.WOOD_SHEET)

        if 'speciesID' in wood_df.columns and 'species' in wood_df.columns:
            # Build cache for all species
            for _, row in wood_df.iterrows():
                code = row['speciesID']
                name = row['species']
                if pd.notna(code) and pd.notna(name):
                    _species_cache[code] = name

            # Return from cache if found
            if species_code in _species_cache:
                return _species_cache[species_code]

        # Fallback: return the code itself and cache it
        _species_cache[species_code] = species_code
        return species_code

    except Exception as e:
        print(f"Error looking up species name: {e}")
        # Cache the fallback value
        _species_cache[species_code] = species_code
        return species_code


def load_results_df() -> pd.DataFrame:
    """
    Load the Results sheet as a DataFrame with competitor names (not IDs).

    Returns:
        DataFrame with historical results including competitor names
    """
    try:
        # Read raw results
        df = pd.read_excel(paths.EXCEL_FILE, sheet_name=paths.RESULTS_SHEET)

        if df.empty:
            print("No results found in Excel.")
            return pd.DataFrame()

        # Get ID to name mapping
        id_to_name, _ = get_competitor_id_name_mapping()

        # Map column names
        column_mapping = {
            'CompetitorID': 'competitor_id',
            'Event': 'event',
            'Time (seconds)': 'raw_time',
            'Size (mm)': 'size_mm',
            'Species Code': 'species',
            'Date': 'date',
            'Quality': 'quality',
            'HeatID': 'heat_id'
        }
        df = df.rename(columns=column_mapping)

        # Convert IDs to names
        if 'competitor_id' in df.columns:
            df['competitor_name'] = df['competitor_id'].apply(
                lambda x: id_to_name.get(str(x).strip(), f"Unknown_{x}")
            )

        # Parse dates to datetime objects (critical for time-decay weighting)
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            # Note: errors='coerce' converts invalid/missing dates to NaT (Not a Time)
            # This maintains backward compatibility with results that have no dates

        return df

    except FileNotFoundError:
        print(f"Results sheet not found in '{paths.EXCEL_FILE}'.")
        return pd.DataFrame()
    except Exception as e:
        print(f"Error loading results: {e}")
        return pd.DataFrame()


def detect_results_sheet(wb: Workbook):
    """
    Return the Results sheet from workbook; create it with proper headers if missing.

    Args:
        wb: openpyxl Workbook object

    Returns:
        Results worksheet
    """
    if paths.RESULTS_SHEET in wb.sheetnames:
        return wb[paths.RESULTS_SHEET]
    else:
        # Create Results sheet with proper headers
        ws = wb.create_sheet(paths.RESULTS_SHEET)
        ws.append([
            "CompetitorID",
            "Event",
            "Time (seconds)",
            "Size (mm)",
            "Species Code",
            "Quality",
            "HeatID",
            "Date"
        ])
        print(f"Created '{paths.RESULTS_SHEET}' sheet with headers.")
        return ws


def save_time_to_results(
    event: str,
    name: str,
    species: str,
    size: float,
    quality: int,
    time: float,
    heat_id: str,
    timestamp: str
) -> None:
    """
    Save a single time entry to the results sheet.

    Args:
        event: Event code (SB or UH)
        name: Competitor name
        species: Wood species code
        size: Diameter in mm
        quality: Wood quality (0-10)
        time: Time in seconds
        heat_id: Heat identifier
        timestamp: Date/time string
    """
    _, name_to_id = get_competitor_id_name_mapping()

    competitor_id = name_to_id.get(name.lower())
    if not competitor_id:
        print(f"Warning: Could not find ID for {name}")
        return

    try:
        wb = load_workbook(paths.EXCEL_FILE)
        ws = detect_results_sheet(wb)

        ws.append([
            competitor_id,
            event,
            time,
            size,
            species,
            quality,
            heat_id,
            timestamp
        ])

        wb.save(paths.EXCEL_FILE)
        wb.close()

    except Exception as e:
        print(f"Error saving time to results: {e}")


def append_results_to_excel(heat_assignment_df, wood_selection, round_object=None, tournament_state=None):
    """
    Append heat results to Excel Results sheet.

    Supports both legacy single-heat system and new multi-round tournament system.

    Args:
        heat_assignment_df (DataFrame): LEGACY - competitors in heat (for backward compatibility)
        wood_selection (dict): Wood characteristics (species, size_mm, quality, event)
        round_object (dict): NEW - Round object from tournament system (optional)
        tournament_state (dict): NEW - Tournament state for context (optional)
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
        rows_to_write.append([
            competitor_id,
            event_code,
            time_val,
            size_mm,
            species,
            timestamp,
            round_name or "",
            heat_id
        ])

        # Store in round_object if using tournament system
        if round_object is not None:
            if 'actual_results' not in round_object:
                round_object['actual_results'] = {}
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
            wb = load_workbook(paths.EXCEL_FILE)
        except Exception:
            wb = Workbook()
            if "Sheet" in wb.sheetnames and paths.RESULTS_SHEET not in wb.sheetnames:
                wb.remove(wb["Sheet"])

        ws = detect_results_sheet(wb)

        # Ensure header exists with Excel column names (includes Round and HeatID)
        if ws.max_row == 0:
            ws.append([
                "CompetitorID",
                "Event",
                "Time (seconds)",
                "Size (mm)",
                "Species Code",
                "Date",
                "Round",
                "HeatID"
            ])

        # Append rows
        for r in rows_to_write:
            ws.append(r)

        wb.save(paths.EXCEL_FILE)
        wb.close()
        print("Results appended to Excel successfully.")

        # NEW: Update round status if using tournament system
        if round_object is not None:
            round_object['status'] = 'in_progress'  # Mark as in progress (not completed until advancers selected)

    except Exception as e:
        print(f"Error appending results: {e}")
