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
