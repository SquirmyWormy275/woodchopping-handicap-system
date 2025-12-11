#Import Pandas, numpy, 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
import json
import statistics
import textwrap
from math import ceil
from datetime import datetime
from openpyxl import load_workbook, Workbook


##file/sheet names (saved in the same directory as this script!!!!!!!!!!!!!!!!!!!!!!!!)
COMPETITOR_FILE  = "woodchopping.xlsx"
COMPETITOR_SHEET = "Competitor"  
WOOD_FILE        = "woodchopping.xlsx"
WOOD_SHEET       = "wood"
RESULTS_FILE     = "woodchopping.xlsx"
RESULTS_SHEET    = "Results"  

# CORE HELPER FUNCTIONS

"""Load competitor data and return two dictionaries: id_to_name and name_to_id
    All Ids and names are stored on the 'Competitor' sheet in the excel file."""

def get_competitor_id_name_mapping():
    
    try:
        df = pd.read_excel(COMPETITOR_FILE, sheet_name=COMPETITOR_SHEET)
        
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


##Load the roster from Excel into a DataFrame.
def load_competitors_df():
    try:
        df = pd.read_excel(COMPETITOR_FILE, sheet_name=COMPETITOR_SHEET)
        
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
        print(f"Excel file '{COMPETITOR_FILE}' not found. Creating new file.")
        # Create the Excel file with proper sheets
        wb = Workbook()
        if "Sheet" in wb.sheetnames:
            wb.remove(wb["Sheet"])
        ws = wb.create_sheet(COMPETITOR_SHEET)
        ws.append(["CompetitorID", "Name", "Country", "State/Province", "Gender"])
        wb.save(COMPETITOR_FILE)
        wb.close()
        return pd.DataFrame(columns=["competitor_name", "competitor_country"])
    except Exception as e:
        print(f"Error loading roster from Excel: {e}")
        print(f"Looking for sheet: {COMPETITOR_SHEET}")
        return pd.DataFrame(columns=["competitor_name", "competitor_country"])


## Load wood speciess data from excel into a DataFrame.
def load_wood_data():
    try:
        df = pd.read_excel(WOOD_FILE, sheet_name=WOOD_SHEET)
        print("Wood data loaded successfully.")
        return df
    except Exception as e:
        print(f"Error loading wood data: {e}")
        return pd.DataFrame(columns=["species", "multiplier"])


def load_results_df():
    """Load the Results sheet as a DataFrame; returns empty DataFrame if missing."""
    try:
        df = pd.read_excel(RESULTS_FILE, sheet_name=RESULTS_SHEET)
        df.columns = [str(c).strip().lower() for c in df.columns]
        
        # Map Excel column names to expected names
        column_mapping = {
            'competitorid': 'competitor_id',
            'time (seconds)': 'raw_time',
            'size (mm)': 'size_mm',
            'species code': 'species'
        }
        df = df.rename(columns=column_mapping)
        
        # Convert competitor IDs to names
        if 'competitor_id' in df.columns:
            id_to_name, _ = get_competitor_id_name_mapping()
            df['competitor_name'] = df['competitor_id'].apply(
                lambda x: id_to_name.get(str(x).strip(), str(x))
            )
        
        return df
    except Exception:
        return pd.DataFrame(columns=["event", "competitor_name", "species", "size_mm", 
                                    "quality", "raw_time", "heat_id", "timestamp"])


#Menu Option 1: Competitor Selection Menu
''' Official will be presented with a list of competitors

    Definitions:
    'Roster'- list of all competitors available in the excel sheet
    'Heat Assignment'- list of competitors selected for the current heat

    1. Select Competitors for Heat from the roster
        a. Reload roster from Excel to ensure we have latest data
        b. Competitors will be displayed with index numbers for selection
        C. Selected competitors will be added to heat assignment list
    2. Add competitors to the roster
        a. Prompt for competitor details (name, country)
        b. Input historcical times for handicap calculation (3x)
            -Event (UH/SB)
            -Time
            -Wood species
            -Size in mm
            -Date (optional)
    3. View Heat Assignment
    -While this is functionally similar to viewing the heat after selecting competitors,
    this allows the judge to review the heat at any time.
    4. Remove competitors from the heat assignment
    -Self Explanatory; this does NOT remove them from the roster, only from the current heat assignment
    5. Return to Main Menu
'''

## Competitor Selection Menu
def competitor_menu(comp_df, heat_assignment_df, heat_assignment_names):

    while True:
        print("\n--- Competitor Menu ---")
        print("1) Select Competitors for Heat from roster")
        print("2) Add new competitor to roster")
        print("3) View Heat Assignment")
        print("4) Remove competitor from Heat Assignment")
        print("5) Back to Main Menu")
        s = input("Choose an option: ").strip()

        if s == "1":
            # Reload roster from Excel to ensure we have latest data
            comp_df = load_competitors_df()
            
            # Select competitors and RETURN TO MAIN MENU
            heat_assignment_df, heat_assignment_names = select_competitors_for_heat(comp_df)
            return comp_df, heat_assignment_df, heat_assignment_names

        elif s == "2":
            # Add new competitor to roster with historical times
            comp_df = add_competitor_with_times()
            # Stay in competitor menu after adding

        elif s == "3":
            # View current heat assignment
            view_heat_assignment(heat_assignment_df, heat_assignment_names)
            # Stay in competitor menu after viewing

        elif s == "4":
            # Remove from heat assignment (not from roster)
            heat_assignment_df, heat_assignment_names = remove_from_heat(heat_assignment_df, heat_assignment_names)
            # Stay in competitor menu after removing

        elif s == "5" or s == "":
            break
        else:
            print("Invalid selection. Try again.")
    
    return comp_df, heat_assignment_df, heat_assignment_names


def select_competitors_for_heat(comp_df):

    """Select competitors from roster to add to heat assignment will display the names of all competitors available in the excel sheet.
    -All competitors will have an index number (different from the competitor ID that is on the excel shet!) assigned to them for easy selection
    -The Judge will enter a competitor's index number to select them for the heat
    -Selected competitors will be added to a separate list for the heat
    -Slected competitors' names will be displayed after selection is complete
    -The judge will be prompted to hit enter on an empty entry to finalize list
"""
    #Obligatory checks

    if comp_df is None or comp_df.empty:
        print("\nRoster is empty. Please add competitors first.")
        input("Press Enter to continue...")
        return pd.DataFrame(), []
    
    if "competitor_name" not in comp_df.columns:
        print("Roster missing 'competitor_name' column.")
        input("Press Enter to continue...")
        return pd.DataFrame(), []
    
    # Display roster with index numbers
    print("\n--- ROSTER (All Available Competitors) ---")
    print("-" * 40)
    for idx in range(len(comp_df)):
        row = comp_df.iloc[idx]
        name = row.get("competitor_name", "Unknown")
        country = row.get("competitor_country", "Unknown")
        print(f"{idx + 1}) {name} ({country})")
    
    print("\n" + "=" * 40)
    print("INSTRUCTIONS:")
    print("- Enter competitor numbers one at a time")
    print("- Press Enter after each number to add them")
    print("- Press Enter with no input when finished")
    print("=" * 40)
    
    selected_indices = []
    selected_names = []
    
    while True:
        selection = input(f"\nEnter competitor number (or press Enter to finish): ").strip()
        
        if selection == "":
            break
        
        try:
            idx = int(selection) - 1
            if 0 <= idx < len(comp_df):
                if idx not in selected_indices:
                    selected_indices.append(idx)
                    name = comp_df.iloc[idx]["competitor_name"]
                    selected_names.append(name)
                    print(f"✓ {name} added to heat")
                else:
                    print("Competitor already selected")
            else:
                print("Invalid number. Please try again.")
        except ValueError:
            print("Please enter a valid number")
    
    if not selected_indices:
        print("\nNo competitors selected.")
        input("Press Enter to return to main menu...")
        return pd.DataFrame(), []
    
    heat_df = comp_df.iloc[selected_indices].copy()
    
    print(f"\n✓ Total {len(selected_names)} competitors added to heat assignment:")
    for name in selected_names:
        print(f"  - {name}")
    
    input("\nPress Enter to return to main menu...")
    return heat_df, selected_names

# Display current heat assignment. This will display the names of all competitors currently selected for the heat
    

def view_heat_assignment(heat_df, heat_names):
    print("\n--- CURRENT HEAT ASSIGNMENT ---")
    print("-" * 40)
    
    if not heat_names or heat_df.empty:
        print("No competitors currently assigned to heat.")
    else:
        for i, name in enumerate(heat_names, 1):
            country = heat_df[heat_df["competitor_name"] == name]["competitor_country"].values
            country_str = country[0] if len(country) > 0 else "Unknown"
            print(f"{i}) {name} ({country_str})")
        
        print(f"\nTotal: {len(heat_names)} competitors in heat")
    
    input("\nPress Enter to continue...")

# Remove competitor from heat assignment (not from roster). This will allow the judge to remove competitors from the heat assignment list
def remove_from_heat(heat_df, heat_names):
  
    if not heat_names or heat_df.empty:
        print("\nHeat assignment is currently empty.")
        input("Press Enter to continue...")
        return heat_df, heat_names
    
    print("\n--- REMOVE FROM HEAT ASSIGNMENT ---")
    print("-" * 40)
    
    for i, name in enumerate(heat_names, 1):
        print(f"{i}) {name}")
    
    print("\nEnter number to remove or press Enter to cancel:")
    choice = input("Your choice: ").strip()
    
    if choice == "":
        print("No changes made.")
        input("Press Enter to continue...")
        return heat_df, heat_names
    
    try:
        idx = int(choice) - 1
        if 0 <= idx < len(heat_names):
            removed_name = heat_names[idx]
            heat_names.remove(removed_name)
            heat_df = heat_df[heat_df["competitor_name"] != removed_name]
            print(f"\n✓ {removed_name} removed from heat assignment.")
        else:
            print("Invalid selection.")
    except ValueError:
        print("Invalid input.")
    
    input("Press Enter to continue...")
    return heat_df, heat_names


def add_competitor_with_times():
    """Add a new competitor to roster and prompt for historical times"""
    try:
        # Get competitor basic info
        print("\n--- Add New Competitor ---")
        name = input("Enter competitor name: ").strip()
        if not name:
            print("Name cannot be empty.")
            return load_competitors_df()
        
        country = input("Enter competitor country: ").strip()
        state = input("Enter state/province (optional): ").strip()
        gender = input("Enter gender (M/F, optional): ").strip().upper()
        
        # Add to competitors sheet
        try:
            wb = load_workbook(COMPETITOR_FILE)
        except FileNotFoundError:
            wb = Workbook()
            if "Sheet" in wb.sheetnames:
                wb.remove(wb["Sheet"])
        
        if COMPETITOR_SHEET not in wb.sheetnames:
            ws = wb.create_sheet(COMPETITOR_SHEET)
            ws.append(["CompetitorID", "Name", "Country", "State/Province", "Gender"])
        else:
            ws = wb[COMPETITOR_SHEET]
            if ws.max_row < 1:
                ws.append(["CompetitorID", "Name", "Country", "State/Province", "Gender"])
        
        # Check for duplicate
        existing_names = set()
        for row in ws.iter_rows(min_row=2, values_only=True):
            if row and len(row) > 1 and row[1]:  # Name is in column 2 (index 1)
                existing_names.add(str(row[1]).strip().lower())
        
        if name.lower() in existing_names:
            print("Competitor already exists in roster.")
            wb.close()
            return load_competitors_df()
        
        # Generate new CompetitorID
        new_id = f"C{str(ws.max_row).zfill(3)}"
        
        # Add competitor to sheet
        ws.append([new_id, name, country, state, gender])
        
        wb.save(COMPETITOR_FILE)
        wb.close()
        print(f"\n✓ {name} added to roster successfully with ID {new_id}")
        
        # Now prompt for historical times
        print("\n--- Enter Historical Competition Times ---")
        print("Minimum 3 times required for handicap calculation.")
        print("You can enter more than 3 if desired.")
        
        add_historical_times_for_competitor(name)
        
        return load_competitors_df()
        
    except Exception as e:
        print(f"Error adding competitor: {e}")
        return load_competitors_df()
    

def add_historical_times_for_competitor(competitor_name):
    """Prompt for and save historical times to results sheet
    Judge will be prompted to select whether each time is for SB or UH
    Judge will be prompted to enter the time, wood species, size, and date (optional)
    The program will store this data in the results sheet
    """
    times_added = 0
    
    while True:
        if times_added < 3:
            print(f"\n--- Historical Time Entry {times_added + 1} (minimum 3 required) ---")
        else:
            cont = input(f"\n{times_added} times entered. Add another? (y/n): ").strip().lower()
            if cont != 'y':
                break
            print(f"\n--- Historical Time Entry {times_added + 1} ---")
        
        # Get event type
        while True:
            event = input("Event type (SB for Standing Block, UH for Underhand): ").strip().upper()
            if event in ["SB", "UH"]:
                break
            print("Please enter SB or UH.")
        
        # Get time
        while True:
            time_str = input("Time in seconds (e.g., 45.3): ").strip()
            try:
                time_val = float(time_str)
                break
            except ValueError:
                print("Please enter a valid number.")
        
        # Get wood species
        species = input("Wood species: ").strip()
        
        # Get size
        while True:
            size_str = input("Wood diameter in mm: ").strip()
            try:
                size_val = float(size_str)
                break
            except ValueError:
                print("Please enter a valid number.")
        
        # Get quality (optional, default to 5)
        quality_str = input("Wood quality (0-10, press Enter for 5): ").strip()
        if quality_str:
            try:
                quality = int(quality_str)
                quality = max(0, min(10, quality))
            except:
                quality = 5
        else:
            quality = 5
        
        # Optional date
        date_str = input("Date (optional, format YYYY-MM-DD or press Enter to skip): ").strip()
        if not date_str:
            timestamp = datetime.now().isoformat(timespec="seconds")
        else:
            try:
                date_obj = datetime.strptime(date_str, "%Y-%m-%d")
                timestamp = date_obj.isoformat(timespec="seconds")
            except:
                timestamp = datetime.now().isoformat(timespec="seconds")
        
        # Save to results sheet
        save_time_to_results(event, competitor_name, species, size_val, quality, time_val, 
                           f"Historical-{event}", timestamp)
        times_added += 1
        print(f"✓ Time #{times_added} saved successfully")
    
    if times_added >= 3:
        print(f"\n✓ {times_added} historical times added for {competitor_name}.")
    else:
        print(f"\n⚠ Warning: Only {times_added} times added. Minimum 3 recommended for handicap calculation.")


def save_time_to_results(event, name, species, size, quality, time, heat_id, timestamp):
    """Helper to save a single time entry to results sheet"""
    try:
        # Convert name to ID
        _, name_to_id = get_competitor_id_name_mapping()
        competitor_id = name_to_id.get(str(name).strip().lower(), name)
        
        try:
            wb = load_workbook(RESULTS_FILE)
        except:
            wb = Workbook()
            if "Sheet" in wb.sheetnames:
                wb.remove(wb["Sheet"])
        
        if RESULTS_SHEET not in wb.sheetnames:
            ws = wb.create_sheet(RESULTS_SHEET)
            ws.append(["CompetitorID", "Event", "Time (seconds)", "Size (mm)", "Species Code", "Date"])
        else:
            ws = wb[RESULTS_SHEET]
            if ws.max_row == 0:
                ws.append(["CompetitorID", "Event", "Time (seconds)", "Size (mm)", "Species Code", "Date"])
        
        # Save with Excel's expected column names
        ws.append([competitor_id, event, time, size, species, timestamp])
        wb.save(RESULTS_FILE)
        wb.close()
        
    except Exception as e:
        print(f"Error saving time to results: {e}")



# MENU OPTION 2: WOOD CHARACTERISTICS MENU

''' Official will be presented with a list of wood characteristics:
    1. Select Wood Species from the list
    2. Enter Size in mm
    3. Enter wood quality 
    4. Return to Main Menu

    This menu will allow the judge to select the characteristics of the wood block being used in the heat and store the 
    selection for handicap calculation.
    Wood species available will be loaded from the wood sheet in the excel file.
'''

## Wood Characteristics Menu 
def wood_menu(wood_selection):
    
    if "event" not in wood_selection:
        wood_selection["event"] = None

    while True:
        print("\n--- Wood Menu ---")
        print(f"Current: species={wood_selection.get('species')}, "
              f"size_mm={wood_selection.get('size_mm')}, "
              f"quality={wood_selection.get('quality')}")
        print("1) Select wood species")
        print("2) Enter size (mm)")
        print("3) Enter quality (0 for poor quality, 1-3 for soft wood, 4-7 for average firmness for species, 8-10 for above average firmness for species)")
        print("4) Back to Main Menu")
        s = input("Choose an option: ").strip()

        if s == "1":
            wood_selection = select_wood_species(wood_selection)

        elif s == "2":
            wood_selection = enter_wood_size_mm(wood_selection)

        elif s == "3":
            wood_selection = enter_wood_quality(wood_selection)

        elif s == "4" or s == "":
            break
        else:
            print("Invalid selection. Try again.")
    
    return wood_selection


##Display species list, accept numeric choice
def select_wood_species(wood_selection):
    """Select wood species"""
    wood_df = load_wood_data()
    
    if wood_df.empty:
        print("No wood data available.")
        return wood_selection
    
    if "species" not in wood_df.columns:
        print("Wood sheet missing 'species' column.")
        return wood_selection
    
    species_list = wood_df["species"].astype(str).tolist()
    
    print("\nAvailable wood species:")
    for i, sp in enumerate(species_list, start=1):
        print(f"{i}) {sp}")
    
    choice = input("Select species by number: ").strip()
    
    try:
        idx = int(choice) - 1
        if 0 <= idx < len(species_list):
            wood_selection["species"] = species_list[idx]
            format_wood(wood_selection)
        else:
            print("Invalid selection.")
    except ValueError:
        print("Invalid input.")
    
    return wood_selection


## Enter block size in mm
def enter_wood_size_mm(wood_selection):
    """Enter block size in mm"""
    size = input("Enter block diameter in mm: ").strip()
    
    try:
        val = float(size)
        wood_selection["size_mm"] = val
        format_wood(wood_selection)
    except ValueError:
        print("Invalid size input.")
    
    return wood_selection


##Prompt for quality (integer 0–10)
'''Higher quality means softer wood and a faster time. 
this needs to account for that because softer wood would favor the front/back marker differently. 
A "0" indicates wood barely suitable for competition.'''
def enter_wood_quality(wood_selection):
    """Enter wood quality rating as an integer 0–10"""
    while True:
        s = input("Enter wood quality (integer 0–10): ").strip()
        
        if s == "":
            print("No change made to wood quality.")
            break
        
        try:
            val = int(s)
            val = max(0, min(10, val))  # Clamp between 0 and 10
            wood_selection["quality"] = val
            format_wood(wood_selection)
            break
        except ValueError:
            print("Invalid input. Please enter an integer between 0 and 10.")
    
    return wood_selection


##Header that displays current wood selection
def format_wood(ws):
    """Return formatted header for wood selection"""
    s = ws.get("species", "—")
    d = ws.get("size_mm", "—")
    q = ws.get("quality", "—")
    header = f"Selected Wood -> Species: {s}, Diameter: {d} mm, Quality: {q}"
    print(f"Wood selection updated: {header}")
    return header


# MENU OPTION 3: SELECT EVENT (SB/UH)
''' Official will be presented with two event codes to select from either SB or UH
Prompt for event code and store as SB or UH. No other values allowed.'''

def select_event_code(wood_selection):
    while True:
        e = input("Select event: type 'SB' for Standing Block or 'UH' for Underhand: ").strip().upper()
        
        if e in ("SB", "UH"):
            wood_selection["event"] = e
            print(f"Event selected: {e}")
            return wood_selection
        
        if e == "":
            print("No change made to event.")
            return wood_selection
        
        print("Invalid input. Please enter SB or UH.")


# AI INTEGRATION - OLLAMA FUNCTIONS (USED BY MENU 4)

#Call in OLLAMA (qwen2.5:7b)
def call_ollama(prompt, model="qwen2.5:7b"):
    
    try:
        response = requests.post(
            'http://localhost:11434/api/generate',
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.3,  # temperature: 0.3 = Low creativity. 
                    "num_predict": 50    # Limit response length for speed
                }
            },
            timeout=30
        )
        
        if response.status_code == 200:
            return response.json()['response'].strip()
        else:
            print(f"Ollama error: {response.status_code}")
            return None
            
    except requests.exceptions.ConnectionError:
        print("\nError: Cannot connect to Ollama. Make sure it's running:")
        print("  Run 'ollama serve' in a terminal")
        return None
    except Exception as e:
        print(f"\nError calling Ollama: {e}")
        return None




def get_competitor_historical_times_flexible(competitor_name, species, event_code, results_df):
  
    if results_df is None or results_df.empty:
        return [], "no data available"
    
    # Match competitor and event (required)
    name_match = results_df["competitor_name"].astype(str).str.strip().str.lower() == str(competitor_name).strip().lower()
    event_match = results_df["event"].astype(str).str.strip().str.upper() == event_code
    
    # Try exact species match first
    if species and "species" in results_df.columns:
        species_match = results_df["species"].astype(str).str.strip().str.lower() == str(species).strip().lower()
        exact_matches = results_df[name_match & event_match & species_match]
        
        times = []
        for _, row in exact_matches.iterrows():
            time = row.get('raw_time')
            if time is not None and time > 0:
                times.append(time)
        
        if times:
            return times, f"on {species} (exact match)"
    
    # Fallback: any species for this competitor and event
    any_species_matches = results_df[name_match & event_match]
    times = []
    for _, row in any_species_matches.iterrows():
        time = row.get('raw_time')
        if time is not None and time > 0:
            times.append(time)
    
    if times:
        return times, "on various wood types"
    
    return [], "no competitor history found"


#Calcualte Baseline

'''Calculate baseline with cascading fallback.

Tries in order:
1. Species + diameter range + event
2. Diameter range + event (any species)
3. Event only (any species, any diameter)
'''
    

def get_event_baseline_flexible(species, diameter, event_code, results_df):
  
    if results_df is None or results_df.empty:
        return None, "no data available"
    
    event_match = results_df["event"].astype(str).str.strip().str.upper() == event_code
    
    # Try species + diameter range + event
    if species and "species" in results_df.columns and "size_mm" in results_df.columns:
        species_match = results_df["species"].astype(str).str.strip().str.lower() == str(species).strip().lower()
        diameter_match = (results_df["size_mm"] >= diameter - 25) & (results_df["size_mm"] <= diameter + 25)
        
        exact_matches = results_df[species_match & diameter_match & event_match]
        times = []
        for _, row in exact_matches.iterrows():
            time = row.get('raw_time')
            if time is not None and time > 0:
                times.append(time)
        
        if len(times) >= 3:
            return statistics.mean(times), f"species/size average ({len(times)} performances)"
    
    # Fallback: diameter range + event (any species)
    if "size_mm" in results_df.columns:
        diameter_match = (results_df["size_mm"] >= diameter - 25) & (results_df["size_mm"] <= diameter + 25)
        size_matches = results_df[diameter_match & event_match]
        
        times = []
        for _, row in size_matches.iterrows():
            time = row.get('raw_time')
            if time is not None and time > 0:
                times.append(time)
        
        if len(times) >= 3:
            return statistics.mean(times), f"size average ({len(times)} performances)"
    
    # Final fallback: event only (all data for this event type)
    event_only = results_df[event_match]
    times = []
    for _, row in event_only.iterrows():
        time = row.get('raw_time')
        if time is not None and time > 0:
            times.append(time)
    
    if len(times) >= 3:
        return statistics.mean(times), f"event average ({len(times)} performances)"
    
    return None, "insufficient data"


#AI predicted Handicaps:

    
''' Predict competitor's time using historical data + LLM reasoning for quality adjustment.
    Now with improved fallback logic for sparse data.
    
'''

def predict_competitor_time_with_ai(competitor_name, species, diameter, quality, event_code, results_df):
    ''' Predict competitor's time using historical data + LLM reasoning for quality adjustment.'''
   
    # Step 1: Get historical data
    historical_times, data_source = get_competitor_historical_times_flexible(
        competitor_name, species, event_code, results_df
    )
    
    # Step 2: Calculate baseline
    if len(historical_times) >= 3:
        weights = [1.5, 1.3, 1.1, 1.0, 0.9, 0.8]
        weighted_times = [t * w for t, w in zip(historical_times[:6], weights[:len(historical_times)])]
        baseline = sum(weighted_times) / sum(weights[:len(historical_times)])
        confidence = "HIGH"
        explanation_source = f"competitor history {data_source}"
        
    elif len(historical_times) > 0:
        baseline = statistics.mean(historical_times)
        confidence = "MEDIUM"
        explanation_source = f"limited competitor history {data_source}"
        
    else:
        baseline, baseline_source = get_event_baseline_flexible(species, diameter, event_code, results_df)
        
        if baseline:
            confidence = "LOW"
            explanation_source = f"{baseline_source} (no competitor history)"
        else:
            if diameter >= 350:
                baseline = 60.0
            elif diameter >= 300:
                baseline = 50.0
            elif diameter >= 275:
                baseline = 45.0
            else:
                baseline = 40.0
            
            confidence = "VERY LOW"
            explanation_source = "estimated from size (no historical data)"
    
    # LOAD ACTUAL WOOD DATA FROM EXCEL
    wood_df = load_wood_data()
    
    # Format wood species data for AI
    wood_data_text = ""
    if wood_df is not None and not wood_df.empty:
        wood_data_text = "\nAVAILABLE WOOD SPECIES DATABASE:\n"
        for _, row in wood_df.iterrows():
            species_name = row.get('species', 'Unknown')
            wood_data_text += f"  - {species_name}"
            if 'hardness_category' in row:
                wood_data_text += f": Category={row.get('hardness_category', 'N/A')}"
            if 'base_adjustment_pct' in row:
                wood_data_text += f", Base Adjustment={row.get('base_adjustment_pct', 0):+.1f}%"
            if 'description' in row:
                wood_data_text += f", Description: {row.get('description', '')}"
            wood_data_text += "\n"
    
    # Step 3: AI prediction prompt
    
    prompt = f"""You are a master woodchopping handicapper making precision time predictions for competition.

HANDICAPPING OBJECTIVE

Your prediction must account for wood characteristics and competitor ability to create fair handicaps.
When handicaps are applied, all competitors should finish simultaneously if your predictions are accurate.
This requires deep understanding of how wood properties affect cutting times.

COMPETITOR PROFILE

Name: {competitor_name}
Baseline Time: {baseline:.1f} seconds
Data Source: {explanation_source}
Confidence Level: {confidence}

BASELINE INTERPRETATION:
- This baseline assumes QUALITY 5 wood (average hardness)
- Your task is to adjust this baseline for the ACTUAL quality rating
- Historical data already accounts for competitor's skill level and typical conditions

WOOD SPECIFICATIONS

Species: {species}
Diameter: {diameter:.0f}mm
Quality Rating: {quality}/10
Event Type: {event_code}

WOOD CHARACTERISTICS DATABASE
{wood_data_text}

QUALITY RATING SYSTEM

Quality measures wood condition on a 0-10 scale:

10 = Extremely soft/rotten
   - Wood breaks apart easily
   - Minimal resistance to axe
   - FASTEST possible cutting time
   - Reduces baseline time by approximately 10-15%

9 = Very soft (ideal competition wood)
   - Excellent cutting conditions
   - Clean grain, well-seasoned
   - Reduces baseline time by approximately 7-10%

8 = Soft
   - Good cutting conditions
   - Easy to work with
   - Reduces baseline time by approximately 5-7%

7 = Moderately soft
   - Better than average
   - Noticeable improvement over baseline
   - Reduces baseline time by approximately 3-5%

6 = Slightly soft
   - Marginally better than average
   - Minor improvement
   - Reduces baseline time by approximately 1-3%

5 = AVERAGE HARDNESS (BASELINE REFERENCE POINT)
   - This is what the baseline time assumes
   - NO ADJUSTMENT needed at quality 5
   - Standard competition wood

4 = Slightly hard
   - Marginally tougher than average
   - Minor slowdown
   - Increases baseline time by approximately 1-3%

3 = Moderately hard
   - Noticeably tougher
   - More resistance
   - Increases baseline time by approximately 3-5%

2 = Hard (difficult cutting)
   - Significant resistance
   - Green wood, tough grain
   - Increases baseline time by approximately 5-8%

1 = Very hard
   - Major difficulty
   - Knots, irregular grain
   - Increases baseline time by approximately 8-12%

0 = Extremely hard/barely suitable
   - Maximum difficulty
   - SLOWEST possible cutting time
   - Increases baseline time by approximately 12-15%

CURRENT SITUATION ANALYSIS

Your wood is quality {quality}, which is {abs(quality - 5)} point(s) {"ABOVE" if quality > 5 else "BELOW" if quality < 5 else "AT"} the baseline reference.

{"This wood is SOFTER than baseline - expect FASTER cutting time." if quality > 5 else "This wood is HARDER than baseline - expect SLOWER cutting time." if quality < 5 else "This wood is AVERAGE hardness - baseline time should be accurate."}

CALCULATION METHODOLOGY

Step 1: Start with baseline time: {baseline:.1f}s

Step 2: Apply species base adjustment (if available in database)
- Check species database above for {species}
- Apply the base adjustment percentage if listed

Step 3: Apply quality adjustment
- Calculate deviation from quality 5: {quality} - 5 = {quality - 5}
- Apply adjustment based on quality scale above
- Use the percentage ranges provided (1.5-2.5% per point as guideline)

Step 4: Consider wood physics
- Softer wood (quality >5): Cuts faster, less resistance
- Harder wood (quality <5): Cuts slower, more resistance
- Effect is roughly linear in the middle range (3-7)
- Effect accelerates at extremes (0-2 and 8-10)

Step 5: Validate against typical ranges for {diameter:.0f}mm diameter
- 275mm diameter: 35-50s typical range
- 300mm diameter: 40-55s typical range
- 325mm diameter: 45-60s typical range
- 350mm diameter: 50-70s typical range
- 375mm+ diameter: 60-90s typical range

CRITICAL FACTORS FOR FAIR HANDICAPPING

Front/Back Marker Dynamics:
- Softer wood (quality >5) disproportionately benefits slower competitors (front markers)
  * They gain more time than expected from easier cutting
  * Risk: Front marker finishes before back marker even starts
  
- Harder wood (quality <5) disproportionately penalizes slower competitors
  * They lose more time than expected from difficult cutting
  * Risk: Back marker wins by excessive margin

Your adjustment must account for this to maintain fair handicapping.

WOOD DENSITY AND SIZE INTERACTION

The {diameter:.0f}mm diameter creates a cutting area of approximately {3.14159 * (diameter/2)**2 / 10000:.2f} square cm.
- Larger diameter = exponentially more wood volume to remove
- Quality affects this proportionally: softer wood on large diameter saves significant time
- This diameter/quality interaction is already partially in baseline, but verify your adjustment makes sense

RESPONSE REQUIREMENT

Calculate the most accurate predicted time for {competitor_name} cutting {species} at quality {quality}.

CRITICAL: Respond with ONLY the predicted time as a decimal number.
- Example: 47.3
- NO units (like "seconds" or "s")
- NO explanations
- NO additional text
- JUST THE NUMBER

Predicted time:"""

    response = call_ollama(prompt)
    
    if response is None:
        quality_adjustment = (5 - quality) * 0.02
        predicted_time = baseline * (1 + quality_adjustment)
        explanation = f"Predicted {predicted_time:.1f}s ({explanation_source}, quality adjusted)"
        return predicted_time, confidence, explanation
    
    try:
        import re
        numbers = re.findall(r'\d+\.?\d*', response)
        if numbers:
            predicted_time = float(numbers[0])
            
            if baseline * 0.5 <= predicted_time <= baseline * 1.5:
                explanation = f"Predicted {predicted_time:.1f}s ({explanation_source}, AI quality adjusted)"
                return predicted_time, confidence, explanation
    except:
        pass
    
    explanation = f"Predicted {baseline:.1f}s ({explanation_source})"
    return baseline, confidence, explanation

def calculate_ai_enhanced_handicaps(heat_assignment_df, species, diameter, quality, event_code, results_df):
    """
    Calculate handicaps using AI-enhanced time predictions.
    
        heat_assignment_df (DataFrame): Competitors in heat
        species (str): Wood species
        diameter (float): Diameter in mm
        quality (int): Quality rating (0-10)
        event_code (str): Event type (SB or UH)
        results_df (DataFrame): Historical results data
    
    Returns:
        list: List of dicts with competitor info, predictions, and marks
    """
    results = []
    
    # Ensure quality is an integer
    if quality is None:
        quality = 5
    quality = int(quality)
    
    # Run predictions quietly (no intermediate printing)
    for _, row in heat_assignment_df.iterrows():
        comp_name = row.get("competitor_name")
        
        predicted_time, confidence, explanation = predict_competitor_time_with_ai(
            comp_name, species, diameter, quality, event_code, results_df
        )
        
        if predicted_time is None:
            continue
        
        results.append({
            'name': comp_name,
            'predicted_time': predicted_time,
            'confidence': confidence,
            'explanation': explanation
        })
    
    # Calculate marks
    if not results:
        return None
    
    # Sort by predicted time (slowest first)
    results.sort(key=lambda x: x['predicted_time'], reverse=True)
    
    # Slowest competitor gets mark 3
    slowest_time = results[0]['predicted_time']
    
    for result in results:
        gap = slowest_time - result['predicted_time']
        mark = 3 + int(gap + 0.999)  # Round up using ceiling logic
        
        # Apply 180-second maximum rule
        if mark > 183:
            mark = 183
        
        result['mark'] = mark
    
    return results



# MONTE CARLO SIMULATION FUNCTIONS (USED BY MENU 4)

""" Simulate a single race with realistic ABSOLUTE performance variation.

    competitors_with_marks (list): List of dicts with name, predicted_time, mark

Returns:
    list: Finish results sorted by finish time """


## Simulate a singular race with absolute variance: ±3 seconds standard deviation for everyone
'''3.0 Defines the standard deviation for time variation

~68% of the time, actual time will be within ±3 seconds of predicted
~95% of the time, actual time will be within ±6 seconds of predicted
This models real-world unpredictability'''

def simulate_single_race(competitors_with_marks):
   
    finish_results = []
    
    ABSOLUTE_VARIANCE = 3.0  # seconds
    
    for comp in competitors_with_marks:
        actual_time = np.random.normal(comp['predicted_time'], ABSOLUTE_VARIANCE)
        
        actual_time = max(actual_time, comp['predicted_time'] * 0.5)
        
        # Calculate finish time accounting for handicap
        start_delay = comp['mark'] - 3  # Mark 3 starts immediately
        finish_time = start_delay + actual_time
        
        finish_results.append({
            'name': comp['name'],
            'mark': comp['mark'],
            'actual_time': actual_time,
            'finish_time': finish_time,
            'predicted_time': comp['predicted_time']
        })
    
    # Sort by finish time
    finish_results.sort(key=lambda x: x['finish_time'])
    
    return finish_results


##Monte Carlo Simulation Function

'''Run Monte Carlo simulation to assess handicap fairness.'''

def run_monte_carlo_simulation(competitors_with_marks, num_simulations=250000):
  
    print("\n" + "="*70)
    print(f"RUNNING MONTE CARLO SIMULATION ({num_simulations} races)")
    print("="*70)
    print("Simulating races with ±3 second absolute performance variation...")
    
    # Track statistics
    finish_spreads = []
    winner_counts = {comp['name']: 0 for comp in competitors_with_marks}
    podium_counts = {comp['name']: 0 for comp in competitors_with_marks}  # Top 3
    finish_position_sums = {comp['name']: 0 for comp in competitors_with_marks}
    
    # Track front marker (slowest predicted, starts first)
    front_marker_name = max(competitors_with_marks, key=lambda x: x['predicted_time'])['name']
    back_marker_name = min(competitors_with_marks, key=lambda x: x['predicted_time'])['name']
    
    # Run simulations
    for i in range(num_simulations):
        if (i + 1) % 50000 == 0:
            print(f"  Completed {i + 1}/{num_simulations} simulations...")
        
        race_results = simulate_single_race(competitors_with_marks)
        
        # Calculate finish spread
        spread = race_results[-1]['finish_time'] - race_results[0]['finish_time']
        finish_spreads.append(spread)
        
        # Track winner
        winner_counts[race_results[0]['name']] += 1
        
        # Track podium (top 3)
        for j in range(min(3, len(race_results))):
            podium_counts[race_results[j]['name']] += 1
        
        # Track average finish positions
        for pos, result in enumerate(race_results, 1):
            finish_position_sums[result['name']] += pos
    
    # Calculate statistics
    avg_finish_positions = {name: pos_sum / num_simulations 
                           for name, pos_sum in finish_position_sums.items()}
    
    analysis = {
        'num_simulations': num_simulations,
        'finish_spreads': finish_spreads,
        'avg_spread': np.mean(finish_spreads),
        'median_spread': np.median(finish_spreads),
        'min_spread': np.min(finish_spreads),
        'max_spread': np.max(finish_spreads),
        'tight_finish_prob': sum(1 for s in finish_spreads if s < 10) / num_simulations,
        'very_tight_finish_prob': sum(1 for s in finish_spreads if s < 5) / num_simulations,
        'winner_counts': winner_counts,
        'winner_percentages': {name: (count / num_simulations * 100) 
                              for name, count in winner_counts.items()},
        'podium_counts': podium_counts,
        'podium_percentages': {name: (count / num_simulations * 100) 
                              for name, count in podium_counts.items()},
        'avg_finish_positions': avg_finish_positions,
        'front_marker_name': front_marker_name,
        'back_marker_name': back_marker_name,
        'front_marker_wins': winner_counts[front_marker_name],
        'back_marker_wins': winner_counts[back_marker_name],
        'competitors': competitors_with_marks
    }
    
    return analysis

#Generate summary of the analysis of the Monte Carlo simulation

def generate_simulation_summary(analysis):
    
    summary = []
    summary.append("\n" + "="*70)
    summary.append("MONTE CARLO SIMULATION RESULTS")
    summary.append("="*70)
    summary.append(f"Simulated {analysis['num_simulations']} races with ±3s absolute performance variation")
    summary.append("")
    
    summary.append("FINISH TIME SPREADS:")
    summary.append(f"  Average spread: {analysis['avg_spread']:.1f} seconds")
    summary.append(f"  Median spread:  {analysis['median_spread']:.1f} seconds")
    summary.append(f"  Range: {analysis['min_spread']:.1f}s - {analysis['max_spread']:.1f}s")
    summary.append(f"  Tight finish (<10s): {analysis['tight_finish_prob']*100:.1f}% of races")
    summary.append(f"  Very tight (<5s):    {analysis['very_tight_finish_prob']*100:.1f}% of races")
    summary.append("")
    
    summary.append("WIN PROBABILITIES:")
    sorted_winners = sorted(analysis['winner_percentages'].items(), 
                           key=lambda x: x[1], reverse=True)
    for name, pct in sorted_winners:
        summary.append(f"  {name:25s} {pct:5.1f}% ({analysis['winner_counts'][name]:4d} wins)")
    summary.append("")
    
    summary.append("AVERAGE FINISH POSITIONS:")
    sorted_positions = sorted(analysis['avg_finish_positions'].items(), 
                             key=lambda x: x[1])
    for name, avg_pos in sorted_positions:
        summary.append(f"  {name:25s} Avg position: {avg_pos:.2f}")
    summary.append("")
    
    summary.append("FRONT/BACK MARKER ANALYSIS:")
    summary.append(f"  Front marker (slowest): {analysis['front_marker_name']}")
    summary.append(f"    Win rate: {analysis['front_marker_wins']/analysis['num_simulations']*100:.1f}%")
    summary.append(f"  Back marker (fastest): {analysis['back_marker_name']}")
    summary.append(f"    Win rate: {analysis['back_marker_wins']/analysis['num_simulations']*100:.1f}%")
    summary.append("="*70)
    
    return "\n".join(summary)

#gENERATE cHART SHOWING WIN-RATES
def visualize_simulation_results(analysis):

    print("\n" + "="*70)
    print("WIN RATE VISUALIZATION")
    print("="*70)
    
    max_pct = max(analysis['winner_percentages'].values())
    
    sorted_winners = sorted(analysis['winner_percentages'].items(), 
                           key=lambda x: x[1], reverse=True)
    
    for name, pct in sorted_winners:
        bar_length = int((pct / max_pct) * 40)  # Scale to 40 chars max
        bar = "█" * bar_length
        print(f"{name:25s} {pct:5.1f}% {bar}")
    
    print("="*70)


#Assess the fairness of handicaps
''' Use LLM to provide expert assessment of handicap fairness. '''

def get_ai_assessment_of_handicaps(analysis):
    
    # Calculate fairness metrics
    max_win_rate = max(analysis['winner_percentages'].values())
    min_win_rate = min(analysis['winner_percentages'].values())
    win_rate_spread = max_win_rate - min_win_rate
    ideal_win_rate = 100.0 / len(analysis['competitors'])
    
    # Calculate per-competitor deviations
    win_rate_deviations = {}
    for name, pct in analysis['winner_percentages'].items():
        deviation = pct - ideal_win_rate
        win_rate_deviations[name] = deviation
    
    # Identify extremes
    most_favored = max(win_rate_deviations, key=win_rate_deviations.get)
    most_disadvantaged = min(win_rate_deviations, key=win_rate_deviations.get)
    
    # Format data for prompt
    winner_data = "\n".join([f"  - {name}: {pct:.2f}% win rate (deviation: {win_rate_deviations[name]:+.2f}%)" 
                            for name, pct in sorted(analysis['winner_percentages'].items(), 
                                                   key=lambda x: x[1], reverse=True)])
    
    competitor_details = "\n".join([f"  - {comp['name']}: {comp['predicted_time']:.1f}s predicted → Mark {comp['mark']}"
                                    for comp in sorted(analysis['competitors'], 
                                                      key=lambda x: x['predicted_time'], reverse=True)])
    
    win_rate_std_dev = np.std(list(analysis['winner_percentages'].values()))
    coefficient_of_variation = (win_rate_std_dev / ideal_win_rate) * 100 if ideal_win_rate > 0 else 0
    
    prompt = f"""You are a master woodchopping handicapper and statistician analyzing the fairness of predicted handicap marks through Monte Carlo simulation.

HANDICAPPING PRINCIPLES

PRIMARY GOAL: Create handicaps where ALL competitors have EQUAL probability of winning.
- In a fair handicap system, skill level should NOT predict victory
- A novice with Mark 3 should win as often as an expert with Mark 25
- The slowest competitor should have the same chance as the fastest

HANDICAPPING MECHANISM:
1. Predict each competitor's raw cutting time
2. Slowest predicted time receives Mark 3 (starts first)
3. Faster predicted times receive higher marks (delayed starts)
4. If predictions are perfect, everyone finishes simultaneously
5. Natural variation (±3s) creates competitive spread

QUALITY FACTORS IN PREDICTIONS:
- Wood species (hardness variations)
- Block diameter (volume to cut)
- Wood quality rating (0-10 scale, affects cutting speed)
- Historical competitor performance

SIMULATION METHODOLOGY

WHAT WE TESTED:
- Simulated {analysis['num_simulations']:,} races with {len(analysis['competitors'])} competitors
- Applied ±3 second ABSOLUTE performance variation (realistic race conditions)
- Variation represents: technique consistency, wood grain, fatigue, environmental conditions

WHY ABSOLUTE VARIANCE (±3s for everyone):
- Real factors affect all skill levels equally in absolute seconds
- Wood grain knot costs 2s for novice AND expert (not proportional to skill)
- Technique wobble affects everyone by similar absolute time
- This is a CRITICAL breakthrough in fair handicapping

STATISTICAL SIGNIFICANCE:
- With {analysis['num_simulations']:,} simulations, margin of error is extremely small
- Patterns in results are REAL, not random noise
- Even 1-2% win rate differences are statistically meaningful

SIMULATION RESULTS

COMPETITOR PREDICTIONS AND MARKS:
{competitor_details}

IDEAL WIN RATE: {ideal_win_rate:.2f}% per competitor
(Perfect handicapping means all competitors win exactly {ideal_win_rate:.2f}% of races)

ACTUAL WIN RATES:
{winner_data}

STATISTICAL MEASURES:
- Win Rate Spread: {win_rate_spread:.2f}% (maximum minus minimum)
- Standard Deviation: {win_rate_std_dev:.2f}%
- Coefficient of Variation: {coefficient_of_variation:.1f}%

FINISH TIME ANALYSIS:
- Average finish spread: {analysis['avg_spread']:.1f} seconds
- Median finish spread: {analysis['median_spread']:.1f} seconds
- Tight finishes (<10s): {analysis['tight_finish_prob']*100:.1f}% of races
- Very tight finishes (<5s): {analysis['very_tight_finish_prob']*100:.1f}% of races

FRONT AND BACK MARKER PERFORMANCE:
- Front Marker (slowest): {analysis['front_marker_name']} - {analysis['front_marker_wins']/analysis['num_simulations']*100:.1f}% wins
- Back Marker (fastest): {analysis['back_marker_name']} - {analysis['back_marker_wins']/analysis['num_simulations']*100:.1f}% wins

PATTERN IDENTIFICATION:
- Most Favored: {most_favored} ({analysis['winner_percentages'][most_favored]:.2f}%, +{win_rate_deviations[most_favored]:.2f}%)
- Most Disadvantaged: {most_disadvantaged} ({analysis['winner_percentages'][most_disadvantaged]:.2f}%, {win_rate_deviations[most_disadvantaged]:.2f}%)

FAIRNESS CRITERIA

RATING SCALE (based on win rate spread):

EXCELLENT (Spread ≤ 3%):
- All win rates within ±1.5% of ideal ({ideal_win_rate-1.5:.1f}% to {ideal_win_rate+1.5:.1f}%)
- Handicaps are nearly perfect
- Predictions are highly accurate
- No adjustments needed

VERY GOOD (Spread ≤ 6%):
- All win rates within ±3% of ideal ({ideal_win_rate-3:.1f}% to {ideal_win_rate+3:.1f}%)
- Handicaps are working well
- Minor prediction inaccuracies
- Only minor adjustments if desired

GOOD (Spread ≤ 10%):
- All win rates within ±5% of ideal ({ideal_win_rate-5:.1f}% to {ideal_win_rate+5:.1f}%)
- Acceptable fairness for competition
- Some prediction bias exists
- Consider adjustments for championship events

FAIR (Spread ≤ 16%):
- Win rates within ±8% of ideal
- Noticeable imbalance
- Predictions need refinement
- Adjustments recommended

POOR (Spread > 16%):
- Significant imbalance detected
- Predictions are systematically biased
- Handicaps require major adjustment
- Not suitable for fair competition

UNACCEPTABLE (Any competitor >2x or <0.5x ideal):
- Extreme bias detected
- One competitor has double (or half) expected win rate
- Fundamental prediction error
- Complete recalibration required

DIAGNOSTIC PATTERNS

COMMON ISSUES TO IDENTIFY:

1. FRONT MARKER ADVANTAGE (soft wood bias):
   Pattern: Front marker wins >ideal, back marker wins <ideal
   Cause: Predictions underestimate benefit of soft wood to slower competitors
   Fix: Increase quality adjustment for front markers on soft wood

2. BACK MARKER ADVANTAGE (hard wood bias):
   Pattern: Back marker wins >ideal, front marker wins <ideal
   Cause: Predictions underestimate difficulty of hard wood for slower competitors
   Fix: Increase time penalties for front markers on hard wood

3. MIDDLE COMPRESSION:
   Pattern: Extreme competitors (fastest/slowest) win less than middle competitors
   Cause: Predictions too conservative at extremes
   Fix: Increase handicap spread (widen gaps between marks)

4. EXPERIENCE BIAS:
   Pattern: Competitors with more historical data win more often
   Cause: Better predictions for experienced competitors
   Fix: Adjust confidence weighting or baseline calculations

5. SPECIES MISCALIBRATION:
   Pattern: Systematic bias across all competitors
   Cause: Species hardness factor incorrect
   Fix: Adjust species baseline percentage

YOUR ANALYSIS TASK

Provide a comprehensive assessment in the following structure:

1. FAIRNESS RATING: State one of: Excellent / Very Good / Good / Fair / Poor / Unacceptable

2. STATISTICAL ANALYSIS (2-3 sentences):
   - Interpret the win rate spread of {win_rate_spread:.2f}%
   - Comment on finish time spreads (average {analysis['avg_spread']:.1f}s)
   - Assess if variation is appropriate for exciting competition

3. PATTERN DIAGNOSIS (2-3 sentences):
   - Identify which diagnostic pattern (if any) is present
   - Explain WHY this pattern occurred based on competitor times
   - Reference specific competitors showing the bias

4. PREDICTION ACCURACY (1-2 sentences):
   - Are the predictions systematically biased or just slightly off?
   - Is the issue with one competitor or system-wide?

5. RECOMMENDATIONS (2-3 specific actions):
   If EXCELLENT or VERY GOOD: Affirm handicaps are ready for use
   If GOOD: Suggest optional refinements
   If FAIR, POOR, or UNACCEPTABLE: Provide specific adjustment recommendations
   
   Format recommendations as bullet points:
   • First specific action (include numbers when possible)
   • Second specific action
   • Final recommendation

RESPONSE REQUIREMENTS:
- Keep total response to 8-12 sentences maximum
- Be specific and actionable
- Use technical terms confidently
- Cite actual numbers from the data above
- Base analysis on ACTUAL DATA, not generic observations
- Reference specific competitors, percentages, and patterns you observe

Your Expert Assessment:"""

    response = call_ollama(prompt)
    
    if response:
        return response
    else:
        # Enhanced fallback assessment
        if win_rate_spread < 3:
            rating = "EXCELLENT"
            assessment = "Handicaps are nearly perfect. Predictions are highly accurate with minimal bias."
        elif win_rate_spread < 6:
            rating = "VERY GOOD"
            assessment = "Handicaps are working very well. Minor prediction variations are within acceptable range."
        elif win_rate_spread < 10:
            rating = "GOOD"
            assessment = "Handicaps are acceptable for competition. Some prediction refinement would improve fairness."
        elif win_rate_spread < 16:
            rating = "FAIR"
            assessment = "Noticeable imbalance detected. Predictions show systematic bias requiring adjustment."
        else:
            rating = "POOR"
            assessment = "Significant imbalance requiring major prediction recalibration."
        
        front_wins = analysis['front_marker_wins']/analysis['num_simulations']*100
        back_wins = analysis['back_marker_wins']/analysis['num_simulations']*100
        
        if front_wins > ideal_win_rate + 3:
            pattern = "Front marker advantage detected (soft wood bias likely)."
        elif back_wins > ideal_win_rate + 3:
            pattern = "Back marker advantage detected (hard wood bias likely)."
        else:
            pattern = "No clear front/back marker bias pattern."
        
        return f"""FAIRNESS RATING: {rating}

STATISTICAL ANALYSIS: With {len(analysis['competitors'])} competitors, ideal win rate is {ideal_win_rate:.1f}% each. Actual spread is {win_rate_spread:.2f}% (from {min_win_rate:.1f}% to {max_win_rate:.1f}%). {assessment} Average finish spread of {analysis['avg_spread']:.1f}s creates exciting competition.

PATTERN DIAGNOSIS: {pattern} {most_favored} is most favored at {analysis['winner_percentages'][most_favored]:.1f}% wins (+{win_rate_deviations[most_favored]:.1f}% above ideal), while {most_disadvantaged} is disadvantaged at {analysis['winner_percentages'][most_disadvantaged]:.1f}% wins ({win_rate_deviations[most_disadvantaged]:.1f}% below ideal).

RECOMMENDATIONS:
- {"Handicaps are ready for competition use - no adjustments needed." if win_rate_spread < 6 else f"Review predictions for {most_favored} and {most_disadvantaged} - time estimates may need adjustment."}
- {"Continue collecting historical data to improve future predictions." if win_rate_spread < 10 else "Consider adjusting quality/species factors in prediction model."}
- {"Monitor real competition results to validate simulation predictions." if win_rate_spread < 16 else "Recalibrate baseline calculations before using these handicaps in competition."}"""

#Main function to simulate the handicaps and assess the fairness

def simulate_and_assess_handicaps(competitors_with_marks, num_simulations=250000):
   
    if not competitors_with_marks or len(competitors_with_marks) < 2:
        print("Need at least 2 competitors to run simulation.")
        return
    
    # Run simulation
    analysis = run_monte_carlo_simulation(competitors_with_marks, num_simulations)
    
    # Display results
    summary = generate_simulation_summary(analysis)
    print(summary)
    
    # Visualize
    visualize_simulation_results(analysis)
    
    # Get AI assessment
    print("\n" + "="*70)
    print("AI HANDICAPPING ASSESSMENT")
    print("="*70)
    print("\nAnalyzing fairness of handicaps...")
    
    ai_assessment = get_ai_assessment_of_handicaps(analysis)
    
    # Wrap text to fit within 70 characters
    wrapped_text = textwrap.fill(ai_assessment, width=70)
    print(f"\n{wrapped_text}")
    
    print("\n" + "="*70)



# MENU OPTION 4: VIEW HANDICAP MARKS (AI-ENHANCED)
''' Official will be presented with the calculated handicap marks for each selected competitor in the heat
    1. View Handicap Marks
    2. Return to Main Menu

 3 Main Functions:
        -One to view the handicap menu
        -One to validate the data
        -One to actually tabulate the data

        1) Official finishes running a heat
        2) Calls this function with competitor list and wood specs
        3) Function validates event is selected
        4) Prompts for Heat ID (optional)
        4) Loops through each competitor asking for their time
        5) Skips anyone with blank time entry
        6) Converts valid times to proper format and collects in list
        7) Opens Excel file (or creates new one)
        8) Finds/creates Results sheet
        9) Write all result rows at once
        10) Save and close file

'''



#View Handicap Marks Menu
def view_handicaps_menu(heat_assignment_df, wood_selection):
    ''' Official will be presented with the calculated handicap marks for each selected competitor in the heat
        1. View handicap marks for current heat
        2. Return to Main Menu
    '''

    #safety check to make sure that eiyther standing block or Underhand is selected. If not, will default to none so program doesn't crash
    if "event" not in wood_selection:
        wood_selection["event"] = None

    #menu loop
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

#Helper to validate heat data is complete

    '''1)CHecks if all data exists
    2) jumps back to menu if data is incomplete

    Checks specifically for: competitors in heat assignment, species, size, and event"
    '''

def validate_heat_data(heat_assignment_df, wood_selection):
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

    # Calculate and display AI-enhanced handicap marks for the heat

def view_handicaps(heat_assignment_df, wood_selection):
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
        #pull explanation gnerated from model
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



## MENU OPTION 5: UPDATE RESULTS WITH HEAT RESULTS

'''This menus wiull allow a judge to append the "Results" tab of the 'woodchopping' excel sheet.
    The idea is that after a heat, the judge can update the results and the next heat a competitor races, the handicap will be updated and a more accurate 
     mark will be assigned (can be used to help order Finals) '''

#Find the excel file and sheet names htat will be used to store results
def detect_results_sheet(wb):
    if RESULTS_SHEET in wb.sheetnames:
        return wb[RESULTS_SHEET]
    
    ws = wb.create_sheet(RESULTS_SHEET)
    ws.append(["CompetitorID", "Event", "Time (seconds)", "Size (mm)", "Species Code", "Date"])
    return ws

#Add the results to the 'Results' sheet in the Excel file
'''Prompts the user for: Event (SB/UH), species, wood, quality, and notes (could be heat ID or notes about finals/qualifier or simply the date)'''

def append_results_to_excel(heat_assignment_df, wood_selection):
    if heat_assignment_df is None or heat_assignment_df.empty:
        print("No competitors in heat assignment.")
        return

    event_code = wood_selection.get("event")
    if event_code not in ("SB", "UH"):
        print("Event not selected. Use Wood Menu -> Select event (SB/UH) or Main Menu option 3.")
        return

    species = wood_selection.get("species")
    size_mm = wood_selection.get("size_mm")
    quality = wood_selection.get("quality")

    heat_id = input("Enter a Heat ID (e.g., SB-01-Qual or any short label): ").strip()
    if not heat_id:
        heat_id = f"{event_code}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    print("\nEnter actual times in seconds for each competitor (press Enter to skip a competitor).")


    # Get ID/name mapping
    _, name_to_id = get_competitor_id_name_mapping()
    
    rows_to_write = []
    for _, row in heat_assignment_df.iterrows():
        name = row.get("competitor_name")
        s = input(f"Time for {name}: ").strip()
        
        if s == "":
            continue
        
        try:
            t = float(s)
            timestamp = datetime.now().isoformat(timespec="seconds")
            competitor_id = name_to_id.get(str(name).strip().lower(), name)
            rows_to_write.append([competitor_id, event_code, t, size_mm, species, timestamp])
        except ValueError:
            print("Invalid time; skipping this entry.")
            continue

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
        
        # Ensure header exists with Excel column names
        if ws.max_row == 0:
            ws.append(["CompetitorID", "Event", "Time (seconds)", "Size (mm)", "Species Code", "Date"])

        # Append rows
        for r in rows_to_write:
            ws.append(r)
        
        wb.save(RESULTS_FILE)
        wb.close()
        print("Results appended to Excel successfully.")
        
    except Exception as e:
        print(f"Error appending results: {e}")