#Import Pandas, numpy, 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
from math import ceil
from openpyxl import load_workbook

#Import Functions from project_functions.py 
import FunctionsLibrary as pf

#Load Competitor Data from Excel  full roster)
'''Read the xlsx file containing data. 
Sheet "competitors" contains competitor data
Sheet "wood" contains wood species data'''
try:
    comp_df = pf.load_competitors_df()
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

# Heat assignment state (competitors selected for current heat)
heat_assignment_df = pd.DataFrame()
heat_assignment_names = []

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


## Main Menu
''' Official will enter the menu and be presented with two options
1. Competitor Selection Menu
2. Wood Characteristics Menu
3. View Handicap Marks
create a loop that will allow the official to return to the main menu after completing tasks in sub-menus
run menu as a function to allow for easy return to main menu
'''
while True:
    print("\nWelcome to the Wood Chopping Handicap Management System")
    print("Please select an option from the Main Menu:")
    print("1. Competitor Selection Menu")
    print("2. Wood Characteristics Menu")
    print("3. Select Event (SB/UH)")
    print("4. View AI Enhanced Handicap Marks")
    print("5. Update Results tab with Heat Results")
    print("6. Reload roster from Excel")
    print("7. Exit")
    menu_choice = input("Enter your choice (1-7): ").strip()

    if menu_choice == '1':
        # enters competitor selection menu
        comp_df, heat_assignment_df, heat_assignment_names = pf.competitor_menu(
            comp_df, heat_assignment_df, heat_assignment_names
        )

    elif menu_choice == '2':
        # returns updated wood selection
        wood_selection = pf.wood_menu(wood_selection)

    elif menu_choice == '3':
        # event selection (SB/UH)
        wood_selection = pf.select_event_code(wood_selection)

    elif menu_choice == '4':
        # displays handicaps for heat assignment
        pf.view_handicaps_menu(heat_assignment_df, wood_selection)

    elif menu_choice == '5':
        # append heat results to Results tab
        pf.append_results_to_excel(heat_assignment_df, wood_selection)

    elif menu_choice == '6':
        try:
            comp_df = pf.load_competitors_df()
            print("Roster reloaded from Excel.")
            # Reset heat assignment when reloading roster
            heat_assignment_df = pd.DataFrame()
            heat_assignment_names = []
            print("Note: Heat assignment has been cleared.")
        except Exception as e:
            print(f"Failed to reload roster: {e}")

    elif menu_choice == '7' or menu_choice == '':
        print("Goodbye.")
        break

    else:
        print("Invalid selection. Try again.")