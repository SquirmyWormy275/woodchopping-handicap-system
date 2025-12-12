"""Wood characteristics configuration UI functions.

This module handles wood block configuration menus including:
- Wood species selection
- Block size entry
- Quality rating
- Event code selection (SB/UH)
"""

from typing import Dict
from woodchopping.data import load_wood_data


def wood_menu(wood_selection: Dict) -> Dict:
    """Wood characteristics menu for configuring block parameters.

    Args:
        wood_selection: Dictionary containing current wood settings

    Returns:
        dict: Updated wood selection dictionary
    """

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


def select_wood_species(wood_selection: Dict) -> Dict:
    """Select wood species from available options.

    Args:
        wood_selection: Current wood selection dictionary

    Returns:
        dict: Updated wood selection with chosen species
    """
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


def enter_wood_size_mm(wood_selection: Dict) -> Dict:
    """Enter block diameter in millimeters.

    Args:
        wood_selection: Current wood selection dictionary

    Returns:
        dict: Updated wood selection with block size
    """
    size = input("Enter block diameter in mm: ").strip()

    try:
        val = float(size)
        wood_selection["size_mm"] = val
        format_wood(wood_selection)
    except ValueError:
        print("Invalid size input.")

    return wood_selection


def enter_wood_quality(wood_selection: Dict) -> Dict:
    """Enter wood quality rating (0-10 scale).

    Quality scale:
    - 0-3: Soft/rotten wood (faster times)
    - 4-7: Average firmness for species
    - 8-10: Above average firmness (slower times)

    Args:
        wood_selection: Current wood selection dictionary

    Returns:
        dict: Updated wood selection with quality rating
    """
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


def format_wood(ws: Dict) -> str:
    """Display formatted header for current wood selection.

    Args:
        ws: Wood selection dictionary

    Returns:
        str: Formatted header string
    """
    s = ws.get("species", "—")
    d = ws.get("size_mm", "—")
    q = ws.get("quality", "—")
    header = f"Selected Wood -> Species: {s}, Diameter: {d} mm, Quality: {q}"
    print(f"Wood selection updated: {header}")
    return header


def select_event_code(wood_selection: Dict) -> Dict:
    """Select event type (Standing Block or Underhand).

    Args:
        wood_selection: Current wood selection dictionary

    Returns:
        dict: Updated wood selection with event code
    """
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
