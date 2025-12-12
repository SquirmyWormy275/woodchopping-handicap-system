"""User interface functions for menus and display.

This module provides all UI components for the woodchopping handicap system:
- Tournament management (multi-round tournaments)
- Wood configuration (species, size, quality)
- Competitor selection (heat assignments)
- Personnel management (roster management)
- Handicap display (marks and results)
"""

# Tournament UI
from woodchopping.ui.tournament_ui import (
    calculate_tournament_scenarios,
    distribute_competitors_into_heats,
    select_heat_advancers,
    generate_next_round,
    view_tournament_status,
    save_tournament_state,
    load_tournament_state,
    auto_save_state
)

# Wood UI
from woodchopping.ui.wood_ui import (
    wood_menu,
    select_wood_species,
    enter_wood_size_mm,
    enter_wood_quality,
    format_wood,
    select_event_code
)

# Competitor UI
from woodchopping.ui.competitor_ui import (
    select_all_event_competitors,
    competitor_menu,
    select_competitors_for_heat,
    view_heat_assignment,
    remove_from_heat
)

# Personnel UI
from woodchopping.ui.personnel_ui import (
    personnel_management_menu,
    add_competitor_with_times,
    add_historical_times_for_competitor
)

# Handicap UI
from woodchopping.ui.handicap_ui import (
    view_handicaps_menu,
    validate_heat_data,
    view_handicaps,
    append_results_to_excel
)


__all__ = [
    # Tournament UI
    'calculate_tournament_scenarios',
    'distribute_competitors_into_heats',
    'select_heat_advancers',
    'generate_next_round',
    'view_tournament_status',
    'save_tournament_state',
    'load_tournament_state',
    'auto_save_state',

    # Wood UI
    'wood_menu',
    'select_wood_species',
    'enter_wood_size_mm',
    'enter_wood_quality',
    'format_wood',
    'select_event_code',

    # Competitor UI
    'select_all_event_competitors',
    'competitor_menu',
    'select_competitors_for_heat',
    'view_heat_assignment',
    'remove_from_heat',

    # Personnel UI
    'personnel_management_menu',
    'add_competitor_with_times',
    'add_historical_times_for_competitor',

    # Handicap UI
    'view_handicaps_menu',
    'validate_heat_data',
    'view_handicaps',
    'append_results_to_excel',
]
