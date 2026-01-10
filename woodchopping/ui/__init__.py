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

# Competitor Dashboard (A9 + B2 + B4)
from woodchopping.ui.competitor_dashboard import (
    display_competitor_dashboard
)

# Schedule Printout (A1)
from woodchopping.ui.schedule_printout import (
    display_and_export_schedule
)

# Adjustment Tracking (A5)
from woodchopping.ui.adjustment_tracking import (
    log_handicap_adjustment,
    prompt_adjustment_reason,
    view_adjustment_history,
    get_adjustment_summary
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

# Multi-Event Tournament UI
from woodchopping.ui.multi_event_ui import (
    create_multi_event_tournament,
    save_multi_event_tournament,
    load_multi_event_tournament,
    auto_save_multi_event,
    add_event_to_tournament,
    view_wood_count,
    view_tournament_schedule,
    remove_event_from_tournament,
    calculate_all_event_handicaps,
    view_analyze_all_handicaps,
    approve_event_handicaps,
    generate_complete_day_schedule,
    view_all_handicaps_summary,
    sequential_results_workflow,
    get_next_incomplete_round,
    display_event_progress,
    extract_event_placements,
    generate_tournament_summary
)

# Championship Simulator UI
from woodchopping.ui.championship_simulator import (
    run_championship_simulator
)

# Payout UI
from woodchopping.ui.payout_ui import (
    configure_event_payouts,
    display_payout_config,
    display_final_results_with_payouts,
    calculate_total_earnings,
    display_tournament_earnings_summary,
    display_single_event_final_results
)

# Bracket UI (NEW V5.0 - Head-to-Head Bracket Tournament)
from woodchopping.ui.bracket_ui import (
    initialize_bracket_tournament,
    generate_bracket_seeds,
    generate_bracket_with_byes,
    generate_double_elimination_bracket,
    render_bracket_tree_ascii,
    render_round_section,
    sequential_match_entry_workflow,
    export_bracket_to_html,
    open_bracket_in_browser
)

# V5.1 Helpers (NEW V5.1 - Entry-Form Workflow)
from woodchopping.ui.v51_helpers import (
    view_tournament_entries,
    edit_event_entries,
    manage_scratches
)

# Tournament Status & Validation (NEW V5.1)
from woodchopping.ui.tournament_status import (
    display_tournament_progress_tracker,
    calculate_tournament_progress,
    check_can_calculate_handicaps,
    check_can_generate_schedule,
    get_progress_summary
)

# Error Display (NEW V5.1 - Improved Error Handling)
from woodchopping.ui.error_display import (
    display_actionable_error,
    display_blocking_error,
    display_warning,
    display_success,
    display_progress_box,
    prompt_with_options
)

# Entry Fee Tracker (NEW V5.1)
from woodchopping.ui.entry_fee_tracker import (
    view_entry_fee_status,
    mark_fees_paid_by_competitor,
    mark_fees_paid_by_event,
    display_payment_grid
)

# Scratch Management (NEW V5.1)
from woodchopping.ui.scratch_management import (
    manage_tournament_scratches,
    view_all_competitors_with_status,
    mark_competitor_scratched,
    view_scratch_history,
    restore_scratched_competitor,
    check_competitor_status,
    get_scratch_count
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

    # Competitor Dashboard (A9 + B2 + B4)
    'display_competitor_dashboard',

    # Schedule Printout (A1)
    'display_and_export_schedule',

    # Adjustment Tracking (A5)
    'log_handicap_adjustment',
    'prompt_adjustment_reason',
    'view_adjustment_history',
    'get_adjustment_summary',

    # Personnel UI
    'personnel_management_menu',
    'add_competitor_with_times',
    'add_historical_times_for_competitor',

    # Handicap UI
    'view_handicaps_menu',
    'validate_heat_data',
    'view_handicaps',
    'append_results_to_excel',

    # Multi-Event Tournament UI
    'create_multi_event_tournament',
    'save_multi_event_tournament',
    'load_multi_event_tournament',
    'auto_save_multi_event',
    'add_event_to_tournament',
    'view_wood_count',
    'view_tournament_schedule',
    'remove_event_from_tournament',
    'calculate_all_event_handicaps',
    'view_analyze_all_handicaps',
    'approve_event_handicaps',
    'generate_complete_day_schedule',
    'view_all_handicaps_summary',
    'sequential_results_workflow',
    'get_next_incomplete_round',
    'display_event_progress',
    'extract_event_placements',
    'generate_tournament_summary',

    # Championship Simulator UI
    'run_championship_simulator',

    # Payout UI
    'configure_event_payouts',
    'display_payout_config',
    'display_final_results_with_payouts',
    'calculate_total_earnings',
    'display_tournament_earnings_summary',
    'display_single_event_final_results',

    # Bracket UI (NEW V5.0)
    'initialize_bracket_tournament',
    'generate_bracket_seeds',
    'generate_bracket_with_byes',
    'generate_double_elimination_bracket',
    'render_bracket_tree_ascii',
    'render_round_section',
    'sequential_match_entry_workflow',
    'export_bracket_to_html',
    'open_bracket_in_browser',

    # V5.1 Helpers (NEW V5.1)
    'view_tournament_entries',
    'edit_event_entries',
    'manage_scratches',

    # Tournament Status & Validation (NEW V5.1)
    'display_tournament_progress_tracker',
    'calculate_tournament_progress',
    'check_can_calculate_handicaps',
    'check_can_generate_schedule',
    'get_progress_summary',

    # Error Display (NEW V5.1)
    'display_actionable_error',
    'display_blocking_error',
    'display_warning',
    'display_success',
    'display_progress_box',
    'prompt_with_options',

    # Entry Fee Tracker (NEW V5.1)
    'view_entry_fee_status',
    'mark_fees_paid_by_competitor',
    'mark_fees_paid_by_event',
    'display_payment_grid',

    # Scratch Management (NEW V5.1)
    'manage_tournament_scratches',
    'view_all_competitors_with_status',
    'mark_competitor_scratched',
    'view_scratch_history',
    'restore_scratched_competitor',
    'check_competitor_status',
    'get_scratch_count',
]
