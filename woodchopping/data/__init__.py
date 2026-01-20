"""Data handling module for woodchopping system."""

from woodchopping.data.excel_io import (
    load_competitors_df,
    load_results_df,
    load_wood_data,
    get_competitor_id_name_mapping,
    get_species_name_from_code,
    detect_results_sheet,
    save_time_to_results,
    append_results_to_excel,
)

from woodchopping.data.validation import (
    validate_results_data,
    validate_heat_data,
    standardize_results_data,
)

from woodchopping.data.preprocessing import (
    engineer_features_for_ml,
    load_and_clean_results,
    fit_wood_hardness_index,
    calculate_adaptive_half_lives,
)

__all__ = [
    # Excel I/O
    "load_competitors_df",
    "load_results_df",
    "load_wood_data",
    "get_competitor_id_name_mapping",
    "get_species_name_from_code",
    "detect_results_sheet",
    "save_time_to_results",
    "append_results_to_excel",
    # Validation
    "validate_results_data",
    "validate_heat_data",
    "standardize_results_data",
    # Preprocessing
    "engineer_features_for_ml",
    # Baseline V2 Preprocessing
    "load_and_clean_results",
    "fit_wood_hardness_index",
    "calculate_adaptive_half_lives",
]
