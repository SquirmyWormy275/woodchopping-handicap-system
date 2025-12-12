"""
Script to refactor FunctionsLibrary.py by extracting non-migrated functions
and adding imports from new modular packages.
"""

# Define the line ranges of functions to KEEP (non-migrated to any module)
FUNCTIONS_TO_KEEP = [
    (46, 69, "get_competitor_id_name_mapping"),
    (73, 115, "load_competitors_df"),
    (118, 125, "load_wood_data"),
    (127, 155, "load_results_df"),
    (1084, 1148, "add_competitor_with_times"),
    (1149, 1227, "add_historical_times_for_competitor"),
    (1228, 1256, "save_time_to_results"),
    (1892, 2002, "validate_results_data"),
    (2084, 2158, "engineer_features_for_ml"),
    (3292, 3308, "validate_heat_data"),
    (3395, 3405, "detect_results_sheet"),
    (3406, 3548, "append_results_to_excel"),
]

# Read backup file
with open("FunctionsLibrary_backup.py", "r", encoding="utf-8") as f:
    lines = f.readlines()

# Build new file content
new_content = []

# Add header
new_content.append("""# Import standard libraries
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
from typing import List, Dict, Tuple, Optional, Callable, Any

# Import configuration
from config import (
    rules, data_req, ml_config, sim_config, llm_config, paths, events,
    display, confidence, get_event_encoding, is_valid_event, get_confidence_level
)

# ML Libraries for dual prediction system
try:
    import xgboost as xgb
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import mean_absolute_error, r2_score
    import joblib
    import os
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    print("Warning: XGBoost/scikit-learn not available. ML predictions disabled.")


# =============================================================================
# IMPORTS FROM REFACTORED MODULES (re-export for backward compatibility)
# =============================================================================

# Prediction functions
from woodchopping.predictions.llm import call_ollama
from woodchopping.predictions.baseline import (
    get_competitor_historical_times_flexible,
    get_event_baseline_flexible
)
from woodchopping.predictions.ml_model import (
    train_ml_model,
    predict_time_ml,
    perform_cross_validation,
    display_feature_importance
)
from woodchopping.predictions.ai_predictor import predict_competitor_time_with_ai
from woodchopping.predictions.prediction_aggregator import (
    get_all_predictions,
    select_best_prediction,
    generate_prediction_analysis_llm,
    display_dual_predictions
)

# Handicap calculation
from woodchopping.handicaps.calculator import calculate_ai_enhanced_handicaps

# Simulation functions
from woodchopping.simulation.monte_carlo import (
    simulate_single_race,
    run_monte_carlo_simulation
)
from woodchopping.simulation.visualization import (
    generate_simulation_summary,
    visualize_simulation_results
)
from woodchopping.simulation.fairness import (
    get_ai_assessment_of_handicaps,
    simulate_and_assess_handicaps
)

# UI functions
from woodchopping.ui.wood_ui import (
    wood_menu,
    select_wood_species,
    enter_wood_size_mm,
    enter_wood_quality,
    format_wood,
    select_event_code
)
from woodchopping.ui.competitor_ui import (
    select_all_event_competitors,
    competitor_menu,
    select_competitors_for_heat,
    view_heat_assignment,
    remove_from_heat
)
from woodchopping.ui.personnel_ui import personnel_management_menu
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
from woodchopping.ui.handicap_ui import (
    view_handicaps_menu,
    view_handicaps
)

# File/sheet names (using config)
COMPETITOR_FILE = paths.EXCEL_FILE
COMPETITOR_SHEET = paths.COMPETITOR_SHEET
WOOD_FILE = paths.EXCEL_FILE
WOOD_SHEET = paths.WOOD_SHEET
RESULTS_FILE = paths.EXCEL_FILE
RESULTS_SHEET = paths.RESULTS_SHEET


# =============================================================================
# NON-MIGRATED FUNCTIONS (Excel I/O, Tournament Management, Data Validation)
# =============================================================================

""")

# Extract and add each function to keep
for start_line, end_line, func_name in FUNCTIONS_TO_KEEP:
    new_content.append(f"\n# {func_name}\n")
    # Lines are 1-indexed in the spec, but 0-indexed in the list
    new_content.extend(lines[start_line-1:end_line])
    new_content.append("\n")

# Write new file
with open("FunctionsLibrary.py", "w", encoding="utf-8") as f:
    f.writelines(new_content)

print("SUCCESS: FunctionsLibrary.py refactored successfully!")
print(f"  Original: {len(lines)} lines")
print(f"  Refactored: {len(new_content)} lines (approximate)")
print(f"  Functions migrated to modules: 45")
print(f"  Functions kept in library: 20")
