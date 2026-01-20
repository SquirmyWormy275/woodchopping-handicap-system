"""
QAA Diameter Scaling Tables

Empirical scaling tables from Queensland Axemen's Association (QAA) with 150+ years
of institutional knowledge. These tables show how handicap marks scale across different
wood diameters, based on actual observed performance data.

Standard diameter: 300mm (12")
All book marks are recorded at 300mm standard, then converted using these tables.

Source: QAA By-Laws, Pages 9-12
"""

from typing import Optional, Tuple
import pandas as pd


# ============================================================================
# OPEN UNDERHAND & STANDING BLOCK HANDICAP SCALE (Page 9)
# ============================================================================
# Standard: 300mm (12") diameter
# Book marks range: 3-43 seconds

QAA_UH_SB_HARDWOOD = {
    # book_mark_300mm: {diameter_mm: scaled_mark}
    3: {225: 2, 250: 2, 275: 3, 300: 3, 325: 3, 350: 3},
    4: {225: 2, 250: 3, 275: 4, 300: 4, 325: 4, 350: 4},
    5: {225: 3, 250: 3, 275: 4, 300: 5, 325: 5, 350: 5},
    6: {225: 3, 250: 4, 275: 5, 300: 6, 325: 6, 350: 6},
    7: {225: 4, 250: 4, 275: 6, 300: 7, 325: 7, 350: 7},
    8: {225: 4, 250: 5, 275: 7, 300: 8, 325: 8, 350: 8},
    9: {225: 5, 250: 6, 275: 8, 300: 9, 325: 9, 350: 9},
    10: {225: 5, 250: 6, 275: 8, 300: 10, 325: 10, 350: 10},
    11: {225: 6, 250: 7, 275: 9, 300: 11, 325: 11, 350: 11},
    12: {225: 6, 250: 8, 275: 10, 300: 12, 325: 12, 350: 12},
    13: {225: 7, 250: 8, 275: 11, 300: 13, 325: 14, 350: 15},
    14: {225: 8, 250: 9, 275: 12, 300: 14, 325: 15, 350: 16},
    15: {225: 8, 250: 9, 275: 12, 300: 15, 325: 16, 350: 17},
    16: {225: 8, 250: 10, 275: 13, 300: 16, 325: 17, 350: 18},
    17: {225: 9, 250: 11, 275: 14, 300: 17, 325: 18, 350: 19},
    18: {225: 9, 250: 11, 275: 15, 300: 18, 325: 19, 350: 20},
    19: {225: 10, 250: 12, 275: 16, 300: 19, 325: 20, 350: 21},
    20: {225: 11, 250: 13, 275: 17, 300: 20, 325: 21, 350: 23},
    21: {225: 11, 250: 13, 275: 17, 300: 21, 325: 23, 350: 25},
    22: {225: 11, 250: 14, 275: 18, 300: 22, 325: 24, 350: 26},
    23: {225: 12, 250: 14, 275: 19, 300: 23, 325: 25, 350: 27},
    24: {225: 13, 250: 15, 275: 20, 300: 24, 325: 26, 350: 28},
    25: {225: 13, 250: 16, 275: 21, 300: 25, 325: 27, 350: 29},
    26: {225: 13, 250: 16, 275: 21, 300: 26, 325: 28, 350: 30},
    27: {225: 14, 250: 17, 275: 22, 300: 27, 325: 29, 350: 31},
    28: {225: 14, 250: 18, 275: 23, 300: 28, 325: 30, 350: 32},
    29: {225: 15, 250: 18, 275: 24, 300: 29, 325: 31, 350: 34},
    30: {225: 16, 250: 19, 275: 25, 300: 30, 325: 32, 350: 34},
    31: {225: 16, 250: 19, 275: 25, 300: 31, 325: 34, 350: 37},
    32: {225: 16, 250: 20, 275: 26, 300: 32, 325: 35, 350: 38},
    33: {225: 17, 250: 21, 275: 27, 300: 33, 325: 36, 350: 39},
    34: {225: 18, 250: 21, 275: 28, 300: 34, 325: 37, 350: 40},
    35: {225: 18, 250: 22, 275: 29, 300: 35, 325: 38, 350: 41},
    36: {225: 19, 250: 23, 275: 30, 300: 36, 325: 39, 350: 42},
    37: {225: 19, 250: 23, 275: 30, 300: 37, 325: 40, 350: 43},
    38: {225: 19, 250: 24, 275: 31, 300: 38, 325: 41, 350: 45},
    39: {225: 20, 250: 24, 275: 32, 300: 39, 325: 42, 350: 46},
    40: {225: 20, 250: 25, 275: 33, 300: 40, 325: 43, 350: 47},
    41: {225: 21, 250: 26, 275: 34, 300: 41, 325: 45, 350: 49},
    42: {225: 21, 250: 26, 275: 34, 300: 42, 325: 46, 350: 50},
    43: {225: 22, 250: 27, 275: 35, 300: 43, 325: 47, 350: 51},
}

QAA_UH_SB_MEDIUM_WOOD = {
    # book_mark_300mm: {diameter_mm: scaled_mark}
    3: {225: 1, 250: 2, 275: 3, 300: 3, 325: 3, 350: 3},
    4: {225: 2, 250: 3, 275: 3, 300: 4, 325: 4, 350: 4},
    5: {225: 2, 250: 3, 275: 4, 300: 4, 325: 4, 350: 4},
    6: {225: 3, 250: 4, 275: 4, 300: 5, 325: 5, 350: 5},
    7: {225: 3, 250: 4, 275: 5, 300: 6, 325: 6, 350: 6},
    8: {225: 3, 250: 4, 275: 6, 300: 7, 325: 7, 350: 7},
    9: {225: 4, 250: 5, 275: 7, 300: 8, 325: 8, 350: 8},
    10: {225: 4, 250: 5, 275: 7, 300: 8, 325: 8, 350: 8},
    11: {225: 4, 250: 6, 275: 8, 300: 9, 325: 9, 350: 9},
    12: {225: 5, 250: 7, 275: 8, 300: 10, 325: 10, 350: 10},
    13: {225: 5, 250: 7, 275: 9, 300: 11, 325: 12, 350: 12},
    14: {225: 6, 250: 8, 275: 10, 300: 12, 325: 12, 350: 13},
    15: {225: 6, 250: 8, 275: 10, 300: 12, 325: 13, 350: 14},
    16: {225: 6, 250: 8, 275: 11, 300: 13, 325: 14, 350: 15},
    17: {225: 7, 250: 9, 275: 12, 300: 14, 325: 15, 350: 16},
    18: {225: 7, 250: 9, 275: 12, 300: 15, 325: 16, 350: 17},
    19: {225: 8, 250: 10, 275: 13, 300: 16, 325: 17, 350: 17},
    20: {225: 8, 250: 11, 275: 14, 300: 17, 325: 17, 350: 19},
    21: {225: 8, 250: 11, 275: 14, 300: 17, 325: 19, 350: 21},
    22: {225: 9, 250: 12, 275: 15, 300: 18, 325: 20, 350: 21},
    23: {225: 9, 250: 12, 275: 16, 300: 19, 325: 21, 350: 22},
    24: {225: 9, 250: 12, 275: 17, 300: 20, 325: 21, 350: 23},
    25: {225: 10, 250: 13, 275: 17, 300: 21, 325: 22, 350: 24},
    26: {225: 10, 250: 13, 275: 17, 300: 21, 325: 23, 350: 25},
    27: {225: 11, 250: 14, 275: 18, 300: 22, 325: 24, 350: 25},
    28: {225: 11, 250: 15, 275: 19, 300: 23, 325: 25, 350: 26},
    29: {225: 11, 250: 15, 275: 20, 300: 24, 325: 25, 350: 28},
    30: {225: 12, 250: 16, 275: 21, 300: 25, 325: 26, 350: 28},
    31: {225: 12, 250: 16, 275: 21, 300: 25, 325: 28, 350: 30},
    32: {225: 13, 250: 17, 275: 21, 300: 26, 325: 29, 350: 31},
    33: {225: 13, 250: 17, 275: 22, 300: 27, 325: 30, 350: 32},
    34: {225: 13, 250: 17, 275: 23, 300: 28, 325: 30, 350: 33},
    35: {225: 14, 250: 18, 275: 24, 300: 29, 325: 31, 350: 34},
    36: {225: 15, 250: 19, 275: 25, 300: 30, 325: 32, 350: 34},
    37: {225: 15, 250: 19, 275: 25, 300: 30, 325: 33, 350: 35},
    38: {225: 16, 250: 20, 275: 25, 300: 31, 325: 34, 350: 37},
    39: {225: 16, 250: 20, 275: 26, 300: 32, 325: 34, 350: 37},
    40: {225: 16, 250: 21, 275: 27, 300: 33, 325: 35, 350: 38},
    41: {225: 16, 250: 21, 275: 28, 300: 34, 325: 37, 350: 40},
    42: {225: 16, 250: 21, 275: 28, 300: 34, 325: 37, 350: 40},
    43: {225: 17, 250: 22, 275: 29, 300: 35, 325: 38, 350: 41},
}

QAA_UH_SB_SOFTWOOD = {
    # book_mark_300mm: {diameter_mm: scaled_mark}
    3: {225: 1, 250: 1, 275: 2, 300: 2, 325: 2, 350: 2},
    4: {225: 2, 250: 2, 275: 2, 300: 3, 325: 3, 350: 3},
    5: {225: 2, 250: 2, 275: 3, 300: 3, 325: 3, 350: 3},
    6: {225: 2, 250: 3, 275: 3, 300: 4, 325: 4, 350: 4},
    7: {225: 2, 250: 3, 275: 4, 300: 4, 325: 4, 350: 4},
    8: {225: 2, 250: 3, 275: 4, 300: 5, 325: 5, 350: 5},
    9: {225: 3, 250: 4, 275: 5, 300: 6, 325: 6, 350: 6},
    10: {225: 3, 250: 4, 275: 5, 300: 6, 325: 6, 350: 6},
    11: {225: 3, 250: 4, 275: 6, 300: 7, 325: 7, 350: 7},
    12: {225: 4, 250: 5, 275: 6, 300: 8, 325: 8, 350: 8},
    13: {225: 4, 250: 5, 275: 7, 300: 8, 325: 9, 350: 9},
    14: {225: 5, 250: 6, 275: 8, 300: 9, 325: 9, 350: 10},
    15: {225: 5, 250: 6, 275: 8, 300: 9, 325: 10, 350: 11},
    16: {225: 5, 250: 6, 275: 8, 300: 10, 325: 11, 350: 11},
    17: {225: 6, 250: 7, 275: 9, 300: 11, 325: 11, 350: 12},
    18: {225: 6, 250: 7, 275: 9, 300: 11, 325: 12, 350: 13},
    19: {225: 6, 250: 8, 275: 10, 300: 12, 325: 13, 350: 13},
    20: {225: 6, 250: 8, 275: 11, 300: 13, 325: 13, 350: 14},
    21: {225: 6, 250: 8, 275: 11, 300: 13, 325: 14, 350: 16},
    22: {225: 7, 250: 9, 275: 11, 300: 14, 325: 15, 350: 16},
    23: {225: 7, 250: 9, 275: 12, 300: 14, 325: 16, 350: 17},
    24: {225: 7, 250: 9, 275: 13, 300: 16, 325: 16, 350: 18},
    25: {225: 8, 250: 10, 275: 13, 300: 16, 325: 17, 350: 18},
    26: {225: 8, 250: 10, 275: 13, 300: 16, 325: 18, 350: 19},
    27: {225: 9, 250: 11, 275: 14, 300: 17, 325: 18, 350: 19},
    28: {225: 9, 250: 11, 275: 14, 300: 18, 325: 19, 350: 20},
    29: {225: 9, 250: 11, 275: 15, 300: 18, 325: 19, 350: 21},
    30: {225: 10, 250: 12, 275: 16, 300: 19, 325: 20, 350: 21},
    31: {225: 10, 250: 12, 275: 16, 300: 19, 325: 21, 350: 23},
    32: {225: 11, 250: 13, 275: 16, 300: 20, 325: 22, 350: 24},
    33: {225: 11, 250: 13, 275: 17, 300: 21, 325: 23, 350: 24},
    34: {225: 11, 250: 13, 275: 18, 300: 21, 325: 23, 350: 25},
    35: {225: 11, 250: 14, 275: 18, 300: 22, 325: 24, 350: 26},
    36: {225: 12, 250: 15, 275: 19, 300: 23, 325: 24, 350: 26},
    37: {225: 12, 250: 15, 275: 19, 300: 23, 325: 25, 350: 27},
    38: {225: 13, 250: 16, 275: 19, 300: 24, 325: 26, 350: 28},
    39: {225: 13, 250: 16, 275: 20, 300: 24, 325: 26, 350: 28},
    40: {225: 13, 250: 16, 275: 20, 300: 25, 325: 27, 350: 29},
    41: {225: 13, 250: 16, 275: 21, 300: 26, 325: 28, 350: 30},
    42: {225: 13, 250: 16, 275: 21, 300: 26, 325: 28, 350: 30},
    43: {225: 14, 250: 17, 275: 22, 300: 27, 325: 30, 350: 31},
}


# ============================================================================
# VETERANS, JUNIORS, NOVICE & WOMEN'S HANDICAP SCALE (Page 10)
# ============================================================================
# Standard: 275mm UH, 250mm SB for these categories
# Book marks range: 3-60 seconds (max 60 for these divisions)

QAA_VETERANS_NOVICE_WOMENS = {
    # Different standard than Open events - varies by category
    # Use hardwood/medium/softwood with 1" smaller/bookmark/1" larger columns
    # Format: book_mark: {size_category: {wood_type: mark}}
    # Categories: 'smaller' (1" smaller than standard), 'standard', 'larger' (1" larger)
}


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def calculate_effective_janka_hardness(
    species_code: str,
    quality: int,
    wood_df: pd.DataFrame
) -> float:
    """
    Calculate EFFECTIVE Janka hardness accounting for BOTH species AND block quality.

    This is critical because:
    - Hard White Pine (quality 9) can be as hard as average Ponderosa Pine
    - Soft Rock Maple (quality 2) can be softer than average White Pine
    - The quality scale captures block-specific firmness variations

    Args:
        species_code: Species code (e.g., 'WP', 'YP', 'RM')
        quality: Wood quality 1-10 (1=very soft, 5=average, 10=very hard)
        wood_df: Wood database with janka_hardness_lbf column

    Returns:
        Effective Janka hardness in lbf (adjusted for quality)

    Quality Adjustment (1 = softest, 10 = hardest):
        Quality 1 (very soft/rotten): 0.6x base Janka (punky, decomposed)
        Quality 5 (average): 1.0x base Janka (normal for species)
        Quality 10 (very hard/firm): 1.5x base Janka (green wood, knots)

    Example:
        White Pine base: 420 Janka
        - Quality 2 (soft): 420 x 0.8 = 336 Janka (softer than average)
        - Quality 5 (avg):  420 x 1.0 = 420 Janka (species baseline)
        - Quality 9 (hard): 420 x 1.4 = 588 Janka (approaching Yellow Pine at 690)
    """
    quality = max(1, min(10, int(quality)))

    # Get base Janka from wood database
    # Match by speciesID (e.g., 'WP', 'YP', 'RM')
    wood_row = wood_df[wood_df['speciesID'].str.upper() == species_code.upper()]

    if wood_row.empty:
        # Fallback if species not found - use medium default
        base_janka = 750.0
    else:
        base_janka = float(wood_row['janka_hard'].values[0])

    # Quality adjustment factor (1 softest -> 10 hardest)
    # Linear interpolation: factor = 1.0 + ((quality - 5) * 0.1)
    # Quality 1: 0.6x (very soft)
    # Quality 5: 1.0x (average)
    # Quality 10: 1.5x (very hard)
    quality_factor = 1.0 + ((quality - 5) * 0.1)

    # Clamp quality factor to reasonable bounds (0.3 to 2.0)
    quality_factor = max(0.3, min(2.0, quality_factor))

    effective_janka = base_janka * quality_factor

    return effective_janka


def calculate_hardness_blend_weights(effective_janka: float) -> dict:
    """
    Calculate blend weights for QAA tables based on effective Janka hardness.

    Uses triangular membership functions for smooth transitions between
    Softwood, Medium, and Hardwood QAA scaling tables.

    Args:
        effective_janka: Effective Janka hardness (species + quality adjusted)

    Returns:
        dict with 'softwood', 'medium', 'hardwood' weights (sum to 1.0)

    Hardness Ranges (database values appear to be in Newtons):
        Softwood peak: 1300 (Cottonwood, soft White Pine) [~290 lbf]
        Medium peak:   2000 (Yellow Pine, Ponderosa) [~450 lbf]
        Hardwood peak: 2800 (Alder, harder woods) [~630 lbf]

    Transition zones: 700 overlap between categories

    Examples:
        1100 (rotten pine):          100% soft,  0% med,   0% hard
        1690 (avg white pine):        75% soft, 25% med,   0% hard
        2000 (ponderosa pine):        25% soft, 50% med,  25% hard
        2800 (alder):                  0% soft,  0% med, 100% hard
    """
    # Define peak centers for each category (adjusted for Newtons in database)
    SOFT_PEAK = 1300
    MED_PEAK = 2000
    HARD_PEAK = 2800

    # Transition width (blend zone)
    TRANSITION = 700

    weights = {
        'softwood': 0.0,
        'medium': 0.0,
        'hardwood': 0.0
    }

    # Softwood weight (triangular: peak at 300, fade to 0 by 800)
    if effective_janka <= SOFT_PEAK:
        weights['softwood'] = 1.0
    elif effective_janka < (SOFT_PEAK + TRANSITION):
        # Linear fade from 1.0 to 0.0
        weights['softwood'] = 1.0 - ((effective_janka - SOFT_PEAK) / TRANSITION)

    # Medium weight (triangular: ramp up from 300, peak at 750, fade by 1250)
    if SOFT_PEAK <= effective_janka <= MED_PEAK:
        # Ramp up from 0 to 1
        weights['medium'] = (effective_janka - SOFT_PEAK) / (MED_PEAK - SOFT_PEAK)
    elif MED_PEAK < effective_janka <= (MED_PEAK + TRANSITION):
        # Ramp down from 1 to 0
        weights['medium'] = 1.0 - ((effective_janka - MED_PEAK) / TRANSITION)

    # Hardwood weight (triangular: start at 750, peak at 1250+)
    if effective_janka >= HARD_PEAK:
        weights['hardwood'] = 1.0
    elif effective_janka > (HARD_PEAK - TRANSITION):
        # Linear ramp from 0.0 to 1.0
        weights['hardwood'] = (effective_janka - (HARD_PEAK - TRANSITION)) / TRANSITION

    # Normalize to sum to 1.0 (safety check)
    total = sum(weights.values())
    if total > 0:
        weights = {k: v/total for k, v in weights.items()}
    else:
        # Fallback if something went wrong
        weights['medium'] = 1.0

    return weights


def interpolate_qaa_tables(
    book_mark: float,
    target_diameter: float,
    effective_janka: float
) -> Tuple[float, dict]:
    """
    Interpolate between QAA tables based on effective Janka hardness.

    Instead of picking one table, blends all three based on where the wood
    falls on the hardness spectrum.

    Args:
        book_mark: Book mark at 300mm standard
        target_diameter: Target diameter in mm
        effective_janka: Effective Janka hardness (quality-adjusted)

    Returns:
        Tuple of (scaled_mark, weights_dict)

    Example:
        book_mark=20, diameter=275mm, janka=750 (Yellow Pine)

        Hardwood table: 17s
        Medium table:   14s
        Softwood table: 11s

        Weights: 25% soft, 50% med, 25% hard
        Result: (11x0.25) + (14x0.50) + (17x0.25) = 14.0s
    """
    # Get scaled values from each table
    hard_value, _ = scale_mark_qaa(book_mark, target_diameter, 'hardwood')
    med_value, _ = scale_mark_qaa(book_mark, target_diameter, 'medium')
    soft_value, _ = scale_mark_qaa(book_mark, target_diameter, 'softwood')

    # Calculate blend weights
    weights = calculate_hardness_blend_weights(effective_janka)

    # Interpolate
    result = (
        soft_value * weights['softwood'] +
        med_value * weights['medium'] +
        hard_value * weights['hardwood']
    )

    return result, weights


def scale_mark_qaa(
    book_mark_300mm: float,
    target_diameter: float,
    wood_type: str = 'hardwood'
) -> Tuple[float, str]:
    """
    Scale a handicap mark using QAA empirical tables.

    Args:
        book_mark_300mm: Competitor's book mark at 300mm standard
        target_diameter: Target diameter in mm (225, 250, 275, 300, 325, 350)
        wood_type: 'hardwood', 'medium', or 'softwood'

    Returns:
        Tuple of (scaled_mark, explanation)

    Example:
        >>> scale_mark_qaa(20, 275, 'hardwood')
        (17, "QAA table: 20s @ 300mm -> 17s @ 275mm (hardwood)")
    """
    # Select appropriate table
    if wood_type == 'softwood':
        table = QAA_UH_SB_SOFTWOOD
    elif wood_type == 'medium':
        table = QAA_UH_SB_MEDIUM_WOOD
    else:  # hardwood or unknown
        table = QAA_UH_SB_HARDWOOD

    # Round book mark to nearest integer for table lookup
    book_mark_int = round(book_mark_300mm)

    # Clamp to valid range (3-43 for Open events)
    if book_mark_int < 3:
        book_mark_int = 3
    elif book_mark_int > 43:
        book_mark_int = 43

    # Round target diameter to nearest standard size
    standard_diameters = [225, 250, 275, 300, 325, 350]
    target_rounded = min(standard_diameters, key=lambda x: abs(x - target_diameter))

    # Look up scaled mark
    if book_mark_int in table and target_rounded in table[book_mark_int]:
        scaled_mark = table[book_mark_int][target_rounded]
        explanation = f"QAA table: {book_mark_int}s @ 300mm = {scaled_mark}s @ {target_rounded}mm ({wood_type})"

        # If target wasn't exactly on standard, note the approximation
        if abs(target_rounded - target_diameter) > 5:
            explanation += f" [target {target_diameter}mm rounded to {target_rounded}mm]"

        return float(scaled_mark), explanation

    else:
        # Fallback: use proportional scaling if outside table range
        # This shouldn't happen for normal Open events (3-43 range)
        ratio = target_diameter / 300.0
        scaled_mark = book_mark_300mm * ratio
        explanation = f"Proportional scaling: {book_mark_300mm:.1f}s x {ratio:.3f} = {scaled_mark:.1f}s (outside QAA table range)"
        return scaled_mark, explanation


def scale_time_qaa(
    historical_time: float,
    historical_diameter: float,
    target_diameter: float,
    species_code: str,
    quality: int = 5,
    wood_df: Optional[pd.DataFrame] = None
) -> Tuple[float, str]:
    """
    Scale a historical TIME (not mark) to different diameter using QAA tables.

    This converts a time to a mark at historical diameter, scales the mark using
    interpolated QAA tables (based on effective Janka hardness), then returns
    the scaled mark as a time prediction.

    Args:
        historical_time: Actual time in seconds (e.g., 27.5s)
        historical_diameter: Diameter of historical performance (mm)
        target_diameter: Target diameter for prediction (mm)
        species_code: Wood species code
        quality: Wood quality 1-10 (1=very soft, 5=average, 10=very hard)

    Returns:
        Tuple of (scaled_time, explanation)

    Example:
        >>> scale_time_qaa(27.5, 300, 275, 'WP', quality=5)
        (24.0, "QAA: 28s@300mm = 24s@275mm (50% medium, 50% soft)")

    Note:
        Uses QAA empirical tables (150+ years validation) with interpolation
        based on effective Janka hardness (species baseline + quality adjustment).

        Quality adjustment examples:
        - Hard White Pine (quality=9): Scales more like Yellow Pine
        - Soft Rock Maple (quality=2): Scales more like White Pine
    """
    # Load wood database to get Janka hardness
    if wood_df is None:
        from woodchopping.data import load_wood_data
        wood_df = load_wood_data()

    if wood_df is None or wood_df.empty:
        # Fallback: use medium table if database unavailable
        wood_type = 'medium'
        scaled_mark, table_explanation = scale_mark_qaa(
            historical_time,
            target_diameter,
            wood_type
        )
        explanation = f"QAA scaling: {historical_time:.1f}s @ {historical_diameter}mm = {scaled_mark:.1f}s @ {target_diameter}mm (medium - database unavailable)"
        return scaled_mark, explanation

    # Calculate effective Janka hardness (species + quality)
    effective_janka = calculate_effective_janka_hardness(species_code, quality, wood_df)

    # Convert time to equivalent book mark at 300mm standard
    # (Assume historical time IS the book mark if at 300mm)
    # This is an approximation - in real QAA system, marks are manually adjusted
    book_mark_300mm = historical_time

    # Interpolate between QAA tables based on effective hardness
    scaled_mark, weights = interpolate_qaa_tables(
        book_mark_300mm,
        target_diameter,
        effective_janka
    )

    # Build explanation showing blend
    blend_parts = []
    if weights['softwood'] > 0.01:
        blend_parts.append(f"{weights['softwood']*100:.0f}% soft")
    if weights['medium'] > 0.01:
        blend_parts.append(f"{weights['medium']*100:.0f}% med")
    if weights['hardwood'] > 0.01:
        blend_parts.append(f"{weights['hardwood']*100:.0f}% hard")

    blend_str = ", ".join(blend_parts)

    # Include effective Janka in explanation for transparency
    explanation = f"QAA scaling: {historical_time:.1f}s @ {historical_diameter}mm = {scaled_mark:.1f}s @ {target_diameter}mm ({blend_str}, {effective_janka:.0f} Janka)"

    return scaled_mark, explanation


def get_qaa_panel_mark(event_code: str, division: str = 'open') -> int:
    """
    Get QAA panel mark (initial assignment for new competitors).

    Args:
        event_code: 'SB' or 'UH'
        division: 'open', 'novice', 'junior', 'veterans', 'womens'

    Returns:
        Initial panel mark in seconds

    QAA Panel Marks (Page 2):
        - Open UH/SB: 15 seconds
        - Novice: 35 seconds
        - Junior: 15 seconds
        - Veterans: Set by committee (default 35)
        - Women's UH: 35 seconds
    """
    if division.lower() in ['novice', 'veterans', 'womens', 'women']:
        return 35
    elif division.lower() == 'junior':
        return 15
    else:  # open
        return 15


# ============================================================================
# COMPARISON FUNCTIONS (For validation against STRATHEX formula)
# ============================================================================

def compare_scaling_methods(
    book_mark: float,
    target_diameter: float,
    species_code: str
) -> pd.DataFrame:
    """
    Compare QAA table scaling vs STRATHEX power-law formula.

    Useful for validation and understanding differences between methods.

    Args:
        book_mark: Book mark at 300mm standard
        target_diameter: Target diameter
        species_code: Wood species

    Returns:
        DataFrame comparing both methods
    """
    # QAA method
    wood_type = get_wood_type_category(species_code)
    qaa_scaled, qaa_exp = scale_mark_qaa(book_mark, target_diameter, wood_type)

    # STRATHEX method (for comparison)
    # Import here to avoid circular dependency
    from woodchopping.predictions.diameter_scaling import scale_time
    strathex_scaled, strathex_meta = scale_time(book_mark, 300, target_diameter)

    comparison = pd.DataFrame({
        'Method': ['QAA Empirical Table', 'STRATHEX Power-Law'],
        'Scaled Mark': [qaa_scaled, strathex_scaled],
        'Difference (s)': [0, strathex_scaled - qaa_scaled],
        'Difference (%)': [0, ((strathex_scaled - qaa_scaled) / qaa_scaled) * 100],
        'Explanation': [qaa_exp, strathex_meta.explanation]
    })

    return comparison
