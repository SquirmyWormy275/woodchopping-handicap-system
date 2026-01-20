"""
QAA legacy handicap calculation (marks-only).

Uses QAA handicap tables to scale book marks from 300mm standard
to the target diameter for SB/UH events.
"""

from typing import Dict, List, Optional, Any
import pandas as pd

from woodchopping.data import load_wood_data, standardize_results_data
from woodchopping.predictions.baseline import predict_baseline_v2_hybrid
from woodchopping.predictions.qaa_scaling import (
    calculate_effective_janka_hardness,
    interpolate_qaa_tables
)


def calculate_qaa_legacy_marks(
    heat_assignment_df: pd.DataFrame,
    species: str,
    diameter: float,
    quality: int,
    event_code: str,
    results_df: pd.DataFrame
) -> Optional[List[Dict[str, Any]]]:
    """
    Calculate QAA legacy marks for SB/UH events only.

    Steps:
    1) Predict a competitor "book mark" at 300mm (Baseline V2, quality=5).
    2) Scale that mark to the target diameter using QAA tables.
    3) Round marks to the nearest second (AAA rule).

    Returns:
        List of dicts with:
        - name
        - mark (integer)
        - book_mark_300
        - scaled_mark
        - explanation
    """
    if heat_assignment_df is None or heat_assignment_df.empty:
        return None

    event_code = str(event_code).strip().upper()
    if event_code not in ("SB", "UH"):
        return None

    results_df, _ = standardize_results_data(results_df)
    wood_df = load_wood_data()

    quality_val = int(quality) if quality is not None else 5
    quality_val = max(1, min(10, quality_val))

    # QAA open events: 300mm standard, max 43s book mark
    standard_diameter = 300.0
    max_book_mark = 43

    effective_janka = calculate_effective_janka_hardness(species, quality_val, wood_df)

    results: List[Dict[str, Any]] = []

    for _, row in heat_assignment_df.iterrows():
        comp_name = row.get("competitor_name")
        if not comp_name:
            continue

        base_time, conf, expl, meta = predict_baseline_v2_hybrid(
            competitor_name=comp_name,
            species=species,
            diameter=standard_diameter,
            quality=5,  # book marks assume standard conditions
            event_code=event_code,
            results_df=results_df,
            wood_df=wood_df,
            tournament_results=None,
            enable_calibration=False
        )

        if base_time is None:
            continue

        book_mark_300 = max(3.0, min(float(base_time), float(max_book_mark)))

        scaled_mark, weights = interpolate_qaa_tables(
            book_mark_300,
            float(diameter),
            effective_janka
        )

        mark = int(round(scaled_mark))
        mark = max(3, mark)

        blend_parts = []
        if weights.get('softwood', 0) > 0.01:
            blend_parts.append(f"{weights['softwood']*100:.0f}% soft")
        if weights.get('medium', 0) > 0.01:
            blend_parts.append(f"{weights['medium']*100:.0f}% med")
        if weights.get('hardwood', 0) > 0.01:
            blend_parts.append(f"{weights['hardwood']*100:.0f}% hard")

        blend_str = ", ".join(blend_parts) if blend_parts else "mixed"

        explanation = (
            f"QAA: {book_mark_300:.1f}s @ 300mm -> {scaled_mark:.1f}s @ {float(diameter):.0f}mm "
            f"({blend_str}, {effective_janka:.0f} Janka)"
        )

        results.append({
            'name': comp_name,
            'mark': mark,
            'book_mark_300': book_mark_300,
            'scaled_mark': scaled_mark,
            'explanation': explanation,
            'method_used': 'QAA'
        })

    return results
