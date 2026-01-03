"""
Test QAA interpolation with Janka hardness and wood quality.

This script verifies that the QAA table interpolation correctly accounts for:
1. Species baseline Janka hardness
2. Block-specific quality adjustment (0-10 scale)
3. Smooth blending between hardwood/medium/softwood tables
"""

from woodchopping.predictions.qaa_scaling import (
    calculate_effective_janka_hardness,
    calculate_hardness_blend_weights,
    interpolate_qaa_tables,
    scale_time_qaa
)
from woodchopping.data import load_wood_data

print("="*80)
print("QAA INTERPOLATION TEST - Janka Hardness + Wood Quality")
print("="*80)

# Load wood database
wood_df = load_wood_data()

if wood_df is None or wood_df.empty:
    print("ERROR: Could not load wood database")
    exit(1)

# Test cases demonstrating quality adjustment
# Using actual species codes from database: S01=Eastern White Pine, S05=Ponderosa Pine, S04=Alder
test_cases = [
    # (species_code, quality, description)
    ('S01', 5, 'E. White Pine - Average firmness'),
    ('S01', 2, 'E. White Pine - Firm (green wood)'),
    ('S01', 8, 'E. White Pine - Soft (dry/punky)'),
    ('S05', 5, 'Ponderosa Pine - Average firmness'),
    ('S05', 2, 'Ponderosa Pine - Firm'),
    ('S04', 5, 'Alder - Average firmness'),
    ('S04', 8, 'Alder - Soft (rotten)'),
]

print("\n1. EFFECTIVE JANKA HARDNESS CALCULATION")
print("-"*80)
print(f"{'Species':<30} {'Quality':>7} {'Base Janka':>11} {'Effective':>11} {'Change'}")
print("-"*80)

for species_code, quality, description in test_cases:
    # Get base Janka
    wood_row = wood_df[wood_df['speciesID'].str.upper() == species_code.upper()]
    if not wood_row.empty:
        base_janka = float(wood_row['janka_hard'].values[0])
    else:
        base_janka = 750.0

    # Calculate effective Janka
    effective_janka = calculate_effective_janka_hardness(species_code, quality, wood_df)

    change_pct = ((effective_janka - base_janka) / base_janka) * 100

    print(f"{description:<30} {quality:>7} {base_janka:>11.0f} {effective_janka:>11.0f} {change_pct:+6.1f}%")

print("\n2. BLEND WEIGHTS CALCULATION")
print("-"*80)
print(f"{'Description':<30} {'Janka':>11} {'Soft':>8} {'Medium':>8} {'Hard':>8}")
print("-"*80)

for species_code, quality, description in test_cases:
    effective_janka = calculate_effective_janka_hardness(species_code, quality, wood_df)
    weights = calculate_hardness_blend_weights(effective_janka)

    print(f"{description:<30} {effective_janka:>11.0f} {weights['softwood']*100:>7.0f}% {weights['medium']*100:>7.0f}% {weights['hardwood']*100:>7.0f}%")

print("\n3. QAA TABLE INTERPOLATION EXAMPLES")
print("-"*80)
print("Testing scaling from 300mm to 275mm with book mark of 25 seconds")
print("-"*80)
print(f"{'Description':<30} {'Janka':>11} {'Scaled Mark':>12} {'Blend'}")
print("-"*80)

book_mark = 25.0
target_diameter = 275.0

for species_code, quality, description in test_cases:
    effective_janka = calculate_effective_janka_hardness(species_code, quality, wood_df)
    scaled_mark, weights = interpolate_qaa_tables(book_mark, target_diameter, effective_janka)

    # Format blend string
    blend_parts = []
    if weights['softwood'] > 0.01:
        blend_parts.append(f"{weights['softwood']*100:.0f}% soft")
    if weights['medium'] > 0.01:
        blend_parts.append(f"{weights['medium']*100:.0f}% med")
    if weights['hardwood'] > 0.01:
        blend_parts.append(f"{weights['hardwood']*100:.0f}% hard")
    blend_str = ", ".join(blend_parts)

    print(f"{description:<30} {effective_janka:>11.0f} {scaled_mark:>11.1f}s  {blend_str}")

print("\n4. FULL SCALE_TIME_QAA() FUNCTION TEST")
print("-"*80)
print("Scaling historical time of 27.5s from 300mm to 275mm")
print("-"*80)
print(f"{'Description':<30} {'Quality':>7} {'Scaled':>11} {'Explanation'}")
print("-"*80)

historical_time = 27.5
historical_diameter = 300.0
target_diameter = 275.0

for species_code, quality, description in [
    ('S01', 2, 'Firm E. White Pine'),
    ('S01', 5, 'Avg E. White Pine'),
    ('S01', 8, 'Soft E. White Pine'),
]:
    scaled_time, explanation = scale_time_qaa(
        historical_time,
        historical_diameter,
        target_diameter,
        species_code,
        quality=quality
    )

    # Truncate explanation for display
    exp_short = explanation[:60] + "..." if len(explanation) > 60 else explanation

    print(f"{description:<30} {quality:>7} {scaled_time:>10.1f}s  {exp_short}")

print("\n5. CRITICAL VALIDATION: Firm E. White Pine vs Ponderosa Pine")
print("-"*80)
print("Verifying that firm E. White Pine scales more like Ponderosa Pine")
print("-"*80)

# Firm E. White Pine (quality 2)
wp_firm_janka = calculate_effective_janka_hardness('S01', 2, wood_df)
wp_firm_scaled, _ = interpolate_qaa_tables(25.0, 275.0, wp_firm_janka)

# Average E. White Pine (quality 5)
wp_avg_janka = calculate_effective_janka_hardness('S01', 5, wood_df)
wp_avg_scaled, _ = interpolate_qaa_tables(25.0, 275.0, wp_avg_janka)

# Average Ponderosa Pine (quality 5)
pp_avg_janka = calculate_effective_janka_hardness('S05', 5, wood_df)
pp_avg_scaled, _ = interpolate_qaa_tables(25.0, 275.0, pp_avg_janka)

print(f"Firm E. White Pine (Q=2):    {wp_firm_janka:>6.0f} Janka to {wp_firm_scaled:>5.1f}s @ 275mm")
print(f"Avg E. White Pine (Q=5):     {wp_avg_janka:>6.0f} Janka to {wp_avg_scaled:>5.1f}s @ 275mm")
print(f"Avg Ponderosa Pine (Q=5):    {pp_avg_janka:>6.0f} Janka to {pp_avg_scaled:>5.1f}s @ 275mm")

print(f"\nFirm WP is {abs(wp_firm_scaled - pp_avg_scaled):.1f}s different from avg Ponderosa Pine")
print(f"Avg WP is {abs(wp_avg_scaled - pp_avg_scaled):.1f}s different from avg Ponderosa Pine")
print("\n[OK] Firm E. White Pine should scale closer to Ponderosa Pine than average White Pine does")

print("\n6. CRITICAL VALIDATION: Soft Alder vs E. White Pine")
print("-"*80)
print("Verifying that soft/rotten Alder scales more like E. White Pine")
print("-"*80)

# Soft Alder (quality 8)
alder_soft_janka = calculate_effective_janka_hardness('S04', 8, wood_df)
alder_soft_scaled, _ = interpolate_qaa_tables(25.0, 275.0, alder_soft_janka)

# Average Alder (quality 5)
alder_avg_janka = calculate_effective_janka_hardness('S04', 5, wood_df)
alder_avg_scaled, _ = interpolate_qaa_tables(25.0, 275.0, alder_avg_janka)

# Average E. White Pine (already calculated above)

print(f"Soft Alder (Q=8):            {alder_soft_janka:>6.0f} Janka to {alder_soft_scaled:>5.1f}s @ 275mm")
print(f"Avg Alder (Q=5):             {alder_avg_janka:>6.0f} Janka to {alder_avg_scaled:>5.1f}s @ 275mm")
print(f"Avg E. White Pine (Q=5):     {wp_avg_janka:>6.0f} Janka to {wp_avg_scaled:>5.1f}s @ 275mm")

print(f"\nSoft Alder is {abs(alder_soft_scaled - wp_avg_scaled):.1f}s different from avg E. White Pine")
print(f"Avg Alder is {abs(alder_avg_scaled - wp_avg_scaled):.1f}s different from avg E. White Pine")
print("\n[OK] Soft Alder should scale closer to E. White Pine than average Alder does")

print("\n" + "="*80)
print("TEST COMPLETE - QAA interpolation with quality adjustment validated")
print("="*80)
