# Woodchopping Handicap Management System

## Executive Summary
In the sport of woodchopping, there are two types of races that can occur. The first type of race is called a 'Championship' race and occurs when all competitors start at the same time on the mark of 'Go.' The second type of race more common in Australia and New Zealand is a handicapped race. A properly handicapped race allows competitors of all ability levels to have roughly the same chance of winning the race. The idea is that it keeps top-tier competitors from growing complacent, and allows newer competitors the potential to win large amounts of money that would allow them to invest in better equipment and travel to more competitions. In Australia, the purses on prestigious handicap races can reach into the tens of thousands of dollars. The tangibility of the handicap 'marks' also allows for unbelievably lucrative sports betting to occur, and many venues make a significant portion of thier revenue through the proceeds. In the United States, there is currently no handicapping system. This forces mid-tier competitors to compete at a loss for years on end, and allows for top-tier competitors to grow complacent. While the Australians have a good framework for handicapping, it is largely subjective. Given the stakes and the amount of money involved, this system becomes highly political. According to the Australian Axemen Assosciation's Competition Rulebook, handicapping is determined by tht following means:

"12. Competitors in Events conducted in heats or divisions shall be handicapped at the discretion of the Committee or, if appointed, the Handicapper. 13. Handicaps will be determined based on form, inherent ability and performances and such other information as may be deemed by the Committee, or Handicapper, to be relevant. "

While this framework provides guidance, the actual calculation of handicap marks requires predicting each competitor's time to sever a specific block of wood. This prediction must account for multiple interacting factors: the competitor's historical performance across different wood species and diameters, the variations in the quality of wood. In a perfectly handicapped system, all of the competitors should sever their blocks at the same time. Since the Australian's have over 150 years of experience handicapping competitors, they are usually fairly accurate at determining marks. Having a more objective system that accurately considers the various factors that impact the time to sever the block would eliminate the guesswork and bias inherhent to the handicapper, and would allow the system to be implemented in countries that lack the instituitonal knowledge. This system would hopefully help eliminate the common practices of "foxing" or "laying down", where competitors perform deliberately poorly in order to get their handicap lowered prior to a bigger race. In short, a proper handicap calculator would create fairer competition, more exciting races for the spectators, and better distribution of purses. This system not only calculates handicap marks, but achieves measurable fairness through Monte Carlo simulation validation,  eliminates subjective bias through transparent calculations based on historical data, and provides the institutional knowledge needed to implement professional handicapping in countries like the United States.'''

### Examples of handicapped Woodchopping events at the Sydeny Royal Easter Show can be found below:

[Standing Block](https://www.youtube.com/watch?v=N8Mt_Abhgjo)

[Underhand](https://www.youtube.com/watch?v=dx8OyDR5fg0&list=PLpuG2w7J_GP62e-mpzuWO41vwclsnQL1e&index=12)

[3-Board Jigger](https://www.youtube.com/watch?v=iSq8cik-EhI&list=PLpuG2w7J_GP62e-mpzuWO41vwclsnQL1e)

## Statement of Scope

### Project Objectives  

This project delivers a data-driven Wood Chopping Handicap Management System that enables fair, objective handicapping for woodchopping competitions compliant with Australian Axemen's Association (AAA) Competition Rules. The system combines historical performance analysis, AI-enhanced prediction modeling, and Monte Carlo simulation validation to calculate handicap marks that give all competitors—regardless of skill level—equal probability of winning. 

### Core Capabilities:

**Competitor Management** - Officials can load, add, remove, and select competitors from a master roster Excel sheet, managing both full rosters and individual heat assignments.
Wood Characteristics Integration - The system accounts for wood species, block diameter (in mm), and quality rating (0-10 scale), recognizing that wood characteristics significantly impact cutting times and handicap fairness.

**Dual Prediction System** - The system employs THREE complementary prediction methods working in parallel:
1. **ML Model (XGBoost)** - Advanced machine learning trained on historical data (R²=0.993, MAE=2.3s) analyzing 6 key features: competitor averages, wood hardness/density, diameter, experience, and event type
2. **AI-Enhanced (LLM)** - Ollama (qwen2.5:7b model) provides intelligent reasoning about wood quality adjustments and contextual factors
3. **Baseline Statistical** - Weighted historical averages ensuring always-available fallback predictions

Each prediction includes confidence levels and explanations. The system automatically selects the most accurate available method (priority: ML > LLM > Baseline) for handicap marks while displaying all three predictions side-by-side for transparency. AI-powered analysis explains differences between methods and provides judge recommendations.

**Fair Handicap Calculation** - The system implements a critical innovation: absolute variance modeling (±3 seconds for all competitors) rather than proportional variance. This breakthrough ensures true fairness—testing revealed proportional variance creates biased results (31% vs 6.7% win rates), while absolute variance achieves equal win probability across all skill levels because real-world factors (technique consistency, wood grain variation, equipment) affect competitors equally in absolute terms, not proportionally.

**Monte Carlo Validation** - Every handicap calculation can be validated through extensive Monte Carlo simulation (250,000 race iterations), providing statistical proof of fairness through win probability distributions, podium frequency analysis, and finish spread assessment.
AAA Rules Compliance - All calculations adhere to official competition standards: 3-second minimum mark, 180-second maximum time limit, marks rounded up to whole seconds, proper front marker/back marker dynamics.

**Results Management** - Heat results can be saved back to Excel, building the historical database that improves future predictions.

### Value Add

Timbersports athletes and officials will finally have the ability to implement a professional handicap system without needing a century-plus of institutional knowledge that only the Australians and New Zealanders currently possess. By replacing subjective handicapping with objective, data-driven calculations validated through Monte Carlo simulation, the system eliminates the political bias that plague discretionary systems, while creating verifiably fair races where a novice competitor with three documented times has an equal shot at winning against the best in the world. This means competitive events can offer significant purses with confidence that the handicaps are actually fair, which should drive more exciting races for spectators and better prize distribution across all skill levels. The system handles all the complex mathematics and AI reasoning behind the scenes, giving officials quick, readable handicap marks with full transparency about confidence levels and data sources. Countries without woodchopping handicapping infrastructure can now implement professional-grade systems immediately, and the discovery of absolute variance modeling represents a fundamental breakthrough in understanding what makes handicapping truly fair.

## Inputs, Processes, Outputs

### Step 1: Load Libraries
* **Description:** Import core libraries used throughout the program.
* **Inputs:** None for this step.
* **Processes:** Import `pandas`, `numpy`, `matplotlib`, `sys`, `math.ceil`, `openpyxl`, and `ProjectFunctions as pf`.
* **Outputs:** None for this step; proceed to **Step 2**.

### Step 2: Load Competitor Roster
* **Description:** Read the competitor roster from Excel into memory.
* **Inputs:** `woodchopping.xlsx`, sheet `"Competitor"`.
* **Processes:** Attempt to load into a DataFrame; on failure, build an empty DataFrame with the expected columns and print a descriptive message.
* **Outputs:** `comp_df` is available in memory; proceed to **Step 3**.

### Step 3: Initialize Program State
* **Description:** Create clean placeholders for the session.
* **Inputs:** None for this step.
* **Processes:** Initialize `wood_selection = {species: None, size_mm: None, quality: None, event: None}`; create empty `heat_assignment_df` and `heat_assignment_names`.
* **Outputs:** Internal state ready; proceed to **Step 4**.

### Step 4: Welcome Message
* **Description:** Display "Welcome to the Wood Chopping Handicap Management System".
* **Inputs:** None for this step.
* **Processes:** Print greeting and prepare to show the main menu options.
* **Outputs:** Display message; proceed to **Step 5**.

### Step 5: Main Menu Selection
* **Description:** Present the primary navigation loop for the main menu.
* **Inputs:** A number from **1–7** entered by the official.
* **Processes:** Evaluate the choice and route accordingly.
* **Outputs:**
  * For Option 1: **Competitor Selection Menu**, go to **Step 6**.
  * For Option 2: **Wood Characteristics Menu**, go to **Step 10**.
  * For Option 3: **Select Event (SB/UH)**, go to **Step 14**.
  * For Option 4: **View AI Enhanced Handicap Marks**, go to **Step 15**.
  * For Option 5: **Update Results tab with Heat Results**, go to **Step 17**.
  * For Option 6: **Reload Roster**, go to **Step 18**.
  * For Option 7: **Exit**, go to **Step 19**.

### Step 6: Competitor Selection Menu
* **Description:** Manage competitor roster and heat assignments.
* **Inputs:** A number from **1–5** entered by the official.
* **Processes:** Display sub-menu with options to select competitors for heat, add new competitor, view heat assignment, remove from heat, or return to main menu.
* **Outputs:**
  * For Option 1: **Select Competitors for Heat**, go to **Step 7**.
  * For Option 2: **Add New Competitor**, go to **Step 8**.
  * For Option 3: **View Heat Assignment**, go to **Step 9**.
  * For Option 4: **Remove from Heat**, return to this menu.
  * For Option 5: **Back to Main Menu**, return to **Step 5**.

### Step 7: Select Competitors for Heat
* **Description:** Choose competitors from roster for the current heat.
* **Inputs:** Index numbers entered one at a time; press Enter with no input when finished.
* **Processes:** Display numbered roster list; accept competitor selections; validate selections; build heat assignment list; reload roster to ensure latest data.
* **Outputs:** Updated `heat_assignment_df` and `heat_assignment_names`; display confirmation of selections; return to **Step 5**.

### Step 8: Add New Competitor to Roster
* **Description:** Register a new competitor with historical performance data.
* **Inputs:** Competitor name, country, state/province (optional), gender (optional); minimum 3 historical times with event type (SB/UH), time in seconds, wood species, diameter in mm, quality (0-10), and date (optional).
* **Processes:** Generate unique CompetitorID; validate inputs; save to `Competitor` sheet in Excel; prompt for historical times; save times to `Results` sheet with proper column mapping.
* **Outputs:** Updated `woodchopping.xlsx` with new competitor and historical data; reload `comp_df`; return to **Step 6**.

### Step 9: View Heat Assignment
* **Description:** Display current competitors assigned to the heat.
* **Inputs:** None for this step.
* **Processes:** Format and display list of selected competitors with country information; show total count.
* **Outputs:** Display heat assignment list; return to **Step 6**.

### Step 10: Wood Characteristics Menu
* **Description:** Configure wood specifications for handicap calculations.
* **Inputs:** A number from **1–4** entered by the official.
* **Processes:** Display current wood selection; present sub-menu options.
* **Outputs:**
  * For Option 1: **Select Wood Species**, go to **Step 11**.
  * For Option 2: **Enter Size (mm)**, go to **Step 12**.
  * For Option 3: **Enter Quality (0-10)**, go to **Step 13**.
  * For Option 4: **Back to Main Menu**, return to **Step 5**.

### Step 11: Select Wood Species
* **Description:** Choose wood species from available options.
* **Inputs:** Numeric selection from numbered species list.
* **Processes:** Load `wood` sheet from Excel; display species list; validate selection; update `wood_selection["species"]`.
* **Outputs:** Updated wood selection displayed; return to **Step 10**.

### Step 12: Enter Wood Size
* **Description:** Specify block diameter in millimeters.
* **Inputs:** Numeric value for diameter (e.g., 300).
* **Processes:** Validate numeric input; update `wood_selection["size_mm"]`.
* **Outputs:** Updated wood selection displayed; return to **Step 10**.

### Step 13: Enter Wood Quality
* **Description:** Rate wood quality on 0-10 scale (0=poor/rotten, 5=average, 10=extremely soft).
* **Inputs:** Integer from 0 to 10.
* **Processes:** Validate range; clamp to 0-10 bounds; update `wood_selection["quality"]`.
* **Outputs:** Updated wood selection displayed; return to **Step 10**.

### Step 14: Select Event Code
* **Description:** Choose event type (Standing Block or Underhand).
* **Inputs:** Text input: "SB" for Standing Block or "UH" for Underhand.
* **Processes:** Validate input matches "SB" or "UH"; update `wood_selection["event"]`.
* **Outputs:** Event confirmation displayed; return to **Step 5**.

### Step 15: View Dual Prediction Handicap Marks
* **Description:** Calculate and display handicap marks using all three prediction methods.
* **Inputs:** Press Enter to view comprehensive prediction comparison; choice to view AI analysis (Enter/n); choice to run Monte Carlo simulation (y/n).
* **Processes:** Validate heat assignment and wood selection are complete; load historical results from Excel; for each competitor, run all three prediction methods in parallel: (1) Train/use XGBoost ML model with 6 engineered features from historical data, (2) Call Ollama AI for quality-adjusted predictions, (3) Calculate statistical baseline; select best prediction for marks (priority: ML > LLM > Baseline); display all three predictions side-by-side with method availability summary; optionally generate AI analysis explaining prediction differences and reliability; offer Monte Carlo simulation (250,000 races) to assess fairness.
* **Outputs:** Display comprehensive prediction table showing Baseline, ML Model, and LLM Model times for each competitor alongside their handicap marks; indicate which method was used for marks; show prediction methods summary with availability and confidence levels; optionally display AI-generated analysis of prediction differences with judge recommendations; optionally display Monte Carlo simulation results with win probabilities and AI fairness assessment; go to **Step 16**.

### Step 16: Monte Carlo Simulation (Optional)
* **Description:** Validate handicap fairness through statistical simulation.
* **Inputs:** Confirmation to run simulation (y/n).
* **Processes:** Simulate 250,000 races with ±3 second absolute performance variation; track win rates, finish positions, and finish time spreads for all competitors; calculate statistical metrics; send results to Ollama AI for expert assessment of fairness.
* **Outputs:** Display simulation summary with win percentages, average finish positions, finish spread statistics, text-based visualization of win rates, and AI assessment with fairness rating (Excellent/Very Good/Good/Fair/Poor/Unacceptable) and recommendations; return to **Step 5**.

### Step 17: Update Results Tab with Heat Results
* **Description:** Append actual competition times to historical results.
* **Inputs:** Heat ID (optional); actual time in seconds for each competitor in heat (press Enter to skip competitor).
* **Processes:** Validate event is selected; prompt for Heat ID (auto-generate if blank); collect times for each competitor; convert competitor names to IDs using bidirectional lookup; save to `Results` sheet in Excel with timestamp.
* **Outputs:** Updated `woodchopping.xlsx` Results sheet; confirmation message; return to **Step 5**.

### Step 18: Reload Roster from Excel
* **Description:** Refresh competitor list from Excel file.
* **Inputs:** None for this step.
* **Processes:** Re-execute `pf.load_competitors_df()`; clear current heat assignment for data integrity.
* **Outputs:** Updated `comp_df`; empty `heat_assignment_df` and `heat_assignment_names`; confirmation messages; return to **Step 5**.

### Step 19: Exit Program
* **Description:** Terminate the application.
* **Inputs:** None for this step.
* **Processes:** Print "Goodbye." message; break main loop.
* **Outputs:** Program terminates; session ends.

## Function Dictionary
| Function | Parameters | Purpose |
|:---|:---|:---|
| `detect_results_sheet` | `Workbook wb` | Return the Results sheet from workbook; create it with proper headers if missing. |
| `validate_heat_data` | `DataFrame heat_assignment_df`, `dict wood_selection` | Helper function to validate that heat data is complete (competitors selected, wood species/size set, event selected); return boolean. |
| `add_competitor_with_times` | *(none)* | Add a new competitor to roster; prompt for basic info (name, country, state, gender); generate CompetitorID; call function to add historical times; save to Excel. |
| `add_historical_times_for_competitor` | `str competitor_name` | Prompt judge to enter historical competition times (minimum 3); for each time, collect event type, time, wood species, size, quality, and optional date; save to results sheet. |
| `append_results_to_excel` | `DataFrame heat_assignment_df`, `dict wood_selection` | Prompt for heat_id and actual times for each competitor; convert names to IDs; append rows to Results sheet in Excel; handle workbook creation if needed. |
| `calculate_ai_enhanced_handicaps` | `DataFrame heat_assignment_df`, `str species`, `float diameter`, `int quality`, `str event_code`, `DataFrame results_df`, `callable progress_callback` | Calculate handicaps using dual prediction system; run all three prediction methods (Baseline, ML, LLM) in parallel for each competitor; select best prediction for marks; store all predictions for display; return list of dicts with marks and complete prediction data. |
| `call_ollama` | `str prompt`, `str model` (default: "qwen2.5:7b") | Send prompt to local Ollama instance; return response text or None if error; handles connection errors with helpful messages. |
| `competitor_menu` | `DataFrame comp_df`, `DataFrame heat_assignment_df`, `list heat_assignment_names` | Present competitor menu with options to select competitors for heat, add new competitors, view heat assignment, remove from heat, or return to main menu. |
| `display_dual_predictions` | `list handicap_results`, `dict wood_selection` | Display handicap marks with all three prediction methods side-by-side; show Baseline, ML Model, and LLM Model times in tabular format; indicate which method was used for marks; display availability summary; offer optional AI analysis of prediction differences. |
| `display_feature_importance` | `XGBRegressor model`, `str event_code`, `list feature_names` | Extract and display feature importance scores from trained XGBoost model; sort features by importance; create visual bar chart with Unicode characters; store importance values in global variables for event-specific models; return None. |
| `engineer_features_for_ml` | `DataFrame results_df`, `DataFrame wood_df` | Engineer features for ML training from historical results; create competitor averages by event, encode event type, join wood properties (janka hardness, specific gravity), calculate experience; return DataFrame with features ready for training. |
| `enter_wood_quality` | `dict wood_selection` | Prompt for wood quality rating as integer 0-10; clamp to valid range; update wood_selection dictionary and display confirmation. |
| `enter_wood_size_mm` | `dict wood_selection` | Prompt for block diameter in mm; validate input; update wood_selection dictionary and display confirmation. |
| `format_wood` | `dict ws` | Generate and display formatted header showing current wood selection (species, diameter, quality). |
| `generate_prediction_analysis_llm` | `list all_competitors_predictions`, `dict wood_selection` | Use LLM to analyze differences between ML and LLM predictions; build concise summary of prediction comparison; generate natural language analysis covering agreement/divergence, reasons for differences, reliability assessment, and judge recommendations. |
| `generate_simulation_summary` | `dict analysis` | Generate human-readable summary of simulation results; format finish spreads, win probabilities, average finish positions, and front/back marker analysis into text. |
| `get_all_predictions` | `str competitor_name`, `str species`, `float diameter`, `int quality`, `str event_code`, `DataFrame results_df` | Get predictions from all three methods for a single competitor; run Baseline statistical calculation, ML model prediction, and LLM prediction in parallel; return dict with all three predictions including times, confidence levels, explanations, and error states. |
| `get_ai_assessment_of_handicaps` | `dict analysis` | Use LLM to provide expert assessment of handicap fairness; calculate fairness metrics; rate as Excellent/Good/Fair/Poor/Unacceptable based on win rate spreads; provide fallback assessment if LLM unavailable. |
| `get_competitor_historical_times_flexible` | `str competitor_name`, `str species`, `str event_code`, `DataFrame results_df` | Get historical times with cascading fallback logic: try exact match (competitor+species+event), then competitor+event (any species); return tuple of (times list, data_source_description). |
| `get_competitor_id_name_mapping` | *(none)* | Load competitor data and return two dictionaries: id_to_name and name_to_id for converting between competitor IDs and names. |
| `get_event_baseline_flexible` | `str species`, `float diameter`, `str event_code`, `DataFrame results_df` | Calculate baseline time with cascading fallback: try species+diameter+event, then diameter+event, then event only; return tuple of (average_time or None, data_source_description). |
| `load_competitors_df` | *(none)* | Load the roster from Excel into a DataFrame; standardize column names; on error, print message and return empty DataFrame with expected columns. |
| `load_results_df` | *(none)* | Load the Results sheet as a DataFrame; map Excel column names to expected names; convert competitor IDs to names; returns empty DataFrame if missing. |
| `load_wood_data` | *(none)* | Load wood species data from Excel into a DataFrame; on error, return empty DataFrame with expected columns. |
| `perform_cross_validation` | `DataFrame X`, `Series y`, `dict model_params`, `int cv_folds` (default: 5) | Perform k-fold cross-validation to estimate ML model accuracy; split data into folds; train and evaluate on each fold; calculate MAE and R² for each fold; return dict with mean and standard deviation for both metrics plus individual fold scores. |
| `predict_competitor_time_with_ai` | `str competitor_name`, `str species`, `float diameter`, `int quality`, `str event_code`, `DataFrame results_df` | Predict competitor's time using historical data plus LLM reasoning for quality adjustment; weight recent performances; use fallback logic for sparse data; return tuple of (predicted_time, confidence_level, explanation). |
| `predict_time_ml` | `str competitor_name`, `str species`, `float diameter`, `int quality`, `str event_code`, `DataFrame results_df`, `DataFrame wood_df` | Predict time using event-specific trained XGBoost ML model; select SB or UH model based on event_code; load or train models with feature engineering; calculate competitor statistics; prepare 6-feature vector; make prediction with sanity checks; return tuple of (predicted_time, confidence, explanation) or error details. |
| `remove_from_heat` | `DataFrame heat_df`, `list heat_names` | Allow removal of competitor from heat assignment (not from roster) by selecting from numbered list. |
| `run_monte_carlo_simulation` | `list competitors_with_marks`, `int num_simulations` (default: 250000) | Run Monte Carlo simulation to assess handicap fairness; track winner counts, podium finishes, finish positions, and finish spreads; return comprehensive analysis dictionary with statistics. |
| `save_time_to_results` | `str event`, `str name`, `str species`, `float size`, `int quality`, `float time`, `str heat_id`, `str timestamp` | Helper function to save a single time entry to results sheet; converts name to CompetitorID and appends row to Excel. |
| `select_best_prediction` | `dict all_predictions` | Select the best prediction from available methods using priority logic (ML > LLM > Baseline); return tuple of (predicted_time, method_name, confidence, explanation) for use in handicap mark calculation. |
| `select_competitors_for_heat` | `DataFrame comp_df` | Display roster with index numbers; allow judge to select competitors one at a time by entering numbers; return heat assignment DataFrame and list of selected names. |
| `select_event_code` | `dict wood_selection` | Prompt for event code (SB for Standing Block or UH for Underhand); validate input; update wood_selection dictionary. |
| `select_wood_species` | `dict wood_selection` | Display available wood species from Excel; accept numeric choice; update wood_selection dictionary and display confirmation. |
| `simulate_and_assess_handicaps` | `list competitors_with_marks`, `int num_simulations` (default: 250000) | Main function to run complete simulation and provide comprehensive assessment; run simulation, display results, visualize, and get AI assessment. |
| `simulate_single_race` | `list competitors_with_marks` | Simulate a single race with realistic absolute performance variation (±3 seconds); apply same variance to all competitors; calculate finish times accounting for handicaps; return finish results sorted by finish time. |
| `train_ml_model` | `DataFrame results_df`, `DataFrame wood_df`, `bool force_retrain`, `str event_code` | Train separate XGBoost regression models for SB and UH events; validate and clean data; engineer features; perform 5-fold cross-validation for each model; ensure minimum 15 records per event; train with optimized hyperparameters (n_estimators=100, max_depth=4); display CV results, training metrics (MAE, R²), and feature importance; cache models separately for reuse; return dict with 'SB' and 'UH' keys or None if insufficient data. |
| `validate_results_data` | `DataFrame results_df` | Validate and clean historical results data; check required columns exist; remove impossible times (<0s or >300s); remove invalid diameters (<150mm or >500mm); remove missing competitor names; validate event codes (SB/UH only); remove statistical outliers using 3x IQR method; return tuple of (cleaned_df, warnings_list). |
| `view_handicaps` | `DataFrame heat_assignment_df`, `dict wood_selection` | Calculate and display AI-enhanced handicap marks for the heat; load historical results; calculate predictions; display compact results with data sources and confidence levels; offer Monte Carlo simulation option. |
| `view_handicaps_menu` | `DataFrame heat_assignment_df`, `dict wood_selection` | Present handicap marks menu with options to view marks for current heat or return to main menu. |
| `view_heat_assignment` | `DataFrame heat_df`, `list heat_names` | Display current heat assignment showing all competitors currently selected for the heat with their countries. |
| `visualize_simulation_results` | `dict analysis` | Create simple text-based bar chart visualization of win distributions using Unicode block characters; scale bars to 40 characters max. |
| `wood_menu` | `dict wood_selection` | Present wood characteristics menu with options to select species, enter size, enter quality, or return to main menu; return updated wood_selection dictionary. |

## Dual Prediction System - Technical Implementation

### ML Model Architecture

**XGBoost Regressor Specifications:**
- **Algorithm:** Gradient Boosted Decision Trees
- **Model Strategy:** Separate models trained for Standing Block (SB) and Underhand (UH) events
- **Hyperparameters:**
  - n_estimators: 100 (number of boosting rounds)
  - max_depth: 4 (tree depth, optimized for small dataset)
  - learning_rate: 0.1
  - objective: reg:squarederror (regression task)
  - tree_method: hist (histogram-based algorithm)
  - random_state: 42 (reproducibility)

**Performance Metrics (validated training data):**
- **MAE (Mean Absolute Error):** 2.30 seconds (cross-validated)
- **R² Score:** 0.993 (99.3% variance explained)
- **Training Time:** ~0.5 seconds per model
- **Prediction Time:** <0.01 seconds per competitor
- **Cross-Validation:** 5-fold CV with MAE ± std reported

### Feature Engineering

The ML model uses 6 carefully engineered features:

1. **competitor_avg_time_by_event** (float)
   - Competitor's historical average time for specific event type
   - Strongest predictor of future performance
   - Fallback hierarchy: event-specific → all events → event baseline

2. **event_encoded** (int: 0 or 1)
   - Binary encoding: SB=0, UH=1
   - Captures fundamental technique differences between events

3. **size_mm** (float)
   - Block diameter in millimeters
   - Exponential impact on cutting time (cutting area ∝ diameter²)

4. **wood_janka_hardness** (float)
   - Janka hardness rating from wood properties sheet
   - Range: 1560-2620 for current species
   - Joined from wood.xlsx via speciesID

5. **wood_spec_gravity** (float)
   - Specific gravity (density) of wood species
   - Range: 0.34-0.40 for current species
   - Correlates with resistance to cutting

6. **competitor_experience** (int)
   - Count of historical events for competitor
   - Captures skill development and consistency improvements

### Model Enhancements

**1. Event-Specific Models (SB vs UH)**
- Separate XGBoost models trained independently for Standing Block and Underhand events
- Captures event-specific performance patterns and technique differences
- Each model optimized for its event's unique characteristics
- Automatic selection of correct model based on event_code parameter
- Reduces cross-event noise in predictions

**2. Cross-Validation for Accuracy Estimation**
- 5-fold stratified cross-validation performed during training
- Provides unbiased estimate of model generalization performance
- Reports Mean Absolute Error (MAE) and R² with standard deviations
- Example output: "CV MAE: 2.30s +/- 0.45s"
- Helps identify overfitting and assess prediction reliability

**3. Data Validation & Cleaning**
- Multi-stage validation pipeline before training:
  - **Required columns check:** Ensures all necessary data fields present
  - **Time range validation:** Removes impossible times (<0s or >300s)
  - **Diameter validation:** Removes invalid sizes (<150mm or >500mm)
  - **Missing data removal:** Filters records with missing competitor names
  - **Event code validation:** Ensures only SB/UH events (no invalid codes)
  - **Statistical outlier detection:** 3x IQR method removes extreme outliers
- Returns cleaned DataFrame + detailed warnings list
- Displays validation summary: "Valid records: 98 / 101 (3 removed)"

**4. Feature Importance Analysis**
- Extracts XGBoost built-in feature importance scores
- Displays visual bar chart after training each model
- Example output:
  ```
  [SB FEATURE IMPORTANCE]
  competitor_avg_time_by_event    0.523  ####################
  wood_janka_hardness             0.187  #######
  size_mm                         0.142  #####
  competitor_experience           0.089  ###
  wood_spec_gravity               0.041  #
  event_encoded                   0.018
  ```
- Helps understand which factors drive predictions
- Stored globally for inspection and debugging

### Prediction Method Priority

**Automatic Selection Logic:**
```
IF ML prediction available AND valid:
    USE ML prediction  (highest accuracy)
ELSE IF LLM prediction available AND valid:
    USE LLM prediction  (quality-adjusted reasoning)
ELSE:
    USE Baseline prediction  (always available fallback)
```

**Confidence Levels:**
- **HIGH:** 5+ historical events for competitor in this event type
- **MEDIUM:** 1-4 historical events, or cross-event data used
- **LOW:** No competitor history, using event baseline

### Model Caching & Retraining

**Session-based Caching:**
- Separate models train once per session on first prediction request
- Cached in memory via global variables `_cached_ml_model_sb` and `_cached_ml_model_uh`
- Subsequent predictions use cached event-specific model (instant)
- Feature importance stored in `_feature_importance_sb` and `_feature_importance_uh`

**Automatic Retraining Triggers:**
- New results added to Excel (data size change detected)
- Force retrain via `force_retrain=True` parameter
- Model invalidation on new session start

**Minimum Data Requirements:**
- 15+ valid records per event required for event-specific training
- Data validation removes outliers and invalid entries before training
- Graceful degradation to LLM/Baseline if insufficient data

### Data Flow

```
Results.xlsx (101 records)
    ↓
load_results_df()
    ↓
engineer_features_for_ml()
    ↓
[Join wood properties, calculate averages, encode features]
    ↓
train_ml_model()
    ↓
XGBoost Training (100 trees, depth=4)
    ↓
Cache trained model in memory
    ↓
predict_time_ml() for each competitor
    ↓
6-feature vector → XGBoost → predicted time
    ↓
Combine with Baseline & LLM predictions
    ↓
select_best_prediction()
    ↓
Display all three methods side-by-side
```

### Dependencies

**Core ML Stack:**
- xgboost 3.1.2 (gradient boosting framework)
- scikit-learn 1.8.0 (metrics and utilities)
- scipy 1.16.3 (scientific computing)
- joblib 1.5.2 (model serialization, currently unused but available)

**Existing Dependencies:**
- pandas 2.2.3 (data manipulation)
- numpy 2.3.5 (numerical operations)
- requests 2.32.5 (Ollama API calls)

**Installation:**
```bash
pip install xgboost scikit-learn
```

### Error Handling & Fallbacks

**Graceful Degradation Chain:**

1. **ML Model Failures:**
   - Insufficient training data (<30 records) → Use LLM
   - Feature engineering errors → Use LLM
   - Prediction out of range (5-300s check) → Use LLM
   - XGBoost not installed → Use LLM

2. **LLM Failures:**
   - Ollama not running → Use Baseline
   - Connection timeout → Use Baseline
   - Invalid response format → Use Baseline

3. **Baseline:** Always succeeds (mathematical calculation)

### Display Format

**Prediction Comparison Table:**
```
Competitor Name                      Mark  Baseline   ML Model   LLM Model  Used
------------------------------------------------------------------------------------
Eric Hoberg                             3     34.2s     31.8s      32.8s     ML
Jane Smith                              7     29.5s     28.1s      28.9s     ML
Mike Johnson                           11     25.3s     24.7s      N/A       ML
```

**Method Summary:**
- Availability count for each method
- Training data size for ML
- Confidence rating (HIGH/MEDIUM/LOW)
- Primary method used for marks

**AI Analysis (Optional):**
- Comparative analysis of prediction differences
- Reliability assessment for current scenario
- Judge recommendations based on data quality
- Explanation of divergence factors

### Future Enhancements

**Model Improvements:**
- Save/load trained models to disk (pickle/joblib) for persistence across sessions
- Hyperparameter tuning based on dataset size and distribution
- Automated retraining schedule when new data accumulates

**Additional Features:**
- Time-weighted historical data (recent events weighted higher)
- Competitor improvement trends over time (performance trajectory modeling)
- Wood grain quality sub-classifications beyond 0-10 scale
- Environmental factors (temperature, humidity, altitude)
- Venue-specific adjustments (stand quality, surface type)

**Advanced ML:**
- Ensemble methods combining multiple model types (XGBoost + RandomForest + LightGBM)
- Neural network exploration for complex non-linear patterns
- Automated feature selection using recursive feature elimination
- Bayesian optimization for hyperparameter search

