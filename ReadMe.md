[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/kJdFmIaG)
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

**AI-Enhanced Time Prediction** - Using Ollama (qwen2.5:7b model), the system predicts competitor completion times by combining historical performance data with AI reasoning about wood quality adjustments. The system implements intelligent cascading fallback logic: exact match (competitor + species + event) → competitor + event (any species) → event baseline. Each prediction includes confidence level and clear explanation of data sources used.

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

### Step 15: View AI Enhanced Handicap Marks
* **Description:** Calculate and display handicap marks using AI predictions.
* **Inputs:** Press Enter to view marks (if data complete) or choice to run Monte Carlo simulation (y/n).
* **Processes:** Validate heat assignment and wood selection are complete; load historical results from Excel; call Ollama AI (qwen2.5:7b) for each competitor to predict time based on wood quality, species, diameter, and historical performance; calculate handicap marks with slowest predicted time = Mark 3; apply cascading fallback logic if AI unavailable; display marks with predicted times and confidence levels; offer Monte Carlo simulation (250,000 races) to assess fairness.
* **Outputs:** Display handicap marks table with competitor names, marks, predicted times, data sources, and confidence ratings; optionally display simulation results with win probabilities and AI fairness assessment; go to **Step 16**.

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
| `calculate_ai_enhanced_handicaps` | `DataFrame heat_assignment_df`, `str species`, `float diameter`, `int quality`, `str event_code`, `DataFrame results_df` | Calculate handicaps using AI-enhanced time predictions; predict time for each competitor; sort by predicted time (slowest first); calculate marks with slowest getting mark 3; apply 180-second maximum rule. |
| `call_ollama` | `str prompt`, `str model` (default: "qwen2.5:7b") | Send prompt to local Ollama instance; return response text or None if error; handles connection errors with helpful messages. |
| `competitor_menu` | `DataFrame comp_df`, `DataFrame heat_assignment_df`, `list heat_assignment_names` | Present competitor menu with options to select competitors for heat, add new competitors, view heat assignment, remove from heat, or return to main menu. |
| `enter_wood_quality` | `dict wood_selection` | Prompt for wood quality rating as integer 0-10; clamp to valid range; update wood_selection dictionary and display confirmation. |
| `enter_wood_size_mm` | `dict wood_selection` | Prompt for block diameter in mm; validate input; update wood_selection dictionary and display confirmation. |
| `format_wood` | `dict ws` | Generate and display formatted header showing current wood selection (species, diameter, quality). |
| `generate_simulation_summary` | `dict analysis` | Generate human-readable summary of simulation results; format finish spreads, win probabilities, average finish positions, and front/back marker analysis into text. |
| `get_ai_assessment_of_handicaps` | `dict analysis` | Use LLM to provide expert assessment of handicap fairness; calculate fairness metrics; rate as Excellent/Good/Fair/Poor/Unacceptable based on win rate spreads; provide fallback assessment if LLM unavailable. |
| `get_competitor_historical_times_flexible` | `str competitor_name`, `str species`, `str event_code`, `DataFrame results_df` | Get historical times with cascading fallback logic: try exact match (competitor+species+event), then competitor+event (any species); return tuple of (times list, data_source_description). |
| `get_competitor_id_name_mapping` | *(none)* | Load competitor data and return two dictionaries: id_to_name and name_to_id for converting between competitor IDs and names. |
| `get_event_baseline_flexible` | `str species`, `float diameter`, `str event_code`, `DataFrame results_df` | Calculate baseline time with cascading fallback: try species+diameter+event, then diameter+event, then event only; return tuple of (average_time or None, data_source_description). |
| `load_competitors_df` | *(none)* | Load the roster from Excel into a DataFrame; standardize column names; on error, print message and return empty DataFrame with expected columns. |
| `load_results_df` | *(none)* | Load the Results sheet as a DataFrame; map Excel column names to expected names; convert competitor IDs to names; returns empty DataFrame if missing. |
| `load_wood_data` | *(none)* | Load wood species data from Excel into a DataFrame; on error, return empty DataFrame with expected columns. |
| `predict_competitor_time_with_ai` | `str competitor_name`, `str species`, `float diameter`, `int quality`, `str event_code`, `DataFrame results_df` | Predict competitor's time using historical data plus LLM reasoning for quality adjustment; weight recent performances; use fallback logic for sparse data; return tuple of (predicted_time, confidence_level, explanation). |
| `remove_from_heat` | `DataFrame heat_df`, `list heat_names` | Allow removal of competitor from heat assignment (not from roster) by selecting from numbered list. |
| `run_monte_carlo_simulation` | `list competitors_with_marks`, `int num_simulations` (default: 250000) | Run Monte Carlo simulation to assess handicap fairness; track winner counts, podium finishes, finish positions, and finish spreads; return comprehensive analysis dictionary with statistics. |
| `save_time_to_results` | `str event`, `str name`, `str species`, `float size`, `int quality`, `float time`, `str heat_id`, `str timestamp` | Helper function to save a single time entry to results sheet; converts name to CompetitorID and appends row to Excel. |
| `select_competitors_for_heat` | `DataFrame comp_df` | Display roster with index numbers; allow judge to select competitors one at a time by entering numbers; return heat assignment DataFrame and list of selected names. |
| `select_event_code` | `dict wood_selection` | Prompt for event code (SB for Standing Block or UH for Underhand); validate input; update wood_selection dictionary. |
| `select_wood_species` | `dict wood_selection` | Display available wood species from Excel; accept numeric choice; update wood_selection dictionary and display confirmation. |
| `simulate_and_assess_handicaps` | `list competitors_with_marks`, `int num_simulations` (default: 250000) | Main function to run complete simulation and provide comprehensive assessment; run simulation, display results, visualize, and get AI assessment. |
| `simulate_single_race` | `list competitors_with_marks` | Simulate a single race with realistic absolute performance variation (±3 seconds); apply same variance to all competitors; calculate finish times accounting for handicaps; return finish results sorted by finish time. |
| `view_handicaps` | `DataFrame heat_assignment_df`, `dict wood_selection` | Calculate and display AI-enhanced handicap marks for the heat; load historical results; calculate predictions; display compact results with data sources and confidence levels; offer Monte Carlo simulation option. |
| `view_handicaps_menu` | `DataFrame heat_assignment_df`, `dict wood_selection` | Present handicap marks menu with options to view marks for current heat or return to main menu. |
| `view_heat_assignment` | `DataFrame heat_df`, `list heat_names` | Display current heat assignment showing all competitors currently selected for the heat with their countries. |
| `visualize_simulation_results` | `dict analysis` | Create simple text-based bar chart visualization of win distributions using Unicode block characters; scale bars to 40 characters max. |
| `wood_menu` | `dict wood_selection` | Present wood characteristics menu with options to select species, enter size, enter quality, or return to main menu; return updated wood_selection dictionary. |

## Conclusion and Discussion

* *Overall Experience with the Project* (2 pts.)
  * The most enjoyable aspect for me was integrating the AI. The first version of the program was very cumbersome and never actually did a good job of accurately accounting for the ways the factors really effect the performance of the wood. By the end of it, I was adjusting the parameters until I got handicaps that were pretty accurate, but it never felt like it was really capable of modeling things consistently across all species and competitors that weren't on the roster at the time. I spent a good amount of time looking at different models and settled on Qwen. Qwen is optimized for mathematical reasoning and has good contextual awareness. It can output text, but it is not optimized for creative responses or dialogue with the user. I really enjoyed researching the models and trying to troubleshoot how to integrate it in a way that actually met the project objectives. The other thing that I enjoyed was learning how to plan a coding project and build a workable scaffholding. My ability to outline what I wanted and think through how I wanted a program to work got drastically better between deliverable one and two. I feel like I have a pretty good system now for any project going forward. 

  * My least favorite part of the project for Deliverable One was actually building the handicap calculator. I had a friend of mine that is an engineer help me understand how each of those factors would change the behavior of the axe and how the variation in those factors would alter the predicted times. He developed a rough formula that became the framework for how I built the code. That portion became pretty enjoyable for Deliverable Two and I ended up really enjoying playing with the model to create a good output that was repeatable across competitors that weren't in the original dataset. Across both deliverables I hated the documentation. It was a nightmare even after trying to work on my commenting and project scaffholding. 

* *Original Vision versus Final Product*: How did the original vision of your project differ from the final? What led to those changes? (2 pts.)
  * There were three major changes I made to the program in addition to general polishing. First, I added the ability to calculate Underhand times. Originally I only had the ability to calcualte standing block times, but most woodchopping tournaments handicap the Standing Block, Underhand, the 3-Board Jigger. Second, I changed the way the handicaps were calculated from using a very cumbersome formula to using qwen2.5:7b to calcualte the handicaps. This produced way more consistent results and was WAY simpler. Realistically, the first formula worked great with everyone that was on the roster, but once I started adding other woodchoppers to the roster I started to run into less consistent and accurate handicaps. The third major change I made was to use qwen2.5:7b to validate those handicap marks with a Monte Carlo simulator. This was super fun and relatively easy to do. It provides the judge with an analysis of the handicap marks by simulating 250,000 potential matches betweent the competitors and showing the win percentage. In theory, with a properly handicapped system the win percentages should be consistent across competitors. The judge is given a readout of the results of the simulations and can visually see the how accurate the handicaps are, and see predicted times for each competitor. To me, this would be invaluable to a judge that was using this program to create as fair of a handicap as possible. 

  * In the future I would like to add the 3-Board Jigger to the events list, but because the event is so long and there is a lot of room for error it is difficult to find consistent times. There is also not nearly as much data to develop a program with as there is with the Standing Block and Underhand. This year when I watch the Sydney Royal Easter Show I plan to test the program against the actual results in real time. Another problem that I would like to progress. I would also like to develop a way to track a competitor's handicap over time. I showed this program to my coach [David Moses Jr.](https://www.youtube.com/watch?v=SPKl98qHF6I) and while generally happy with the results, he recommended that there I find a way to weight more recent times so that older professional competitors that had incredible careers in their 20s and 30s don't wind up with handicaps that unfairly punish them in their later years.

* *Lessons Learned and Experience*: What lessons did you learn from your project experience? You can discuss your critical thinking skills, programming skills, communication with Mentor, etc. (2 pts.)

  * This was a super fun project for me. In terms of programming skills, I feel like my ability to plan, scaffold code, and use comments to make a plan improved drstically. I feel like I have a pretty good work-flow (that I would love to continue to refine) that will allow me to have a good starting point for future projects. I also feel like my ability to problem solve and research for programming specifically really imporved. Between Stack overflow, youtube, and reddit I feel like I have a basic framework for how to learn new skills and implement new tools. I also really enjoyed the community aspect of this project. I had so much fun calling other woodchoppers and discussing the project, and the reactions I got from everyone that I demo'd the product to were overwhelmingly positive. There is a strong chance that Missoula Pro-Am and Mason County Western Qualifier will feature handicap races in the upcoming season!