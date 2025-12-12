# New Features for V3
* RECTIFY ALL MODELS TO RUN QWEN 37B
## Manual Formula

* Re-Add Manual Handicap claculations for comparison (from V1)

## Ai black box explanation and Validation
* Add new portion of prompt to have model justify decisions
* Add part of prompt to develop manual formula and validate existing manual formula
* Run manual model in tandem with AI model


## Heat Generator

* currently the heat selection works by having the judge select competitors from a list of competitors on a roster and manually assigning them to a heat. This is not realistic.

* I want a system where I select a number of competitors for an event and after calcualting he handicaps, it organzes them into heats. Typically in a woodchopping competition, the top 1 or two from every heat advance to a semi final or final. For instance, if I have 8 available stands, then I can run up to 32 choppers (4 heats of 8), take the top 2, and then put them into a final heat.

* This would obviously be restricted by the number of stands (VERY IMPORTANT!). Before I select competitors for the event, I would want the system to tell me the maximal amount of possible competitors if I went from initial heats --> final heats, and an option to go from inital heats --> semi final heat --> final heats. I want the system to also tell me how many total blocks I would need for each event (1 block per competitor per heat) so that I can plan accordingly.

* To ensure fairness, after calculating the handicap marks, I want to sort the competitors with an even distribution of skill. For example, I want one or two front markers, one or two back markers, and an even distribition of mid-markers in each heat so that there is an even chance of even distribution of skill levels in the semi-finals/finals. We want to avoid a situtation where all the frontmarkers and backmarkers are in the same heat.

* There is currently an option to update the excel sheet with heat times. I want to keep that and I would also want the option to select the winning competitors for each heat for the system to store to formulate the semi-final heats and final heats.

* The core functionality can remain the same, but I would like to switch up the menu options and reorganize the feautures in to something that makes more sense to reflect all of this. 

    * The workflow for a judge should look something like this: Select the wood/Characteristics (Keep wood menu the same) --> Input the number of available stands to chop on --> system displays number of competitors (and needed blocks!) that can be entered for running either initial heats into finals or initial heats into semi finals into final heats --> Judge selects all competitors from master roster that will be entered into the event --> Judge runs handicap calculator/montecarlo like normal --> Judge veiws handicaps and fairness analysis --> Judge has system equitably distribute into heats for even --> after every heat judge can can enter competitor name that will advance to either the semi finals or finals

* I want there to still be the option to add/remove/append competitors from heats and the master roster on excel. Maybe make a separate menu for "peronnelle management?"




# New Features for V4

## MAJOR Fix to 'Record Heat Results & Select Advancers' (Menu Option 7)

* After selecting a heat to record, there is a major glaring oversight to how winners are selected to advance. While it is important to record their times (especially for a Championship style race!) in a handicap race it is MORE important to record the ORDER in which they finished. Remeber, the idea of a handicap is to create a system where the start times are offset. So even though their recorded time would be from when their handicap mark is called to the time the block is severed, the judges are really looking at who severs their blcok first with the delay added. I would like to keep the functionality of adding the times, but I need a very quick and easy way to add the finish order of the heat to select who stays and who moves on.

## Explanation of how the system works

* Because judges may not be be familiar with advanced statitstics/modeling, let alon how ML/LLMs work I want an option that explains in pain English to a lay judge how the manual prediction works, how the LLM prediction works, and how the ML prediction works. This needs to build trust so plainly state the advantages and disadvantages of each system.

* Similar to how the Monte Carlo System is functionally accessible, at the end of this explanation offer a small sub menu:
    * menu Option 1 will allow you to access a "Statistical Glossary" That explains what EVERY SINGLE statistcal term used in the program means (Think especially about your abbreviations like "3IQR") AND includes an explanation as to why it is relevant to woodchopping and handicap calculations. Be thorough be clear in your explanations. Also cite where in the process the user will see this method, term, or acronym/abbreviation.

    * menu option 2 will be batshit crazy. I need this to be a failsafe if a judge or competitor is questioning the integrity of the system or is more technically inclined to understand how this program works. For menu option 2, I want a no-bullshit explanation of EXACTLY how this program manually tabulates the handicaps, tabulates them via LLM, and tabulates them via ML. This needs to hit the PERFECT balance between accurate technical explanations and accessibility to someone who does not have time to read the entire project documentation. I know there is an IPO diagram and functions library in the documentation, but I just want a clear and concise explanation only of how the handicaps themselves are calculated. maybe include some awesome ASCII art featuring a wizard that says "Warning: Turbo Nerd Territory! Proceed with Caution!" or something like that. I am leaving it up to your artistic interpretation but make this menu option fun, lightherated, and feauture a badass wizard.

* I want these explanations/submenus accessible in 2 separate points in the program. The most visible place should be a #15 option on the main menu. The second place as I have stated before is after the handicaps analysis has been presented (after all of the other options within Main Menu option 5 have been exhausted). I am giving you creative license to name the menus/submenus to best meet the needs of my intent (make it accessible to judges and competitors that have questions on the integrity of the system)