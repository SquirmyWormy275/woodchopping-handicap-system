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




## 