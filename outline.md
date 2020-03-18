
Results from abstract analysis:

    - analyzed first 200ms of target / reference responses
    
    - grouped all targets / references for active / passive
        - dprime higher during behavior. Improvement correlated with mean performance for the overall session
        - Improvement all explained by first order effects, not second order (but this is weird, bc lumping diff refs together)
    
    - noise correlations calculated on target only
        - significantly reduced for all sites between active and passive
        - magnitude of change correlated with coarse behavioral performance
    
    - euclidean distance analysis in PC space
        - project single trials into 3-D pc space defined by the mean responses over all stim / states
        - arousal enhances ref/ref comparison but active/passive transition does not
        - arousal AND active/passive transition enhance tar/ref comparison


Outline for poster:

    Background 
        
        - both task engagement and arousal shown to modulate activity in similar ways (distinction between sustained / selective attention?)
        - some work dissociating at the level of single cells (Daniela paper, Vinck?)
        - No work dissociating at the population level
        - task engagement has been shown to enhance target representation (David, Fritz, Bagur etc.)

    Methods

        - Schematic of behavioral task and recording set up
        - Example pupil trace / task status / pupil neuron / task neuron
            (steal a figure from Daniela's paper)
        - possibly also figure of trial evoked pupil by Hit/Miss/FA/passive?

    Results

        1) Replication of previous work (now at population level)
            - dprime improvement during active relative to passive listening
            - magnitude of improvement correlated with behavioral performance on that block
            - Analyze miss trials? e.g. could project single hit/miss trials on decoding axis and
                look at mean separation on hit / miss trials. If hit more separated (and miss looks like passive)
                then more evidence that this dprime enhancement is relavent. 
            - Plot example time course of projection on decoding axis for active / passive / miss trials.
        
        2) Task-engagement selectively enhances category discrimination
            - split passive into large / small pupil. Show that changes in arousal enhance both 
                tar/ref discrimination and ref/ref discrimination while task engagement only enhances
                category discrimination.
            - think about this analysis (mean pairwise euclidean distance is simply the distance between
                means, I think. So, this doesn't say anything about single trial variability)

        3) Noise correlation significantly reduced during active relative to passive (all and passive big)
            - effect is also strongly correlated with behavior performance
            - same is not true for passive big vs. passive small pupil. 
            - Do these changes affect decoding?
                - To answer this, I think I have to look at first PC of noise change in targets, then 
                  compare that with the decoding axis for each reference (and for grouped refs)? 
                  i.e. is the noise shrinking along the relevant axes, or a null axis?
                - Doing the simulation approach (from the NAT paper) is tricky because what do 
                  correlations mean if you're embedding signal correlations by grouping targets / refs?

    To Explore / TO-DO
        
        - what is the relationship between noise, evoked activty, and decoding axis for the different 
            comparisons analyzed above (how do they map onto the heatmap I made for NAT)
        - redo discrimination analyses for poster (save stats relevant for ^)
            - perform for different time windows, not just the first 200ms
        - rather than mean pairwise euclidean distance, just do mean distance. Is result the same? Should be...
        - Per trial decoding (project onto decoding axis, look at separation over time)
            - try fixed axis, axis that shifts over trial, how similar are they over the trial etc. lots of options
        - state regression models - remove pupil dependent variability, how does dprime change? remove behavior, etc.
        - decoding analysis with LDA?