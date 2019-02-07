# To do
- put in 2 columns (6pages) format for http://sceweb.sce.uhcl.edu/bicob19/paper_submission.html ?
- put in BMC format?


# 0. Exploratory analysis
- Using `scripts/exploratory.r`, which will read the raw data from `data/cytof-5-data`, and plot some exploratory things.
- Output 
    - `results/hist-all.pdf`
    - `results/hist-all-transformed.pdf`
    - `results/scattermatrix1.pdf`

# 1. Reading the data and creating input files for NN
- Using `scripts/create-input.jl` script, which will read data from `data/cytof-5-data`:
    CyTOF54_Tube01_Day1_Unstim1_curated.fcs_eventnum_Ungated_Jnstim1_Day1_normalized_1_Unstim1_Singlets.fcs.txt
    CyTOF54_Tube02_Day1_Unstim2_curated.fcs_eventnum_Ungated_Unstim2_Day1_normalized_2_Unstim2_Singlets.fcs.txt
    CyTOF54_Tube03_Day2_Unstim3_curated.fcs_eventnum_Ungated_Unstim3_Day2_normalized_3_Unstim3_Singlets.fcs.txt
    CyTOF54_Tube04_Day2_Unstim4_curated.fcs_eventnum_Ungated_Unstim4_Day2_normalized_4_Unstim4_Singlets.fcs.txt
    CyTOF54_Tube05_Day3_Unstim5_curated.fcs_eventnum_Ungated_Unstim5_Day3_normalized_5_Unstim5_Singlets.fcs.txt
    - and will create:
        cytof-5-data-test.hdf5
        cytof-5-data-test.txt
        cytof-5-data-train.hdf5
        cytof-5-data-train.txt
    - For now, using only one response at a time, we want to change this for the whole vector
    - Question: Do we want to include the extra column of "dataset" even though it will be completely different (equal to 5) for the testing data? For now, I will not include this column.

# 2. Fitting neural network
- Using `script/fit-nn.jl`. Considerations:
    - response is a real number, so we can have big errors by using one node at the end; but creating categories is not good idea either
    - We will start with the same network architecture as in Liu 2017: depth-4 feed-forward neural network with three softplus hidden layers and a softmax output layer. The hidden layer sizes are set to 12,6,3.

# 3. Comparing analyses

Scripts `analysis-comb.jl` for original submission, and `analysis-comb-cv.jl` for revision.
Results are saved in this file and then copy-pasted in `plots.r` for the final plots.