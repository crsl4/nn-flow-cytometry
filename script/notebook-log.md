# To do
- Plot the relationships between surface and functional markers, to see if they follow a linear trend
- Plot the histograms of the markers to see if we need to transform. In the other deep learning paper for cytometry data, they did transform their variables to achieve normality. I do not have much experience with neural networks to tell how much this will affect the performance, but let's look at the histograms and decide
- Run another method (linear regression/splines) to have a baseline predication performance, to compare to our neural network
- Try different options in the neural network

- read Li 2017 paper


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