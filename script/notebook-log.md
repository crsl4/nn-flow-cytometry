# To do
- check data; write NN script in julia
- read Li 2017 paper

# 1. Reading the data and creating input files for NN
- Using `scripts/create-input.jl` script
    - For now, using only one response at a time, we want to change this for the whole vector
    - Question: Do we want to include the extra column of "dataset" even though it will be completely different (equal to 5) for the testing data? For now, I will not include this column.