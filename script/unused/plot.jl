using CSV, DataFrames
cd("../results/")
Valid1 = CSV.read("Valid_1.txt",delim="\t")
short = Valid1[1:100,:]

using Gadfly
plot(short,y=[:True,:LR])