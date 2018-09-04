## julia script to read the data/cytof-5-data
## and create the HDF5 and TXT files needed by Mocha.jl
## new files will be put in the same data folder
## Claudia April 2018
## to do (fixit):
## - for now, using only one response, can we use the whole vector of functional variables as response?
## - not including the column of dataset number, because the testing set will have a different value

using DataFrames, CSV

## -----------------------------
## Reading the data
## -----------------------------
datafolder = "C:/Users/xma72/Documents/Deep_Learning_project/nn-flow-cytometry-master/data/cytof-5-data/"
dat1 = "CyTOF54_Tube01_Day1_Unstim1_curated.fcs_eventnum_Ungated_Jnstim1_Day1_normalized_1_Unstim1_Singlets.fcs.txt"
dat2 = "CyTOF54_Tube02_Day1_Unstim2_curated.fcs_eventnum_Ungated_Unstim2_Day1_normalized_2_Unstim2_Singlets.fcs.txt"
dat3 = "CyTOF54_Tube03_Day2_Unstim3_curated.fcs_eventnum_Ungated_Unstim3_Day2_normalized_3_Unstim3_Singlets.fcs.txt"
dat4 = "CyTOF54_Tube04_Day2_Unstim4_curated.fcs_eventnum_Ungated_Unstim4_Day2_normalized_4_Unstim4_Singlets.fcs.txt"
dat5 = "CyTOF54_Tube05_Day3_Unstim5_curated.fcs_eventnum_Ungated_Unstim5_Day3_normalized_5_Unstim5_Singlets.fcs.txt"

df1 = CSV.read(string(datafolder,dat1), delim='\t')
df2 = CSV.read(string(datafolder,dat2), delim='\t')
df3 = CSV.read(string(datafolder,dat3), delim='\t')
df4 = CSV.read(string(datafolder,dat4), delim='\t')
df5 = CSV.read(string(datafolder,dat5), delim='\t')


## -------------------------------------
## Extracting predictors/responses
## -------------------------------------

## Surface markers: predictors, 15 markers in total
surface = ["191-DNA","193-DNA","115-CD45","139-CD45RA","142-CD19","144-CD11b","145-CD4","146-CD8","148-CD34",
           "147-CD20","158-CD33","160-CD123","167-CD38","170-CD90","110_114-CD3"]
## Functional markers: responses, 18 markers in total
functional=["141-pPLCgamma2","150-pSTAT5", "152-Ki67","154-pSHP2","151-pERK1/2","153-pMAPKAPK2",
            "156-pZAP70/Syk","159-pSTAT3","164-pSLP-76","165-pNFkB","166-IkBalpha","168-pH3","169-pP38",
            "171-pBtk/Itk","172-pS6","174-pSrcFK","176-pCREB","175-pCrkL"]
# try predicting all functional markers together
# functional=functional[1:2] ##fixit: using only one response now, but need to take two to convert to Array{Float32,2}

## Creating predictor matrix, using dat1,dat2,dat3,dat4 as training samples, and dat5 as testing sample
df1pred = df1[Symbol.(surface)]
df2pred = df2[Symbol.(surface)]
df3pred = df3[Symbol.(surface)]
df4pred = df4[Symbol.(surface)]
df5pred = df5[Symbol.(surface)]

## This is to include the dataset number (see notebook-log.md)
## Not doing it now (fixit).
if(false)
    df1pred[:dataset] = fill(1,size(df1pred,1))
    df2pred[:dataset] = fill(2,size(df2pred,1))
    df3pred[:dataset] = fill(3,size(df3pred,1))
    df4pred[:dataset] = fill(4,size(df4pred,1))
    df5pred[:dataset] = fill(5,size(df5pred,1))
end

dfpredTrain = vcat(df1pred,df2pred,df3pred,df4pred)
dfpredTest = df5pred

## Creating response vector
df1resp = df1[Symbol.(functional)]
df2resp = df2[Symbol.(functional)]
df3resp = df3[Symbol.(functional)]
df4resp = df4[Symbol.(functional)]
df5resp = df5[Symbol.(functional)]

dfrespTrain = vcat(df1resp,df2resp,df3resp,df4resp)
dfrespTest = df5resp


## -------------------------------------
## Creating input txt files
## -------------------------------------
resultsfolder = "C:/Users/xma72/Documents/Deep_Learning_project/nn-flow-cytometry-master/results/"

CSV.write(string(resultsfolder,"cytof-5-data-train-pred.txt"), dfpredTrain)
CSV.write(string(resultsfolder,"cytof-5-data-train-resp.txt"), dfrespTrain)
CSV.write(string(resultsfolder,"cytof-5-data-test-pred.txt"), dfpredTest)
CSV.write(string(resultsfolder,"cytof-5-data-test-resp.txt"), dfrespTest)


## -------------------------------------
## Creating input files for Mocha.jl
## -------------------------------------
# using HDF5,JLD
# include("C:/Users/xma72/Documents/Deep_Learning_project/nn-flow-cytometry-master/script/functions.jl")
# resultsfolder = "C:/Users/xma72/Documents/Deep_Learning_project/nn-flow-cytometry-master/results/"
#
# traindata = string(resultsfolder,"cytof-5-data-train")
# writeMochaInput(dfpredTrain,dfrespTrain,traindata)
#
# testdata = string(resultsfolder,"cytof-5-data-test")
# writeMochaInput(dfpredTest,dfrespTest,testdata)
