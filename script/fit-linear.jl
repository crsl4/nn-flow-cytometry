## fit linear regression model to compare performance
using CSV, MultivariateStats
wd = "C:/Users/xma72/Documents/Deep_Learning_project/nn-flow-cytometry-master/results/"
cd(wd)
trainX = CSV.read(string(wd,"cytof-5-data-train-pred.txt"))
trainX = convert(Array{Float32,2}, trainX)
trainY = CSV.read(string(wd,"cytof-5-data-train-resp.txt"))
trainY = convert(Array{Float32,2}, trainY)
testX = CSV.read(string(wd,"cytof-5-data-test-pred.txt"))
testX = convert(Array{Float32,2}, testX)
testY = CSV.read(string(wd,"cytof-5-data-test-resp.txt"))
testY = convert(Array{Float32,2}, testY)

## fit the linear model
sol = MultivariateStats.llsq(trainX, trainY)
## extract coefficients
a, b = sol[1:end-1,:], sol[end,:]
b = reshape(b,(1,18))
## prediction for testing data and calculate MSE
predY = testX*a .+ b
norm = mapslices(norm,predY-testY,2)
squared_loss = sum(norm.^2)/(2*size(predY)[1])  ## 6.41401