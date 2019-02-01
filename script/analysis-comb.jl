## section 1: import and separate data into training, validation 
##            and testing samples
using DataFrames, CSV

datafolder = "../cytof 5 data/"
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

## Surface markers: predictors, 15 markers in total
surface = ["191-DNA","193-DNA","115-CD45","139-CD45RA","142-CD19","144-CD11b","145-CD4","146-CD8","148-CD34",
           "147-CD20","158-CD33","160-CD123","167-CD38","170-CD90","110_114-CD3"]
## Functional markers: responses, 18 markers in total
functional=["141-pPLCgamma2","150-pSTAT5", "152-Ki67","154-pSHP2","151-pERK1/2","153-pMAPKAPK2",
            "156-pZAP70/Syk","159-pSTAT3","164-pSLP-76","165-pNFkB","166-IkBalpha","168-pH3","169-pP38",
            "171-pBtk/Itk","172-pS6","174-pSrcFK","176-pCREB","175-pCrkL"]

dfpred = vcat(df1[Symbol.(surface)],df2[Symbol.(surface)],df3[Symbol.(surface)],df4[Symbol.(surface)],df5[Symbol.(surface)])
dfresp = vcat(df1[Symbol.(functional)],df2[Symbol.(functional)],df3[Symbol.(functional)],df4[Symbol.(functional)],df5[Symbol.(functional)])

## randomize 1,223,228 rows into three sets: train (750,000 rows), valid (250,000 rows), test (223,228 rows)
srand(5312)
rand_seq = randperm(1223228)
dfpredTrain = convert(Array{Float64,2}, dfpred[rand_seq[1:750000],:])
dfrespTrain = convert(Array{Float64,2}, dfresp[rand_seq[1:750000],:])
dfpredValid = convert(Array{Float64,2}, dfpred[rand_seq[750001:1000000],:])
dfrespValid = convert(Array{Float64,2}, dfresp[rand_seq[750001:1000000],:])
dfpredTest = convert(Array{Float64,2}, dfpred[rand_seq[1000001:1223228],:])
dfrespTest = convert(Array{Float64,2}, dfresp[rand_seq[1000001:1223228],:])

###########################################################################
## section 2: regularized linear regression (RLR), 
##            random forest regression (RFR), 
##            and support vector regression (SVR)
###########################################################################
### fit the linear model, no regularity
using ScikitLearn
@sk_import linear_model: LinearRegression
tic()
lr = fit!(LinearRegression(),dfpredTrain,dfrespTrain)
toc()  ## 1.98s
lr_vlpred = predict(lr,dfpredValid)
diff = mapslices(Base.norm, lr_vlpred - dfrespValid,[2])
lr_vec_loss = sum(diff.^2)/(2*size(diff)[1])
# no regularity, mse = 6.7036

### Lasso
@sk_import linear_model: Lasso
pel = [0.1,1,10,100,1000]
vec_loss=zeros(5)
for i=1:5
    rlr = fit!(Lasso(alpha=pel[i]),dfpredTrain,dfrespTrain)
    rlr_pred = predict(rlr,dfpredValid)
    diff = mapslices(Base.norm, rlr_pred - dfrespValid,[2])
    vec_loss[i] = sum(diff.^2)/(2*size(diff)[1])
    println("iter:",i,", loss:",vec_loss[i])
end
vec_loss  ## 7.0253, 9.0376, 9.1934, 9.1934, 9.1934

## choose linear regression with no regulation
@sk_import linear_model: LinearRegression
lr = fit!(LinearRegression(),dfpredTrain,dfrespTrain)
lr_tepred = predict(lr,dfpredTest)
diff = mapslices(Base.norm, lr_tepred - dfrespTest,[2])
lr_vec_loss = sum(diff.^2)/(2*size(diff)[1])  ## 6.7148
# MSE for individual response
lr_pt_loss=zeros(18)
for i=1:18
    lr_pt_loss[i] = sum((lr_tepred[:,i]-dfrespTest[:,i]).^2)/(2*size(lr_tepred)[1])
end
lr_pt_loss  ## 0.2308, 0.4301, 0.5954, 0.2353, 0.2490, 0.2427
            ## 0.2414, 0.3983, 0.2016, 0.3520, 0.2792, 0.4322
            ## 0.3795, 0.9793, 0.4539, 0.4587, 0.2931, 0.2624

###########################################################################
## Decision Tree regression
@sk_import tree: DecisionTreeRegressor
tic()
dtr = fit!(DecisionTreeRegressor(criterion="mse"),dfpredTrain,dfrespTrain)
toc()  ## 63.36s
dtr_vlpred = predict(dtr,dfpredValid)
diff = mapslices(Base.norm,dtr_vlpred-dfrespValid,[2])
vec_loss = sum(diff.^2)/(2*size(diff)[1])
## default setting: 11.5040
## "friedman_mse": 11.9040
## choose mse criterion for testing data
dtr_tepred = predict(dtr,dfpredTest)
diff = mapslices(Base.norm,dtr_tepred-dfrespTest,[2])
dtr_vec_loss = sum(diff.^2)/(2*size(diff)[1]) ## 11.5280
# MSE for individual response
dtr_pt_loss=zeros(18)
for i=1:18
    dtr_pt_loss[i] = sum((dtr_tepred[:,i]-dfrespTest[:,i]).^2)/(2*size(dtr_tepred)[1])
end
dtr_pt_loss  ## 0.4207, 0.7510, 0.8740, 0.4556, 0.4644, 0.4460,
             ## 0.4363, 0.7468, 0.3831, 0.5550, 0.4696, 0.6693,
             ## 0.6602, 1.5796, 0.7696, 0.8004, 0.5239, 0.5227

###########################################################################
## random forest regression
@sk_import ensemble: RandomForestRegressor
tic()
rfr = fit!(RandomForestRegressor(n_estimators=20),dfpredTrain,dfrespTrain)
toc()  ## 826.83s
rfr_vlpred = predict(rfr,dfpredValid)
diff = mapslices(Base.norm,rfr_vlpred-dfrespValid,[2])
vec_loss = sum(diff.^2)/(2*size(diff)[1])
## 10 trees: 6.2226
## 20 trees: 5.9319
## use 20 trees for testing data for now due to limited time
rfr_tepred = predict(rfr,dfpredTest)
diff = mapslices(Base.norm,rfr_tepred-dfrespTest,[2])
rfr_vec_loss = sum(diff.^2)/(2*size(diff)[1]) ## 5.9503
# MSE for individual response
rfr_pt_loss=zeros(18)
for i=1:18
    rfr_pt_loss[i] = sum((rfr_tepred[:,i]-dfrespTest[:,i]).^2)/(2*size(rfr_tepred)[1])
end
rfr_pt_loss  ## 0.2183, 0.3881, 0.4480, 0.2374, 0.2429, 0.2321,
             ## 0.2278, 0.3863, 0.2005, 0.2869, 0.2441, 0.3433,
             ## 0.3425, 0.8015, 0.3982, 0.4095, 0.2725, 0.2703

###########################################################################
## support vector regression, can only train scalar response
## toooooo slow, more than three hours unfinished
# @sk_import svm: SVR
# svr = fit!(SVR(kernel="rbf",gamma=0.1),dfpredTrain,dfrespTrain[:,1])
# svr_pred = predict(svr,dfpredValid)
# pt_loss = sum((svr_pred-dfrespValid[:,1]).^2)/(2*size(svr_pred)[1])
## 'rbf',gamma=[0.1,auto]:
## 'poly',degree=[3,2,4]:
## 'sigmoid',gamma=[auto,0.1]: 
## for testing data


###########################################################################
## section 3: fit Neural Network
###########################################################################
using Mocha
cd("../results/")
expdir = "snapshots/"  ## WARNING: you should not have snapshots folder already in wd
rootname = "cytof-5"

trX = transpose(dfpredTrain)
trY = transpose(dfrespTrain)
vlX = transpose(dfpredValid)
vlY = transpose(dfrespValid)
teX =transpose(dfpredTest)
teY = transpose(dfrespTest)

# Configure and build
backend = CPUBackend()
init(backend)

# network structure
data_layer  = MemoryDataLayer(name=rootname, data=Array[trX, trY], batch_size=100, shuffle=true)

fc1_layer  = InnerProductLayer(name="ip1", output_dim=90,
neuron=Neurons.Tanh(), weight_init = GaussianInitializer(std=.01),
bottoms=[:data], tops=[:ip1])

fc2_layer  = InnerProductLayer(name="ip2", output_dim=90,
neuron=Neurons.Tanh(), weight_init = GaussianInitializer(std=.01),
bottoms=[:ip1], tops=[:ip2])

fc3_layer  = InnerProductLayer(name="ip3", output_dim=45,
neuron=Neurons.Tanh(), weight_init = GaussianInitializer(std=.01),
bottoms=[:ip2], tops=[:ip3])

fc4_layer  = InnerProductLayer(name="ip4", output_dim=45,
neuron=Neurons.Tanh(), weight_init = GaussianInitializer(std=.01),
bottoms=[:ip3], tops=[:ip4])

fc5_layer  = InnerProductLayer(name="ip5", output_dim=18,bottoms=[:ip4], tops=[:ip5])

loss_layer = SquareLossLayer(name="loss", bottoms=[:ip5,:label])

common_layers = [fc1_layer, fc2_layer, fc3_layer, fc4_layer, fc5_layer]
net = Net(rootname, backend, [data_layer, common_layers..., loss_layer])

# Setting up the solver
method = Adam()
params = make_solver_parameters(method, max_iter=50000, regu_coef=0.0001,
    mom_policy=MomPolicy.Fixed(0.8),
    lr_policy=LRPolicy.Inv(0.01, 0.0001, 0.75),
    load_from=expdir)
solver = Solver(method, params)

setup_coffee_lounge(solver, save_into="$expdir/statistics.jld", every_n_iter=1000)
add_coffee_break(solver, TrainingSummary(), every_n_iter=10000)
add_coffee_break(solver, Snapshot(expdir), every_n_iter=50000)


# add_coffee_break(solver, ValidationPerformance(test_net), every_n_iter=10000)
# add_coffee_break(solver,ValidationPerformance(final_net),every_n_iter=50000)

@time solve(solver, net)  ## 181.82s

open(string(rootname,".dot"), "w") do out net2dot(out, net) end

# Evaluation network. Run against the validation set
# Loss function is set to be squared loss as we have continuous outcomes (Xin)
data_layer_test = MemoryDataLayer(name=string(rootname,"-valid"), data=Array[vlX,vlY],batch_size=1)
# acc_layer = SquareLossLayer(name=string(rootname,"-accuracy"), bottoms=[:ip5, :label])
pred_layer = HDF5OutputLayer(filename=string(rootname,"-vlpred.h5"),bottoms=[:ip5])
test_net = Net(string(rootname,"-valid"), backend, [data_layer_test, common_layers..., pred_layer])
load_snapshot(test_net,"snapshots/snapshot-050000.jld")
init(test_net)
forward_epoch(test_net)

## on testing data
test_data = MemoryDataLayer(name=string(rootname,"-test"),data=Array[teX,teY],batch_size=1)
pred_layer = HDF5OutputLayer(filename=string(rootname,"-tepred.h5"),bottoms=[:ip5])
final_net = Net(string(rootname,"-test"),backend,[test_data, common_layers..., pred_layer])
load_snapshot(final_net,"snapshots/snapshot-050000.jld")
init(final_net)
forward_epoch(final_net)


destroy(net)
destroy(test_net)
destroy(final_net)
shutdown(backend)

## load prediction for MSE calculation
using HDF5
nn_vlpred = h5open("cytof-5-vlpred.h5", "r") do file
    read(file, "ip5")
end
nn_tepred = h5open("cytof-5-tepred.h5", "r") do file
    read(file, "ip5")
end
## validation mse 5.6334
diff = mapslices(Base.norm,nn_vlpred-vlY,[1])
vec_loss = sum(diff.^2)/(2*size(diff)[2])
## testing mse
diff = mapslices(Base.norm,nn_tepred-teY,[1])
nn_vec_loss = sum(diff.^2)/(2*size(diff)[2]) ## 5.6447
# MSE for individual response
nn_pt_loss=zeros(18)
for i=1:18
    nn_pt_loss[i] = sum((nn_tepred[i,:]-teY[i,:]).^2)/(2*size(nn_tepred)[2])
end
nn_pt_loss  ## 0.2067, 0.3586, 0.4266, 0.2254, 0.2331, 0.2212,
            ## 0.2172, 0.3665, 0.1904, 0.2735, 0.2315, 0.3271,
            ## 0.3260, 0.7565, 0.3790, 0.3897, 0.2593, 0.2564



## 'Tanh' best; train - regu_coef, mom, lr (3)
## 0.005, 0.9, (0.01, 0.0001, 0.75) [90,45,45]: 5.8803 /te: 5.8940
## 0.005, 0.8, (0.01, 0.0001, 0.75): 5.8599 /te: 5.8706
## 0.005, 0.7, (same)              : 5.8783 /te: 5.8886
## 0.01,  0.8, (same)              : 6.0398 /te: 6.0502
## 0.005, 0.8, (same) using Adam   : 5.8393 /te: 5.8508
## 0.001, 0.8, (same) Adam         : 5.7208 /te: 5.7320
## 0.0001, 0.8, (same) Adam        : 5.6655 /te: 5.6794 *
## 0.0001, 0.8, (0.005,same) Adam  : 5.6836 /te: 5.6966
## 0.0001, 0.8, (0.05,same) Adam   : 5.7394 /te: 5.7544
## as * enlarge layers[180,90,45]  : 5.6654 /te: 5.6735
## * [360,90,45]                   : 5.6805 /te: 5.6919
## * one more layer[90,90,45,45]   : 5.6433 /te: 5.6568 ** (results rerun)

#########################################################################
## create dataframe file for validation / testing prediction values wit true values
nn_vlpred = transpose(nn_vlpred)
nn_tepred = transpose(nn_tepred)

for k=1:18
    Valid = DataFrame(True=dfrespValid[:,k],LR=lr_vlpred[:,k],DTR=dtr_vlpred[:,k],RFR=rfr_vlpred[:,k],NN=nn_vlpred[:,k])
    Test = DataFrame(True=dfrespTest[:,k],LR=lr_tepred[:,k],DTR=dtr_tepred[:,k],RFR=rfr_tepred[:,k],NN=nn_tepred[:,k])
    CSV.write(string("Valid_",k,".txt"),Valid,delim='\t')
    CSV.write(string("Test_",k,".txt"),Test,delim='\t')
end