## analysis for the revision, running 10-fold cross validation for all the methods
using DataFrames, CSV, ScikitLearn
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
## randomize 1,223,228 rows into two sets: train (1,000,000 rows), test (223,228 rows)
srand(5312)
rand_seq = randperm(1223228)
dfpredTrain = convert(Array{Float64,2}, dfpred[rand_seq[1:1000000],:])
dfrespTrain = convert(Array{Float64,2}, dfresp[rand_seq[1:1000000],:])
dfpredTest = convert(Array{Float64,2}, dfpred[rand_seq[1000001:1223228],:])
dfrespTest = convert(Array{Float64,2}, dfresp[rand_seq[1000001:1223228],:])

## cross-validation using the train set to pick the models
##################################################################################
# linear model, no penalty
@sk_import linear_model: LinearRegression
lr_vec_loss=0
for k=0:9
    lr = fit!(LinearRegression(),dfpredTrain[setdiff(1:1000000,(k*100000+1):(k*100000+100000)),:],dfrespTrain[setdiff(1:1000000,(k*100000+1):(k*100000+100000)),:])
    lr_vlpred = predict(lr,dfpredTrain[(k*100000+1):(k*100000+100000),:])
    diff = mapslices(Base.norm,lr_vlpred-dfrespTrain[(k*100000+1):(k*100000+100000),:],[2])
    lr_vec_loss = lr_vec_loss + 0.1*sum(diff.^2)/(2*size(diff)[1])
end
lr_vec_loss
# no regularity, avg_mse = 6.706441

# Lasso
@sk_import linear_model: Lasso
pel = [0.001,0.01,0.1,1,10]
la_vec_loss=0
for k=0:9
    rlr = fit!(Lasso(alpha=0.0001),dfpredTrain[setdiff(1:1000000,(k*100000+1):(k*100000+100000)),:],dfrespTrain[setdiff(1:1000000,(k*100000+1):(k*100000+100000)),:])
    rlr_pred = predict(rlr,dfpredTrain[(k*100000+1):(k*100000+100000),:])
    diff = mapslices(Base.norm, rlr_pred - dfrespTrain[(k*100000+1):(k*100000+100000),:],[2])
    la_vec_loss = la_vec_loss + 0.1*sum(diff.^2)/(2*size(diff)[1])
end
la_vec_loss
#penalty (avg_mse): 0.0001 (6.706442); 0.001 (6.7066); 0.01 (6.7172); 0.1 (7.0262); 1 (9.0398)

# choose linear model with no penalty
tic()
lr = fit!(LinearRegression(),dfpredTrain,dfrespTrain)
toc() # 1.905s
lr_tepred = predict(lr,dfpredTest)
diff = mapslices(Base.norm, lr_tepred - dfrespTest,[2])
lr_vec_loss = sum(diff.^2)/(2*size(diff)[1])  ## 6.7147
# MSE for individual response
lr_pt_loss=zeros(18)
for i=1:18
    lr_pt_loss[i] = sum((lr_tepred[:,i]-dfrespTest[:,i]).^2)/(2*size(lr_tepred)[1])
end
lr_pt_loss # 0.2308 0.4301 0.5953 0.2353 0.2490 0.2427 0.2414 0.3983 0.2016 0.3520 0.2792 0.4321 0.3795 0.9793 0.4539 0.4586 0.2931 0.2624

###########################################################
# decision tree
@sk_import tree: DecisionTreeRegressor
dt_vec_loss=0
for k=0:9
    dtr = fit!(DecisionTreeRegressor(criterion="friedman_mse"),dfpredTrain[setdiff(1:1000000,(k*100000+1):(k*100000+100000)),:],dfrespTrain[setdiff(1:1000000,(k*100000+1):(k*100000+100000)),:])
    dtr_vlpred = predict(dtr,dfpredTrain[(k*100000+1):(k*100000+100000),:])
    diff = mapslices(Base.norm,dtr_vlpred-dfrespTrain[(k*100000+1):(k*100000+100000),:],[2])
    dt_vec_loss = dt_vec_loss + 0.1*sum(diff.^2)/(2*size(diff)[1])
end
dt_vec_loss
## default setting: 11.4966
## "friedman_mse": 11.8922
## choose mse criterion for testing data
tic()
dtr = fit!(DecisionTreeRegressor(criterion="mse"),dfpredTrain,dfrespTrain)
toc() # 99.951s
dtr_tepred = predict(dtr,dfpredTest)
diff = mapslices(Base.norm,dtr_tepred-dfrespTest,[2])
sum(diff.^2)/(2*size(diff)[1]) ## 11.4857
# MSE for individual response
dtr_pt_loss=zeros(18)
for i=1:18
    dtr_pt_loss[i] = sum((dtr_tepred[:,i]-dfrespTest[:,i]).^2)/(2*size(dtr_tepred)[1])
end
dtr_pt_loss # 0.4199 0.7462 0.8713 0.4555 0.4659 0.4451 0.4365 0.7423 0.3821 0.5529 0.4703 0.6672 0.6621 1.5584 0.7700 0.7952 0.5231 0.5219



############################################################
# random forest
@sk_import ensemble: RandomForestRegressor
rf_vec_loss1=0
for k=0:9
    rfr = fit!(RandomForestRegressor(n_estimators=10),dfpredTrain[setdiff(1:1000000,(k*100000+1):(k*100000+100000)),:],dfrespTrain[setdiff(1:1000000,(k*100000+1):(k*100000+100000)),:])
    rfr_vlpred = predict(rfr,dfpredTrain[(k*100000+1):(k*100000+100000),:])
    diff = mapslices(Base.norm,rfr_vlpred-dfrespTrain[(k*100000+1):(k*100000+100000),:],[2])
    rf_vec_loss1 = rf_vec_loss1 + 0.1*sum(diff.^2)/(2*size(diff)[1])
end
rf_vec_loss1

rf_vec_loss2=0
for k=0:9
    rfr = fit!(RandomForestRegressor(n_estimators=20),dfpredTrain[setdiff(1:1000000,(k*100000+1):(k*100000+100000)),:],dfrespTrain[setdiff(1:1000000,(k*100000+1):(k*100000+100000)),:])
    rfr_vlpred = predict(rfr,dfpredTrain[(k*100000+1):(k*100000+100000),:])
    diff = mapslices(Base.norm,rfr_vlpred-dfrespTrain[(k*100000+1):(k*100000+100000),:],[2])
    rf_vec_loss2 = rf_vec_loss2 + 0.1*sum(diff.^2)/(2*size(diff)[1])
end
rf_vec_loss2

## 10 trees: 6.2289
## 20 trees: 5.9442
## use 20 trees for testing data for now due to limited time
tic()
rfr = fit!(RandomForestRegressor(n_estimators=20),dfpredTrain,dfrespTrain)
toc() ## 1093.24s
rfr_tepred = predict(rfr,dfpredTest)
diff = mapslices(Base.norm,rfr_tepred-dfrespTest,[2])
sum(diff.^2)/(2*size(diff)[1]) ## 5.9399
# MSE for individual response
rfr_pt_loss=zeros(18)
for i=1:18
    rfr_pt_loss[i] = sum((rfr_tepred[:,i]-dfrespTest[:,i]).^2)/(2*size(rfr_tepred)[1])
end
rfr_pt_loss ## 0.2180 0.3867 0.4463 0.2373 0.2429 0.2315 0.2275 0.3861 0.2005 0.2867 0.2436 0.3429 0.3419 0.7989 0.3979 0.4096 0.2720 0.2697




##############################################################
# neural network
using Mocha, HDF5
cd("../results2/")
expdir = "snapshots/"  ## WARNING: you should not have snapshots folder already in wd
rootname = "cytof-5"

sum_loss=0
for k = 0:9
    trX = transpose(dfpredTrain[setdiff(1:1000000,(k*100000+1):(k*100000+100000)),:])
    trY = transpose(dfrespTrain[setdiff(1:1000000,(k*100000+1):(k*100000+100000)),:])
    vlX = transpose(dfpredTrain[(k*100000+1):(k*100000+100000),:])
    vlY = transpose(dfrespTrain[(k*100000+1):(k*100000+100000),:])
    
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
    
    setup_coffee_lounge(solver, save_into="$expdir/statistics.jld", every_n_iter=10000)
    add_coffee_break(solver, TrainingSummary(), every_n_iter=10000)
    add_coffee_break(solver, Snapshot(expdir), every_n_iter=50000)
    
    
    # add_coffee_break(solver, ValidationPerformance(test_net), every_n_iter=10000)
    # add_coffee_break(solver,ValidationPerformance(final_net),every_n_iter=50000)
    
    solve(solver, net)
    
    #open(string(rootname,".dot"), "w") do out net2dot(out, net) end
    
    # Evaluation network. Run against the validation set
    # Loss function is set to be squared loss as we have continuous outcomes (Xin)
    data_layer_test = MemoryDataLayer(name=string(rootname,"-valid"), data=Array[vlX,vlY],batch_size=1)
    # acc_layer = SquareLossLayer(name=string(rootname,"-accuracy"), bottoms=[:ip5, :label])
    pred_layer = HDF5OutputLayer(filename=string(rootname,"-vlpred.h5"),bottoms=[:ip5])
    test_net = Net(string(rootname,"-valid"), backend, [data_layer_test, common_layers..., pred_layer])
    load_snapshot(test_net,"snapshots/snapshot-050000.jld")
    init(test_net)
    forward_epoch(test_net)
    destroy(net)
    destroy(test_net)
    shutdown(backend)
    
    nn_vlpred = h5open("cytof-5-vlpred.h5", "r") do file
        read(file, "ip5")
    end
    diff = mapslices(Base.norm,nn_vlpred-vlY,[1])
    sum_loss = sum_loss + sum(diff.^2)/(2*size(diff)[2])
    rm("cytof-5-vlpred.h5")
    rm("snapshots/snapshot-000000.jld")
    rm("snapshots/snapshot-050000.jld")
    rm("snapshots/statistics.jld")
end
sum_loss/10

## all using Adam
## 'Tanh' best; train - regu_coef, mom, lr (3)
## 0.001, 0.8, (0.01, 0.0001, 0.75) [90,45,45]: 5.7222
## 0.0001, 0.8, (same)                        : 5.6759 *
## 0.0001, 0.8, (0.005,same)                  : 5.6929
## 0.0001, 0.8, (0.05,same)                   : 5.7354
## as * one more layer[90,90,45,45]           : 5.6545 ** (final model)



##########################################
## apply to test set
trX = transpose(dfpredTrain)
trY = transpose(dfrespTrain)
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
@time solve(solver, net)  ## 195.64s

open(string(rootname,".dot"), "w") do out net2dot(out, net) end
test_data = MemoryDataLayer(name=string(rootname,"-test"),data=Array[teX,teY],batch_size=1)
pred_layer = HDF5OutputLayer(filename=string(rootname,"-tepred.h5"),bottoms=[:ip5])
final_net = Net(string(rootname,"-test"),backend,[test_data, common_layers..., pred_layer])
load_snapshot(final_net,"snapshots/snapshot-050000.jld")
init(final_net)
forward_epoch(final_net)
destroy(net)
destroy(final_net)
shutdown(backend)

nn_tepred = h5open("cytof-5-tepred.h5", "r") do file
    read(file, "ip5")
end
diff = mapslices(Base.norm,nn_tepred-teY,[1])
sum(diff.^2)/(2*size(diff)[2]) ## 5.6474
# MSE for individual response
nn_pt_loss=zeros(18)
for i=1:18
    nn_pt_loss[i] = sum((nn_tepred[i,:]-teY[i,:]).^2)/(2*size(nn_tepred)[2])
end
nn_pt_loss
## 0.2072 0.3602 0.4260 0.2256 0.2338 0.2218 0.2174 0.3666 0.1904 0.2729 0.2307 0.3275 0.3255 0.7566 0.3793 0.3902 0.2594 0.2563
