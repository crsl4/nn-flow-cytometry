## Julia script to fit a neural network, after creating the input data
## with create-input.jl
## WARNING: you should not have snapshots folder already in wd
## Claudia April 2018

using Mocha, CSV

wd = "C:/Users/xma72/Documents/Deep_Learning_project/nn-flow-cytometry-master/results/"
cd(wd)
trainX = CSV.read(string(wd,"cytof-5-data-train-pred.txt"))
trX = convert(Array{Float32,2}, trainX)
trX = transpose(trX)
trainY = CSV.read(string(wd,"cytof-5-data-train-resp.txt"))
trY = convert(Array{Float32,2}, trainY)
trY = transpose(trY)
testX = CSV.read(string(wd,"cytof-5-data-test-pred.txt"))
teX = convert(Array{Float32,2}, testX)
teX = transpose(teX)
testY = CSV.read(string(wd,"cytof-5-data-test-resp.txt"))
teY = convert(Array{Float32,2}, testY)
teY = transpose(teY)

expdir = "snapshots/"
rootname = "cytof-5"

# Input
data_layer  = MemoryDataLayer(name=rootname, data=Array[trX, trY], batch_size=64, shuffle=true)
# data_layer  = HDF5DataLayer(name=rootname, source=traindata, batch_size=64, shuffle=false)

# Inner product layer, input is determined by the "bottoms" option,
# in this case, the data layer

fc1_layer  = InnerProductLayer(name="ip1", output_dim=90,
neuron=Neurons.Tanh(), weight_init = GaussianInitializer(std=.01),
bottoms=[:data], tops=[:ip1])

fc2_layer  = InnerProductLayer(name="ip2", output_dim=45,
neuron=Neurons.Tanh(), weight_init = GaussianInitializer(std=.01),
bottoms=[:ip1], tops=[:ip2])

fc3_layer  = InnerProductLayer(name="ip3", output_dim=45,
neuron=Neurons.Tanh(), weight_init = GaussianInitializer(std=.01),
bottoms=[:ip2], tops=[:ip3])


# Output dim is 18, identity activation function
fc4_layer  = InnerProductLayer(name="ip4", output_dim=18,bottoms=[:ip3], tops=[:ip4])

# Loss layer -- connected to the second IP layer and "label" from
# the data layer.
loss_layer = SquareLossLayer(name="loss", bottoms=[:ip4,:label])

# Configure and build
backend = CPUBackend()
init(backend)

# Putting the network together
common_layers = [fc1_layer, fc2_layer, fc3_layer, fc4_layer]
net = Net(rootname, backend, [data_layer, common_layers..., loss_layer])



# Setting up the solver, this is identical to the MNIST tutorial
method = SGD()
params = make_solver_parameters(method, max_iter=10000, regu_coef=0.005,
    mom_policy=MomPolicy.Fixed(0.9),
    lr_policy=LRPolicy.Inv(0.01, 0.0001, 0.75),
    load_from=expdir)

# params = make_solver_parameters(method, max_iter=20000, regu_coef=0.0,
#     lr_policy=LRPolicy.Inv(0.01, 0.0001, 0.5),
#     #lr_policy=LRPolicy.Fixed(0.002),
#     load_from=exp_dir)
solver = Solver(method, params)


## "This sets up the coffee lounge, which holds data reported during coffee breaks.
## Here we also specify a file to save the information we accumulated in coffee
## breaks to disk. Depending on the coffee breaks, useful statistics such as
## objective function values during training will be saved, and can be loaded later
## for plotting or inspecting."
setup_coffee_lounge(solver, save_into="$expdir/statistics.jld", every_n_iter=1000)

## "First, we allow the solver to have a coffee break after every 100 iterations
## so that it can give us a brief summary of the training process. By default
## TrainingSummary will print the loss function value on the last training mini-batch."
add_coffee_break(solver, TrainingSummary(), every_n_iter=100)

# Snapshot
add_coffee_break(solver, Snapshot(expdir), every_n_iter=500)

# Evaluation network. Run against the test set
# Loss function is set to be squared loss as we have continuous outcomes (Xin)
data_layer_test = MemoryDataLayer(name=string(rootname,"-test"), data=Array[teX,teY], batch_size=100)
acc_layer = SquareLossLayer(name=string(rootname,"-accuracy"), bottoms=[:ip4, :label])
test_net = Net(string(rootname,"-test"), backend, [data_layer_test, common_layers..., acc_layer])

add_coffee_break(solver, ValidationPerformance(test_net), every_n_iter=1000)

@time solve(solver, net)

# Generate a net.dot file to create a .png visualizing the network
# Comment out if Graphviz is not installed
# If GraphViz is installed and in your PATH,
# this command: dot -Tpng net.dot |> net.png
# will generate a .png from the .dot file
# open(string(rootname,".dot"), "w") do out net2dot(out, net) end

destroy(net)
destroy(test_net)
shutdown(backend)

