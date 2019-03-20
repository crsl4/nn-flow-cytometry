## function that takes the two Array{Float32,2}: X,Y and writes
## the Mocha input files as HDF5 (rootname.hdf5) and txt (rootname.txt)
function writeMochaInput(dfX::DataFrame,dfY::DataFrame,rootname::AbstractString)
    X = convert(Array{Float32,2}, dfX)
    Y = convert(Array{Float32,2}, dfY)

    # Mocha needs transposed before writing to file
    Xt = transpose(X)
    Yt = transpose(Y)
    Yt = Yt[1,:] ##fixit: using only one response

    # Each .hdf5 file has two datasets "data" and "label"
    hdf5file = string(rootname,".hdf5")
    h5open(hdf5file, "w") do h5
        write(h5, "data", Xt)
        write(h5, "label", Yt)
    end

    # Mocha reads as input a textfile with the name of the datafile
    txtfile = string(rootname,".txt")
    run(pipeline(Cmd(`echo $hdf5file`), stdout=txtfile, stderr=string(rootname,".err")))
    println("Writing files to $hdf5file and $txtfile")
end
