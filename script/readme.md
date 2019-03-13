This is the readme file for the script folder.\
The analysis process shown in the manuscript has been combined into the Julia script named 'analysis-comb-cv.jl'. First section of 
the code preprocessed the data into the format for analysis purpose. The samples were split randomly into training and testing sets. 
The training set was then used in 10-fold cross-validation to tune the parameters in the models including linear regression, decision tree 
regression, random forest regression and feedforward neural network. The testing set was reserved to evaluate the performance across the 
different models following tuning process.
