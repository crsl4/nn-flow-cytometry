# Answers to reviewers

We thank the reviewers for such insightful comments which had certainly improved the manuscript greatly.

## From reviewer 1

1. "For the deep learning framework, there is not too much novelty. In my opinion, the paper is a pioneer of practitioner of the deep learning in their specific research area."
yes, we are not inventing neural networks, we are using them in this area where they have not been used before. we added this in the intro/discussion

2. "When they split the data into training, validation, test datasets, it’s better to follow the cross-validation schema, that is, try different repetition and report the average results to reduce the randomness."
definitely! we changed the analysis:  first the data was split into training set (1000,000 samples) and testing set (223,228 samples), and the training set was used to perform the 10-fold cross validation. Based on the average MSE from the cross validation, we chose the model with the smallest average MSE. Then we used all the training samples to obtain the final model and reported the prediction MSE of the testing samples.

3. "For the performance, we can see that the margin between NN and Random forest or linear model is not too much. SO, a better deep learning framework is still desired."
answer: in deep learning margins in performance can be relative. As we are comparing accuracy in prediction, random forest and NN are both quite competitive, with the former losing in computation time, rather than prediction accuracy. Other input is welcome. we added this in the intro/discussion


## From reviewer 2

1. "The study did not compare with support vector regression (SVR), citing “computational time constraints”. It is somewhat surprising since SVR is not a time-consuming algorithm comparing to deep learning.  Although the paper doesn’t have to do new work, some discussions along these lines could be added" 
It is our understanding that even when SVM are widely used in a variety of scenarios, SVR have not been as adopted in big data settings given its complexity.
One possible reason for the slow computation speed for SVR is if we are fitting a non-linear SVR then it could involve a radial basis function that involves computing a nXn kernel matrix where n is the number of samples which is very large for our application. 

2. "The quality of the figures can be improved. Some labels in the figure are in light color, which is not easy to see. Some graphs (e.g. two in Fig. 2) have no axis labels."
Done, and figure 2 cannot have axis as it is the neural network architecture

3. "It would be good if the data used in this study can be made available to the public so that it can be served as a benchmark for studies in the community."
From Peng: The data is already publically available from the original paper that published the experiment, although one needs to learn a little bit about Cytobank to be able to get the data. Here is the link: http://reports.cytobank.org/1/v1