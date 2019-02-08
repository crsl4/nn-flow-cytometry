# Answers to reviewers

We thank the reviewers for such insightful comments which had certainly improved the manuscript greatly.

## From reviewer 1

1. "For the deep learning framework, there is not too much novelty. In my opinion, the paper is a pioneer of practitioner of the deep learning in their specific research area."
We agree! Neural networks are widely studied models that have been applied to a variety of real-life datasets. We are not proposing new model developments for the area of neural network; we are applying existing neural network tools to the case of flow cytometry area which has not fully adopted the deep learning schemes yet. To our knowledge, there has been a limited development of neural network approaches for prediction of cell proteins based on flow cytometry data, and the proposed method fills this gap. We hope that this work will encourage the development of future deep learning approaches in flow cytometry.
We added these points in the introduction/discussion

2. "When they split the data into training, validation, test datasets, it’s better to follow the cross-validation schema, that is, try different repetition and report the average results to reduce the randomness."
This is a great point! We have changed the analysis accordingly: first the data was split into training set (1000,000 samples) and testing set (223,228 samples), and the training set was used to perform the 10-fold cross validation. Based on the average MSE from the cross validation, we chose the model with the smallest average MSE. Then we used all the training samples to obtain the final model and reported the prediction MSE of the testing samples. 

SK comment: we need to insert what changes in the results we observed, if any, when we used the cross-validation approach versus the results reported earlier. 

3. "For the performance, we can see that the margin between NN and Random forest or linear model is not too much. So, a better deep learning framework is still desired."
We agree! Sometimes in deep learning comparisons, margins in performance can be relative. As we are comparing accuracy in prediction, random forest and NN are both quite competitive, with the former losing in computation time, rather than prediction accuracy. One key advantage of the NN over random forest is that the former is able to handle a larger number of covariates, and hence is more scalable to higher dimensional feature spaces compared to the random forest. We expect the computation time under random forest to slow down even further as the number of covariates increases. We added this discussion to the manuscript.

SK comment: did the performance of the NN improve relative to random forest under the cross-validation criteria?


## From reviewer 2

1. "The study did not compare with support vector regression (SVR), citing computational time constraints. It is somewhat surprising since SVR is not a time-consuming algorithm comparing to deep learning.  Although the paper doesn't have to do new work, some discussions along these lines could be added" 
It is our understanding that even when SVM are widely used in a variety of scenarios, SVR models have not been successfully adopted in big data settings given their complexity.
One possible reason for the slow computation speed for SVR is that non-linear SVR involve a radial basis function that computes a n x n kernel matrix at each iteration, where n is the number of samples (cells) which is very large for our application.  Computing and storing such a matrix is likely to slow down computation and this is likely what we experienced when we tried to implement the SVR approach. We have added these details and some references to the paper.

2. "The quality of the figures can be improved. Some labels in the figure are in light color, which is not easy to see. Some graphs (e.g. two in Fig. 2) have no axis labels."
Thank you for this comment. We have increased the font and changed the color of the figure labels. Figure 2 cannot have axis as it represents the neural network architecture.

3. "It would be good if the data used in this study can be made available to the public so that it can be served as a benchmark for studies in the community."
Yes we fully agree. The data is already publically available from the original paper that published the experiment. Here is the link: http://reports.cytobank.org/1/v1
We added a section about Data Sharing in the manuscript.