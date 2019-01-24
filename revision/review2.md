# To do
- improve writing based on review1 (doc file)
- improve writing and plots based on review2 (below)
- answers to reviewers

Report 1:

This paper presented a new deep learning architecture for predicting functional markers in the cells given data on surface markers. This is a very meaningful and useful study. The idea is novel and the method is technically sound.


The deep learning model could be more sophisticated. The study did not compare with support vector regression (SVR), citing “computational time constraints”. It is somewhat surprising since SVR is not a time-consuming algorithm comparing to deep learning.  Although the paper doesn’t have to do new work, some discussions along these lines could be added.


The writing of this paper is sketchy. More details should be added to make this paper standalone. For example, what exactly are the x1, x2, …, x15 input features? How do they related to the graphs in Fig. 1?


The quality of the figures can be improved. Some labels in the figure are in light color, which is not easy to see. Some graphs (e.g. two in Fig. 2) have no axis labels.


It would be good if the data used in this study can be made available to the public so that it can be served as a benchmark for studies in the community.


# Email to group
Hello,

Xin and I met this morning to go over the reviews. They seem fairly easy to address.

From all of you, I need to know:
-- If you have any input/comments/suggestions to address the reviews I described below. In particular, if you have any info about SVR, please let me know!
-- Peng, one reviewer wanted the data to be made public to serve as a benchmark for the community. So this is completely up to you, and let me know what to reply.


Regarding the specific reviews:
1) Xin will address the "cross-validation" suggestion by re-running our analyses with a 10-fold cross-validation framework. I think this is a good suggestion
2) I will address the comments regarding the wording, plots and discussion. Mainly addressing:
2.1) "for the deep learning framework, there is not too much novelty. In my opinion, the paper is a pioneer of practitioner of the deep learning in their specific research area" -> answer: yes, we are not inventing neural networks, we are using them in this area where they have not been used before
2.2) "for the performance, we can see that the margin between NN and Random forest or linear model is not too much. SO, a better deep learning framework is still desired" --> answer: in deep learning margins in performance can be relative. As we are comparing accuracy in prediction, random forest and NN are both quite competitive, with the former losing in computation time, rather than prediction accuracy. Other input is welcome.
2.3) "The study did not compare with support vector regression (SVR), citing “computational time constraints”. It is somewhat surprising since SVR is not a time-consuming algorithm comparing to deep learning.  Although the paper doesn’t have to do new work, some discussions along these lines could be added" answer -> We really tried to fit SVR and we could not make it work. Perhaps the reviewer is confusing it with SVM. SVR are a different animal altogether, but I do not know enough about SVR to give proper citations to backup our claims. Any suggestions are welcome.
2.4) Improve readability of figures



Please let me know your suggestions for the revision by the end of next week, if possible, as Xin and I work on the new version of the manuscript.
Thanks,
Claudia

