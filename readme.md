# Data
- For individual 1, we have collected 100,000 cells, and for each cell we have 50 features: 18 surface markers (which identify the type of cell) and 32 functional markers (which identify the function of the cell)
- We collect this information at baseline: matrix 100k by 50, and also at several experimental moments (increase temperature, chemicals, electroshock).
- So, we have N experiments => N+1 matrices 100k by 50: B (baseline), E_1,...,E_N
- We want to use the baseline information to predict the funcional markers at different experiments (see below for more details) (also, surface markers do not change). That is, use B to predict E_i with a neural network
- Using the individual 1 as training, we can then only get baseline information for future individuals and predict their functional markers. This is great because this experiments are expensive, so our method would be widely used to save money.


# Questions
- how do we distinguish among the matrices from the experimental moments? That is, we would need to know what experiment, or just add a column that says "experiment", and simply distinguish between experiments (instead of adding a temperature column, a electroshock column, a chemical column). Adding columns with experiments' characteristics would allow us to extrapolate beyond the current experimental settings.


# Email log
- From Suprateek
Regarding the flow cytometry problem, please see the attached write-up (`PQ.docx`). We want to do prediction of functional markers using deep learning, so this is different than the typical classification problem. The functional markers can be considered to be normally distributed after a simple transformation. There is a recent paper in Bioinformatics on gating (classification) of cells using deep learning, but they make certain assumptions which may not be realistic. The problem we are trying to solve is different and more meaningful and makes limited assumptions. My collaborator has the data ready to go and suggested we move quickly on this project. The  bioinformatics paper gives me hope of finding a suitable outlet for our work on the flow cytometry project.

https://www.ncbi.nlm.nih.gov/pubmed/29036374

- From Suprateek regarding the data.
Hi Guys
Please find that data attached. This data contains five different samples from the same individual at baseline. The goal is to predict the functional markers at baseline using information on the surface markers using a DL framework.  This can be our Aim1.

 
For Aim 2, which is the Aim that we discussed in the last meeting, we can build a DL framework for predicting functional markers for the different experimental conditions given the baseline surface markers and functional markers. If we have data on experimental conditions E_1,...,E_M, then we first train the model to learn the relationship between the surface and functional markers at baseline with the functional markers at E_1,...,E_M, based on data from one subject. Then we can use this trained model to predict the functional markers for E_1,...,E_M for the other subjects, given their baseline surface and functional marker measurements. This could potentially be used to avoid recording the functional marker measurements for E_1,...,E_M, for new subjects, saving time, effort, cost etc. We have data on 24 different experimental conditions. I am currently working to get access to this data set. But as a first step, it is good to start with Aim 1. 
 

The structure of the data is given below. Please let me know if you have questions. Also, I am happy to meet this Thu if needed.


Each row is a cell. The meaning of the columns are as follows:
 
surface markers
191-DNA             
193-DNA             
115-CD45
139-CD45RA      
142-CD19
144-CD11b
145-CD4
146-CD8
148-CD34
147-CD20
158-CD33
160-CD123
167-CD38
170-CD90
110_114-CD3
 
functional markers
141-pPLCgamma2           
150-pSTAT5       
152-Ki67
154-pSHP2
151-pERK1/2
153-pMAPKAPK2
156-pZAP70/Syk
159-pSTAT3
164-pSLP-76
165-pNFkB
166-IkBalpha
168-pH3
169-pP38
171-pBtk/Itk
172-pS6
174-pSrcFK
176-pCREB
175-pCrkL
 
cell_type_index (not needed for what we want to do)
our previous analysis of this dataset, clustering cells into 26 cell types.
 
the 26 subsequent columns (you can ignore these)
one-hot encoding of which of the 26 cell types each cell belongs to
 
the columns not mentioned above (please just ignore them)
