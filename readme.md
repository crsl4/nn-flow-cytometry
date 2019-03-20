# Prediction of functional markers of mass cytometry data via deep learning

All scripts for the analysis of the paper:
```
Prediction of functional markers of mass cytometry data via deep learning (2019). Solis-Lemus, C., X. Ma, M. Hostetter II, S. Kundu, P. Qiu, D. Pimentel-Alarcon.
```

# Data
- For individual 1, we have collected 100,000 cells, and for each cell we have 50 features: 18 surface markers (which identify the type of cell) and 32 functional markers (which identify the function of the cell)
- We collect this information at baseline: matrix 100k by 50. Future: collect data at several experimental moments. So, if we have N experiments => N+1 matrices 100k by 50: B (baseline), E_1,...,E_N
- We want to use the baseline information to predict the funcional markers from surface markers (which do not change with experimental settings). That is, use B to predict E_i with a neural network
- The structure of the data is given by: each row is a cell. The meaning of the columns are as follows:
 
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
 
 # Analyses

 See `script` folder. The file `notebook-log.md` has the detailed steps in the analyses.