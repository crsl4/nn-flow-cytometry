## R script for exploratory plots of raw data in data/cytof-5-data
## Claudia April 2018

datafolder = "../data/cytof-5-data/"
dat1 = read.table(paste0(datafolder,"CyTOF54_Tube01_Day1_Unstim1_curated.fcs_eventnum_Ungated_Jnstim1_Day1_normalized_1_Unstim1_Singlets.fcs.txt"), header=TRUE, sep="\t")
#dat2 = read.table(paste0(datafolder,"CyTOF54_Tube02_Day1_Unstim2_curated.fcs_eventnum_Ungated_Unstim2_Day1_normalized_2_Unstim2_Singlets.fcs.txt"), header=TRUE, sep="\t")
#dat3 = read.table(paste0(datafolder,"CyTOF54_Tube03_Day2_Unstim3_curated.fcs_eventnum_Ungated_Unstim3_Day2_normalized_3_Unstim3_Singlets.fcs.txt"), header=TRUE, sep="\t")
#dat4 = read.table(paste0(datafolder,"CyTOF54_Tube04_Day2_Unstim4_curated.fcs_eventnum_Ungated_Unstim4_Day2_normalized_4_Unstim4_Singlets.fcs.txt"), header=TRUE, sep="\t")
#dat5 = read.table(paste0(datafolder,"CyTOF54_Tube05_Day3_Unstim5_curated.fcs_eventnum_Ungated_Unstim5_Day3_normalized_5_Unstim5_Singlets.fcs.txt"), header=TRUE, sep="\t")

## extract only surface and functional markers
subdat = subset(dat1,select=c("X191.DNA","X193.DNA","X115.CD45","X139.CD45RA","X142.CD19","X144.CD11b","X145.CD4","X146.CD8","X148.CD34","X147.CD20","X158.CD33","X160.CD123","X167.CD38","X170.CD90","X110_114.CD3","X141.pPLCgamma2","X150.pSTAT5","X152.Ki67","X154.pSHP2","X151.pERK1.2","X153.pMAPKAPK2","X156.pZAP70.Syk","X159.pSTAT3","X164.pSLP.76","X165.pNFkB","X166.IkBalpha","X168.pH3","X169.pP38","X171.pBtk.Itk","X172.pS6","X174.pSrcFK","X176.pCREB","X175.pCrkL"))

## Identifying which ones are surface and which ones are functional
attr(subdat, "surface") <- c("X191.DNA","X193.DNA","X115.CD45","X139.CD45RA","X142.CD19","X144.CD11b","X145.CD4","X146.CD8","X148.CD34","X147.CD20","X158.CD33","X160.CD123","X167.CD38","X170.CD90","X110_114.CD3")
attr(subdat, "functional") <- c("X141.pPLCgamma2","X150.pSTAT5","X152.Ki67","X154.pSHP2","X151.pERK1.2","X153.pMAPKAPK2","X156.pZAP70.Syk","X159.pSTAT3","X164.pSLP.76","X165.pNFkB","X166.IkBalpha","X168.pH3","X169.pP38","X171.pBtk.Itk","X172.pS6","X174.pSrcFK","X176.pCREB","X175.pCrkL")
attributes(subdat)
head(subdat)


## Histograms
library(Hmisc)
pdf("../results/hist-all.pdf")
hist.data.frame(subdat)
dev.off()

pdf("../results/hist-all-transformed.pdf")
hist.data.frame(log(subdat+1))
dev.off()

## Scatterplot
library(ggplot2)
library(GGally)
ggpairs(subdat[sample(1:nrow(subdat),9000,replace=FALSE),10:20])


## Only taking the first 5 each to simplify the plot
surface <- attr(subdat, "surface")[1:5]
functional <- attr(subdat, "functional")[1:5]

pdf("../results/scattermatrix1.pdf")
ggduo(
  subdat[sample(1:nrow(subdat),2000,replace=FALSE),], surface, functional,
  types = list(continuous = "smooth_lm"),
  xlab = "Surface",
  ylab = "Functional"
)
dev.off()
