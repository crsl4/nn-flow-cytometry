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
##attributes(subdat)
head(subdat)

#######################################################################
## Figure 1: exploratory analysis
#######################################################################
library(ggplot2)
p1 <- ggplot(subdat,aes(x=X193.DNA))+geom_density()+
    theme(
        plot.title = element_text(hjust=0.5, size=rel(1.8)),
        axis.title.x = element_text(size=rel(1.8)),
        axis.title.y = element_text(size=rel(1.8), angle=90, vjust=0.5, hjust=0.5),
        axis.text.x = element_text(colour="grey", size=rel(1.5), angle=0, hjust=.5, vjust=.5, face="plain"),
        axis.text.y = element_text(colour="grey", size=rel(1.5), angle=90, hjust=.5, vjust=.5, face="plain"),
        legend.text=element_text(size=rel(1.2)), legend.title=element_text(size=rel(1.5)),
        panel.background = element_blank(),
        axis.line = element_line(colour = "grey")
        ) +
    xlab("Surface marker (193.DNA)") + ylab("")
pdf("surface-hist.pdf",width=5, height=5)
p1
dev.off()

p1 <- ggplot(subdat,aes(x=X150.pSTAT5))+geom_density()+
    theme(
        plot.title = element_text(hjust=0.5, size=rel(1.8)),
        axis.title.x = element_text(size=rel(1.8)),
        axis.title.y = element_text(size=rel(1.8), angle=90, vjust=0.5, hjust=0.5),
        axis.text.x = element_text(colour="grey", size=rel(1.5), angle=0, hjust=.5, vjust=.5, face="plain"),
        axis.text.y = element_text(colour="grey", size=rel(1.5), angle=90, hjust=.5, vjust=.5, face="plain"),
        legend.text=element_text(size=rel(1.2)), legend.title=element_text(size=rel(1.5)),
        panel.background = element_blank(),
        axis.line = element_line(colour = "grey")
        ) +
    xlab("Functional marker (150.pSTAT5)") + ylab("")

pdf("functional-hist.pdf",width=5, height=5)
p1
dev.off()

n = length(subdat$X175.pCrkL)
prop = 0.1
subsubdat = subdat[sample(1:n,floor(prop*n),replace=FALSE),]
p <- ggplot(subsubdat, aes(x=X191.DNA,y=X151.pERK1.2)) + geom_point(alpha=0.1) +
        theme(
        plot.title = element_text(hjust=0.5, size=rel(1.8)),
        axis.title.x = element_text(size=rel(1.8)),
        axis.title.y = element_text(size=rel(1.8), angle=90, vjust=0.5, hjust=0.5),
        axis.text.x = element_text(colour="grey", size=rel(1.5), angle=0, hjust=.5, vjust=.5, face="plain"),
        axis.text.y = element_text(colour="grey", size=rel(1.5), angle=90, hjust=.5, vjust=.5, face="plain"),
        legend.text=element_text(size=rel(1.2)), legend.title=element_text(size=rel(1.5)),
        panel.background = element_blank(),
        axis.line = element_line(colour = "grey")
        ) +
    xlab("Surface marker (191.DNA)") + ylab("Functional marker (151.pERK1)")

pdf("scatterplot.pdf",width=5, height=5)
p
dev.off()



#######################################################################
## Figure results
#######################################################################

## Results copied from analysis-comb.jl, but
## later we will save them to files
## For revision: results copied from analysis-comb-cv.jl

## Original:
vecloss = c(6.7148, 11.5280, 5.9503, 5.6447) ## search for vec_loss
time = c(1.98,63.36,826.83,181.82) ##search for toc

## Revision:
vecloss = c(6.7147, 11.4857, 5.9399, 5.6474) ##search for vec_loss
time = c(1.905,99.95,1093.24,195.64) ##search for toc

method = c("Linear Model","Decision Tree", "Random Forest", "Neural Network")



df = data.frame(vecloss=vecloss, method=method)
str(df)
library(ggplot2)
##library(viridis)

p <- ggplot(df,aes(x=reorder(method,-vecloss),y=vecloss,fill=method))+geom_bar(stat="identity")+
        theme(
            plot.title = element_text(hjust=0.5, size=rel(1.8)),
            axis.title.x = element_text(size=rel(1.8)),
            axis.title.y = element_text(size=rel(1.8), angle=90, vjust=0.5, hjust=0.5),
            axis.text.x = element_text(colour="black", size=rel(1.7), angle=0, hjust=.5, vjust=.5, face="plain"),
            axis.text.y = element_text(colour="black", size=rel(1.7), angle=0, hjust=.5, vjust=.5, face="plain"),
            ##legend.text=element_text(size=rel(1.2)), legend.title=element_text(size=rel(1.5)),
            panel.background = element_blank(),
            axis.line = element_line(colour = "black")
            ) +
    xlab("") + ylab("Vector MSE") + guides(fill=FALSE) +coord_flip()

p

pdf("vector-mse.pdf", width=6,height=4)
p
dev.off()


###################################################################
## Figure time
###################################################################
df = data.frame(time=time, method=method)
str(df)

p <- ggplot(df,aes(x=reorder(method,-time),y=time,fill=method))+geom_bar(stat="identity")+
        theme(
            plot.title = element_text(hjust=0.5, size=rel(1.8)),
            axis.title.x = element_text(size=rel(1.8)),
            axis.title.y = element_text(size=rel(1.8), angle=90, vjust=0.5, hjust=0.5),
            axis.text.x = element_text(colour="black", size=rel(1.7), angle=0, hjust=.5, vjust=.5, face="plain"),
            axis.text.y = element_text(colour="black", size=rel(1.7), angle=0, hjust=.5, vjust=.5, face="plain"),
            ##legend.text=element_text(size=rel(1.2)), legend.title=element_text(size=rel(1.5)),
            panel.background = element_blank(),
            axis.line = element_line(colour = "black")
            ) +
    xlab("") + ylab("Time (sec.)") + guides(fill=FALSE) +coord_flip()

pdf("time.pdf", width=6,height=4)
p
dev.off()

##########################################################################3
## Individual MSE
##########################################################################3
method = c("Linear Model","Decision Tree", "Random Forest", "Neural Network")

## Original: search for pt_loss
lm = c(0.2308, 0.4301, 0.5954, 0.2353, 0.2490, 0.2427, 0.2414, 0.3983, 0.2016, 0.3520, 0.2792, 0.4322, 0.3795, 0.9793, 0.4539, 0.4587, 0.2931, 0.2624)
dt = c(0.4207, 0.7510, 0.8740, 0.4556, 0.4644, 0.4460, 0.4363, 0.7468, 0.3831, 0.5550, 0.4696, 0.6693, 0.6602, 1.5796, 0.7696, 0.8004, 0.5239, 0.5227)
rf = c(0.2183, 0.3881, 0.4480, 0.2374, 0.2429, 0.2321, 0.2278, 0.3863, 0.2005, 0.2869, 0.2441, 0.3433, 0.3425, 0.8015, 0.3982, 0.4095, 0.2725, 0.2703)
nn = c(0.206722,0.358628,0.426626,0.225359,0.233064,0.221218,0.217177,0.366514,0.190379,0.273522,0.231513,0.327084,0.325995,0.756497,0.379003,0.389693,0.25933,0.256416)

## Revision: search for pt_loss
lm = c(0.2308, 0.4301, 0.5953, 0.2353, 0.2490, 0.2427, 0.2414, 0.3983, 0.2016, 0.3520, 0.2792, 0.4322, 0.3795, 0.9793, 0.4539, 0.4587, 0.2931, 0.2624)
dt = c(0.4199, 0.7462, 0.8713, 0.4555, 0.4659, 0.4451, 0.4365, 0.7423, 0.3821, 0.5529, 0.4703, 0.6672, 0.6621, 1.5584, 0.7700, 0.7952, 0.5231, 0.5219)
rf = c(0.2180, 0.3867, 0.4463, 0.2373, 0.2429, 0.2315, 0.2275, 0.3861, 0.2005, 0.2867, 0.2436, 0.3429, 0.3419, 0.7989, 0.3979, 0.4096, 0.2720, 0.2697)
nn = c(0.2072,0.3602,0.4260,0.2256,0.2338,0.2218,0.2174,0.3666,0.1904,0.2729,0.2307,0.3275,0.3255,0.7566,0.3793,0.3902,0.2594,0.2563)

m = matrix(c(lm,dt,rf,nn),ncol=4)
vec = as.vector(t(m))
met = rep(method,18)
pred = rep(1:18, each=4)


dff = data.frame(method=met, mse=vec, pred=pred)
head(dff)

## p <- ggplot(dff,aes(x=method,y=mse))+geom_point()+
##     facet_wrap(~pred, nrow=3, ncol=6) +
##     theme(
##         plot.title = element_text(hjust=0.5, size=rel(1.8)),
##         axis.title.x = element_text(size=rel(1.8)),
##         axis.title.y = element_text(size=rel(1.8), angle=90, vjust=0.5, hjust=0.5),
##         axis.text.x = element_text(colour="grey", size=rel(1.5), angle=90, hjust=.5, vjust=.5, face="plain"),
##         axis.text.y = element_text(colour="grey", size=rel(1.5), angle=90, hjust=.5, vjust=.5, face="plain"),
##         legend.text=element_text(size=rel(1.2)), legend.title=element_text(size=rel(1.5)),
##         panel.background = element_blank(),
##         axis.line = element_line(colour = "grey")
##         ) +
##     xlab("") + ylab("MSE")

## pdf("mse.pdf",width=8,height=5.5)
## p
## dev.off()


p <- ggplot(dff,aes(x=pred,y=mse,col=method))+geom_point() + geom_line()+
    theme(
        plot.title = element_text(hjust=0.5, size=rel(1.8)),
        axis.title.x = element_text(size=rel(1.8)),
        axis.title.y = element_text(size=rel(1.8), angle=90, vjust=0.5, hjust=0.5),
        axis.text.x = element_text(colour="black", size=rel(1.7), angle=0, hjust=.5, vjust=.5, face="plain"),
        axis.text.y = element_text(colour="black", size=rel(1.7), angle=0, hjust=.5, vjust=.5, face="plain"),
        legend.text=element_text(size=rel(1.2)),
        panel.background = element_blank(),legend.title=element_blank(),
        axis.line = element_line(colour = "black")
        ) +
    scale_x_continuous(breaks=c(1:18))+
    xlab("Response") + ylab("MSE")


pdf("mse.pdf",width=8,height=5.5)
p
dev.off()



################################################################################################
## Plots per response
## We will focus in response 1 (good), 3, 8, 14 (bad)

## Original
dat1 = read.table("../results/Predictions/Test_1.txt", header=TRUE)
dat3 = read.table("../results/Predictions/Test_3.txt", header=TRUE)
dat8 = read.table("../results/Predictions/Test_8.txt", header=TRUE)
dat14 = read.table("../results/Predictions/Test_14.txt", header=TRUE)

## Revision
dat1 = read.table("../results/Test_prediction/Test_1_revision.txt", header=TRUE)
dat3 = read.table("../results/Test_prediction/Test_3_revision.txt", header=TRUE)
dat8 = read.table("../results/Test_prediction/Test_8_revision.txt", header=TRUE)
dat14 = read.table("../results/Test_prediction/Test_14_revision.txt", header=TRUE)


newdat = data.frame(response = c(dat1$True,dat3$True,dat8$True,dat14$True),
    which = c(rep(1,length(dat1$True)),rep(3,length(dat3$True)),rep(8,length(dat8$True)),rep(14,length(dat14$True)))
    )
str(newdat)

library(ggplot2)
library(viridis)
vpal = viridis(4,end=0.9)

p <- ggplot(newdat,aes(x=factor(which),y=response, fill=factor(which)))+geom_violin(draw_quantiles = c(0.25, 0.5, 0.75))+
    scale_fill_manual(values=vpal) +
    scale_x_discrete(name="Response") +
    scale_y_continuous(name="") +
    theme(
        plot.title = element_text(hjust=0.5, size=rel(1.8)),
        axis.title.x = element_text(size=rel(1.8)),
        axis.title.y = element_text(size=rel(1.8), angle=90, vjust=0.5, hjust=0.5),
        axis.text.x = element_text(colour="black", size=rel(1.7), angle=0, hjust=.5, vjust=.5, face="plain"),
        axis.text.y = element_text(colour="black", size=rel(1.7), angle=0, hjust=.5, vjust=.5, face="plain"),
        legend.text=element_text(size=rel(1.2)), legend.title=element_blank(),
        panel.background = element_blank(),
        axis.line = element_line(colour = "black"),
        legend.position = "none"
        )

##pdf("../manuscript/templates4Authors/latex_template/figures/violin.pdf",width=4,height=4)
pdf("../revision/latex_template/figures/violin.pdf",width=4,height=4)
p
dev.off()


## Response 14
ntot = length(dat14$True)
n = floor(0.01*ntot)
ind = sample(1:ntot,n,replace=FALSE)

true = c(dat14$True[ind],dat14$True[ind],dat14$True[ind],dat14$True[ind])
pred = c(dat14$LR[ind], dat14$DTR[ind], dat14$RFR[ind], dat14$NN[ind])
method = c(rep("Linear Model",n),rep("Decision Tree",n),
    rep("Random Forest",n),rep("Neural Network",n))


newdat2 = data.frame(true=true,pred=pred,method=method)
str(newdat2)

p <- ggplot(newdat2,aes(x=true,y=pred, col=method))+geom_point(alpha=0.1)+
    geom_smooth(method='lm')+
    geom_abline(slope=1, col="black")+
    scale_x_continuous(name="True 14th response", limits=c(-3,7.5)) +
    scale_y_continuous(name="Predicted 14th response", limits=c(-3, 7.5)) +
    theme(
        plot.title = element_text(hjust=0.5, size=rel(1.8)),
        axis.title.x = element_text(size=rel(1.8)),
        axis.title.y = element_text(size=rel(1.8), angle=90, vjust=0.5, hjust=0.5),
        axis.text.x = element_text(colour="black", size=rel(1.7), angle=0, hjust=.5, vjust=.5, face="plain"),
        axis.text.y = element_text(colour="black", size=rel(1.7), angle=0, hjust=.5, vjust=.5, face="plain"),
        legend.text=element_text(size=rel(1.2)), legend.title=element_blank(),
        panel.background = element_blank(),
        axis.line = element_line(colour = "black"),
        )

##pdf("../manuscript/templates4Authors/latex_template/figures/response14.pdf",width=6,height=4)
pdf("../revision/latex_template/figures/response14.pdf",width=6,height=4)
p
dev.off()

## Response 1
ntot = length(dat1$True)
n = floor(0.01*ntot)
ind = sample(1:ntot,n,replace=FALSE)

true = c(dat1$True[ind],dat1$True[ind],dat1$True[ind],dat1$True[ind])
pred = c(dat1$LR[ind], dat1$DTR[ind], dat1$RFR[ind], dat1$NN[ind])
method = c(rep("Linear Model",n),rep("Decision Tree",n),
    rep("Random Forest",n),rep("Neural Network",n))


newdat2 = data.frame(true=true,pred=pred,method=method)
str(newdat2)

p <- ggplot(newdat2,aes(x=true,y=pred, col=method))+geom_point(alpha=0.1)+
    geom_smooth(method='lm')+
    geom_abline(slope=1, col="black")+
    scale_x_continuous(name="True 1st response", limits=c(-3,7.5)) +
    scale_y_continuous(name="Predicted 1st response", limits=c(-3,7.5)) +
    theme(
        plot.title = element_text(hjust=0.5, size=rel(1.8)),
        axis.title.x = element_text(size=rel(1.8)),
        axis.title.y = element_text(size=rel(1.8), angle=90, vjust=0.5, hjust=0.5),
        axis.text.x = element_text(colour="black", size=rel(1.7), angle=0, hjust=.5, vjust=.5, face="plain"),
        axis.text.y = element_text(colour="black", size=rel(1.7), angle=0, hjust=.5, vjust=.5, face="plain"),
        legend.text=element_text(size=rel(1.2)), legend.title=element_blank(),
        panel.background = element_blank(),
        axis.line = element_line(colour = "black"),
        legend.position="none"
        )

##pdf("../manuscript/templates4Authors/latex_template/figures/response1.pdf",width=4,height=4)
pdf("../revision/latex_template/figures/response1.pdf",width=4,height=4)
p
dev.off()


## Scatterplots

## Response 14
pdat = read.table("../results/Predictions/Test_pred.txt", header=TRUE)
str(pdat)
colnames(pdat) = c("X191.DNA","X193.DNA","X115.CD45","X139.CD45RA","X142.CD19","X144.CD11b","X145.CD4","X146.CD8","X148.CD34","X147.CD20","X158.CD33","X160.CD123","X167.CD38","X170.CD90","X110.114.CD3")

## Original
rdat = read.table("../results/Predictions/Test_14.txt", header=TRUE)
## Revision
rdat = read.table("../results/Test_prediction/Test_14_revision.txt", header=TRUE)

str(rdat)

dat = cbind(pdat,rdat)
str(dat)

n = length(dat$NN)
prop = 0.01
subdat = dat[sample(1:n,floor(prop*n),replace=FALSE),]
str(subdat)

library(ggplot2)
p <- ggplot(subdat, aes(x=X115.CD45,y=True)) + geom_point(alpha=0.1) + geom_smooth()+
    scale_y_continuous(limits=c(-3,7.5))+
        theme(
        plot.title = element_text(hjust=0.5, size=rel(1.8)),
        axis.title.x = element_text(size=rel(1.8)),
        axis.title.y = element_text(size=rel(1.8), angle=90, vjust=0.5, hjust=0.5),
        axis.text.x = element_text(colour="black", size=rel(1.7), angle=0, hjust=.5, vjust=.5, face="plain"),
        axis.text.y = element_text(colour="black", size=rel(1.7), angle=0, hjust=.5, vjust=.5, face="plain"),
        legend.text=element_text(size=rel(1.2)), legend.title=element_text(size=rel(1.5)),
        panel.background = element_blank(),
        axis.line = element_line(colour = "black")
        ) +
    xlab("Surface marker (115.CD45)") + ylab("Functional marker (171.pBtk.Itk)")

pdf("scatterplot14.pdf",width=6, height=6)
p
dev.off()


## Response 1

## Original
rdat = read.table("../results/Predictions/Test_1.txt", header=TRUE)
## Revision
rdat = read.table("../results/Test_prediction/Test_1_revision.txt", header=TRUE)

str(rdat)

dat = cbind(pdat,rdat)
str(dat)

n = length(dat$NN)
prop = 0.01
subdat = dat[sample(1:n,floor(prop*n),replace=FALSE),]
str(subdat)

library(ggplot2)
p <- ggplot(subdat, aes(x=X115.CD45,y=True)) + geom_point(alpha=0.1) + geom_smooth()+
    scale_y_continuous(limits=c(-3,7.5))+
        theme(
        plot.title = element_text(hjust=0.5, size=rel(1.8)),
        axis.title.x = element_text(size=rel(1.8)),
        axis.title.y = element_text(size=rel(1.8), angle=90, vjust=0.5, hjust=0.5),
        axis.text.x = element_text(colour="black", size=rel(1.7), angle=0, hjust=.5, vjust=.5, face="plain"),
        axis.text.y = element_text(colour="black", size=rel(1.7), angle=0, hjust=.5, vjust=.5, face="plain"),
        legend.text=element_text(size=rel(1.2)), legend.title=element_text(size=rel(1.5)),
        panel.background = element_blank(),
        axis.line = element_line(colour = "black")
        ) +
    xlab("Surface marker (115.CD45)") + ylab("Functional marker (141.pPLCgamma2)")

pdf("scatterplot1.pdf",width=6, height=6)
p
dev.off()
