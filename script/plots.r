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

vecloss = c(6.7148, 11.5280, 5.9503, 5.6447)
method = c("Linear Model","Decision Tree", "Random Forest", "Neural Network")
time = c(1.98,63.36,826.83,181.82)


df = data.frame(vecloss=vecloss, method=method)
str(df)
library(ggplot2)
##library(viridis)

p <- ggplot(df,aes(x=reorder(method,-vecloss),y=vecloss,fill=method))+geom_bar(stat="identity")+
        theme(
            plot.title = element_text(hjust=0.5, size=rel(1.8)),
            axis.title.x = element_text(size=rel(1.8)),
            axis.title.y = element_text(size=rel(1.8), angle=90, vjust=0.5, hjust=0.5),
            axis.text.x = element_text(colour="grey", size=rel(1.5), angle=0, hjust=.5, vjust=.5, face="plain"),
            axis.text.y = element_text(colour="grey", size=rel(1.5), angle=0, hjust=.5, vjust=.5, face="plain"),
            ##legend.text=element_text(size=rel(1.2)), legend.title=element_text(size=rel(1.5)),
            panel.background = element_blank(),
            axis.line = element_line(colour = "grey")
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
            axis.text.x = element_text(colour="grey", size=rel(1.5), angle=0, hjust=.5, vjust=.5, face="plain"),
            axis.text.y = element_text(colour="grey", size=rel(1.5), angle=0, hjust=.5, vjust=.5, face="plain"),
            ##legend.text=element_text(size=rel(1.2)), legend.title=element_text(size=rel(1.5)),
            panel.background = element_blank(),
            axis.line = element_line(colour = "grey")
            ) +
    xlab("") + ylab("Time (sec.)") + guides(fill=FALSE) +coord_flip()

pdf("time.pdf", width=6,height=4)
p
dev.off()

##########################################################################3
## Individual MSE
##########################################################################3
method = c("Linear Model","Decision Tree", "Random Forest", "Neural Network")

lm = c(0.2308, 0.4301, 0.5954, 0.2353, 0.2490, 0.2427, 0.2414, 0.3983, 0.2016, 0.3520, 0.2792, 0.4322, 0.3795, 0.9793, 0.4539, 0.4587, 0.2931, 0.2624)
dt = c(0.4207, 0.7510, 0.8740, 0.4556, 0.4644, 0.4460, 0.4363, 0.7468, 0.3831, 0.5550, 0.4696, 0.6693, 0.6602, 1.5796, 0.7696, 0.8004, 0.5239, 0.5227)
rf = c(0.2183, 0.3881, 0.4480, 0.2374, 0.2429, 0.2321, 0.2278, 0.3863, 0.2005, 0.2869, 0.2441, 0.3433, 0.3425, 0.8015, 0.3982, 0.4095, 0.2725, 0.2703)
nn = c(0.206722,0.358628,0.426626,0.225359,0.233064,0.221218,0.217177,0.366514,0.190379,0.273522,0.231513,0.327084,0.325995,0.756497,0.379003,0.389693,0.25933,0.256416)

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
        axis.text.x = element_text(colour="grey", size=rel(1.5), angle=0, hjust=.5, vjust=.5, face="plain"),
        axis.text.y = element_text(colour="grey", size=rel(1.5), angle=0, hjust=.5, vjust=.5, face="plain"),
        legend.text=element_text(size=rel(1.2)), legend.title=element_text(size=rel(1.5)),
        panel.background = element_blank(),
        axis.line = element_line(colour = "grey")
        ) +
    scale_x_continuous(breaks=c(1:18))+
    xlab("Predictor") + ylab("MSE")


pdf("mse.pdf",width=8,height=5.5)
p
dev.off()


