substrRight <- function(x, n){
  substr(x, nchar(x)-n+1, nchar(x))
}

# load data
data = read.csv("InfluenzaData.csv", header=T)
dim(data)
#get indices of the six vaccination datasets
table.ind = as.character(substrRight(unlist(data[1]),3))
table(table.ind)
#create data matrix
data=data[,-1]
data2=matrix(as.numeric(unlist(data)), ncol=81)
colnames(data2)=colnames(data)
rownames(data2)=rownames(data)
data2=log10(data2)

#Example 1: predict antibody responses against the virus "A.PANAMA.2007.99" in Table S14 using Table S13.

data.t = data2[which(table.ind==unique(table.ind)[6]), ]
data.s.list = list(data2[which(table.ind==unique(table.ind)[5]), ])
feature.t = "A.PANAMA.2007.99"

#run the RF.complete function
out = RF.complete(data.t, data.s.list, feature.t)

#plot the predicted values against the true values
plot(data.t[,which(colnames(data.t)==feature.t)], out$predictions, xlab = "log10(HAI Measurements)", ylab = "log10(Predicted Values)")
#add error bars from predicted RMSE (in log10 scale)
arrows(x0=data.t[,which(colnames(data.t)==feature.t)], y0=out$predictions-out$errors, 
       y1=out$predictions+out$errors, code=3, angle=90, length=0.05)
#actual prediction error (as n-folds)
10^sqrt(mean(((data.t[,which(colnames(data.t)==feature.t)])-(out$predictions))^2, na.rm=T))
#predicted error (as n-folds)
10^(out$errors)


#Example 2: predict antibody responses against the virus "A.AUCKLAND.5.96" in Table S14 using Tables S5, S6 and S13.

data.t = data2[which(table.ind==unique(table.ind)[6]), ]
data.s.list = list(data2[which(table.ind==unique(table.ind)[5]), ],
                   data2[which(table.ind==unique(table.ind)[4]), ],
                   data2[which(table.ind==unique(table.ind)[3]), ])
feature.t = "A.AUCKLAND.5.96"

#run the RF.complete function
out = RF.complete(data.t, data.s.list, feature.t)

#plot the predicted values against the true values
plot(data.t[,which(colnames(data.t)==feature.t)], out$predictions, xlab = "log10(HAI Measurements)", ylab = "log10(Predicted Values)")
#add error bars from predicted RMSE (in log10 scale)
arrows(x0=data.t[,which(colnames(data.t)==feature.t)], y0=out$predictions-out$errors, 
       y1=out$predictions+out$errors, code=3, angle=90, length=0.05)
#actual prediction error (as n-folds)
10^sqrt(mean(((data.t[,which(colnames(data.t)==feature.t)])-(out$predictions))^2, na.rm=T))
#predicted error (as n-folds)
10^(out$errors)
