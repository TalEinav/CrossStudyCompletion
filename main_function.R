library(rpart)


#data.t: dataset-of-interest with columns being features and rows being samples, containing feature-of-interest whose measurements we predict;
#data.s.list: a list of other assisting dataset(s) with columns being features and rows being samples, each containing the feature-of-interest and at least 5 additional features that overlap with the data.t
#data.t and the datasets in data.s.list should have the same features.
#feature.t: name of the feature-of-interest contained in data.t and as well as the datasets in data.s.list.
#n.tree: number of decision trees trained for each assiting data; default 50
#n.feature: number of other features in each assisting data used for prediction; default 5
#f.sample: proportion of samples in each assiting data used for training; default 0.3.

RF.complete <- function(data.t, data.s.list, feature.t, n.tree = 50, n.feature = 5, f.sample = 0.3,
                        n.best.tree = 5){
  
  ##########################
  ####   check for input
  ##########################
   
  #get number of source datasets  
   K = length(data.s.list)
   print(paste0(K," additional datasets used for prediction."))
   
  #check feature-of-interest
   if(sum(feature.t == colnames(data.t))!=1){
     print("feature-of-interest not found in data.t! please check column names of input data!")
   }else{
       print("feature-of-interest located!")
   }
   
   ########################
   #### tree construction
   ########################
   
   mu = list()
   sigma = list()
   for(k in 1:K){
     
     data.assist = data.s.list[[k]]
     
     
     ###locate feature-of-interest
     if(sum(colnames(data.assist)==feature.t)>0){
       f.t.ind = which(colnames(data.assist)==feature.t)  #find location of feature.t
     }else{
       print(paste0("feature.t not found in assisting data ", k,"!"))
     }
     
     ###check feature match
     if(sum(colnames(data.t) != colnames(data.assist))>0){
       print(paste0("Features not matched for assisting data ", k, "! Skipped to next data."))
       next
     }
    
     ####transferability function
     trans.true.err = c()
     trans.pred.err = c()
     for(j in (1:dim(data.assist)[2])[-f.t.ind]){
       
       feature.trans = colnames(data.assist)[j]
       if(sum(!is.na(data.t[,j]))>0){
         #print(j)
         RF1t1 = RF.complete.1t1(data.assist, data.t, feature.t=feature.trans, n.tree = n.tree, n.feature = n.feature,
                                 f.sample = f.sample, k=k)
         if(!is.logical(RF1t1)){
         trans.true.err = c( trans.true.err, (RF1t1$true.err)) #mean or not mean
         trans.pred.err = c( trans.pred.err, (RF1t1$pred.err)) #mean or not mean
         }
       }
     
     }
     
     if(length(trans.true.err)==0){
       next
     }else{
     lm.out = lm(trans.true.err~trans.pred.err)
     a = coef(lm.out)[2]
     b = coef(lm.out)[1]
     c = sqrt(mean((a*trans.pred.err+b-trans.true.err)^2, na.rm=T))
     
     f.transfer <- function(x){
       return(max(c(x, a*x+b+c)))
     }
     print(c(paste0("a=",round(a,3)),paste0("b=",round(b,3)),paste0("c=",round(c,3))))
     
     
     ##### predict feature.t
     RF1t1 = RF.complete.1t1(data.assist, data.t, feature.t=feature.t, n.best.tree = n.best.tree, n.tree = n.tree, n.feature = n.feature,
                             f.sample = f.sample, k=k)
     mu[[k]] = RF1t1$mu
     sigma[[k]] = f.transfer(mean(RF1t1$pred.err))
     #print(c(mean(RF1t1$pred.err), sigma[[k]]))
     }
   }
   
   A=0
   B=0
   tt=0
  for(k in 1:K){
    if(!is.null(sigma[[k]])){
    tt=tt+1
    A = A + mu[[k]]/sigma[[k]]^2
    B = B + 1/sigma[[k]]^2
    }
  }
   print(paste0(tt, " assisting data used for prediction."))
  return(list(predictions = A/B, errors = 1/sqrt(B)))
  
}


#the following funnction is also needed for RF.complete
RF.complete.1t1 <- function(data.assist, data.t, feature.t, n.tree = n.tree, n.feature = n.feature, f.sample = f.sample,
                            n.best.tree = 5, k = 1){
  
  
  
  ###locate feature-of-interest
  if(sum(colnames(data.assist)==feature.t)>0){
    f.t.ind = which(colnames(data.assist)==feature.t)  
  }else{
    print(paste0("feature.t not found in assisting data ", k,"!"))
  }
  
  
  ###find feasible features
  if(sum(apply(!is.na(data.assist), 2, sum)/dim(data.assist)[1] > 0.8)>n.feature){
    f.ind = which(apply(!is.na(data.assist), 2, sum)/dim(data.assist)[1] > 0.8)
    #remove feature-of-interest from f.ind
    if(sum(names(f.ind)==feature.t)>0){
      f.ind = f.ind[-which(names(f.ind)==feature.t)]
    }
  }else{
    print(paste0("n.feature too large for assisting data ", k, "! Skipped to next data."))
    return(FALSE)
  }
  
  ### eliminate features that are not more than 2 samples in data.t
  f.tmp.ind = match(names(f.ind), colnames(data.t))
  f.feasible = colnames(data.t)[f.tmp.ind][which(colSums(!is.na(data.t[,f.tmp.ind]))>2)]
  f.ind = f.ind[match(f.feasible, names(f.ind))]
  if(length(f.ind)<2){
    print(paste0("n.feature too large for assisting data ", k, "! Skipped to next data."))
    return(FALSE)
  }
  
  ###get bootstrap samples and features, normalize
  data.assist = data.assist[!is.na(data.assist[,f.t.ind]),]#remove samples with missing feature-of-interest
  
  
  RMSE = c()
  f.sel.ind = list()
  tree=list()
  for(i in 1:n.tree){
    f.sel.ind[[i]] = f.ind[sample(length(f.ind), n.feature, replace = TRUE)]
    sample.sel = sample(dim(data.assist)[1], dim(data.assist)[1]*f.sample, replace = TRUE)
    data.train = data.assist[sample.sel, c(f.sel.ind[[i]], f.t.ind)]

    colm.t=apply(data.train, 1, mean, na.rm=T)
    data.train=data.train- t(rep(1, dim(data.train)[2]) %*% t(colm.t))
    colnames(data.train)[n.feature+1] = "target"
    
    ###get decision tree and RMSE
    tree[[i]]=rpart(`target`~., data=data.frame(data.train), control=rpart.control(minsplit = 5))
    data.test =  data.assist[-sample.sel, c(f.sel.ind[[i]], f.t.ind)]

    colm.t=apply(data.test, 1, mean, na.rm=T)
    data.test=data.test- t(rep(1, dim(data.test)[2]) %*% t(colm.t))
    pred.t = predict(tree[[i]], newdata = data.frame(data.test), type="vector")
    RMSE[i] = sqrt(mean((pred.t - data.test[,n.feature+1])^2))
  }
  
  ###predict in the target dataset
  pred.list = matrix(ncol=n.best.tree, nrow = dim(data.t)[1])
  for(i in 1:n.best.tree){
    j=order(RMSE, decreasing = F)[i]
    f.t.ind = which(colnames(data.t)==feature.t) 
    f.t.sel.ind = match(names(f.sel.ind[[j]]), colnames(data.t))
    data.test =  data.t[, c(f.t.sel.ind, f.t.ind)]

    colm.t=apply(data.test, 1, mean, na.rm=T)
    data.test = data.test- t(rep(1, dim(data.test)[2]) %*% t(colm.t))
    colnames(data.test)[n.feature+1] = "target"
    pred.t = predict(tree[[j]], newdata = data.frame(data.test), type="vector")
    pred.t[which(rowSums(is.na(data.test[,1:n.feature]))>0)]=NA
    pred.list[,i]=pred.t
  }
  
  ###obtain predictions and errors
  true.err = sqrt(colMeans((pred.list-data.test[,n.feature+1])^2, na.rm=T))
  pred.list = pred.list + t(rep(1, n.best.tree) %*% t(colm.t))
  pred.err = (RMSE[order(RMSE, decreasing = F)[1:n.best.tree]])
  mu = rowMeans(pred.list)
  
  
  return(list(mu=mu, true.err=true.err, pred.err=pred.err))
  
}

