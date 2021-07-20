path <- "C:/Users/Shikha/Downloads/train"
setwd(path)

library(data.table)
train <- fread("train.csv",na.strings = c(""," ","?","NA",NA))
test <- fread("test.csv",na.strings = c(""," ","?","NA",NA))

View(train)
View(test)

unique(train$income_level)

train[,income_level := ifelse(income_level == "-50000",0,1)]
test[,income_level := ifelse(income_level == "-50000",0,1)]

round(prop.table(table(train$income_level))*100)

factcols <- c(2:5,7,8:16,20:29,31:38,40,41)
numcols <- setdiff(1:40,factcols)

train[,(factcols) := lapply(.SD, factor), .SDcols = factcols]
train[,(numcols) := lapply(.SD, as.numeric), .SDcols = numcols]

test[,(factcols) := lapply(.SD, factor), .SDcols = factcols]
test[,(numcols) := lapply(.SD, as.numeric), .SDcols = numcols]

cat_train <- train[,factcols, with=FALSE]
cat_test <- test[,factcols,with=FALSE]


num_train <- train[,numcols,with=FALSE]
num_test <- test[,numcols,with=FALSE]
rm(train,test)

library(ggplot2)
library(plotly)

tr <- function(a){
            ggplot(data = num_train, aes(x= a, y=..density..)) + geom_histogram(fill="blue",color="red",alpha = 0.5,bins =100) + geom_density()
 ggplotly()
}
tr(num_train$capital_losses)
tr(num_train$age)

ggplot(data=num_train,aes(x = age, y=wage_per_hour))+geom_point(aes(colour=income_level))+scale_y_continuous("wage per hour", breaks = seq(0,10000,1000))

all_bar <- function(i){
 ggplot(cat_train,aes(x=i,fill=income_level))+geom_bar(position = "dodge",  color="black")+scale_fill_brewer(palette = "Pastel1")+theme(axis.text.x =element_text(angle  = 60,hjust = 1,size=10))
}

all_bar(cat_train$class_of_worker)

all_bar(cat_train$education)

table(is.na(num_train))
table(is.na(num_test))

library(caret)

ax <-findCorrelation(x = cor(num_train), cutoff = 0.7)

num_train <- num_train[,-ax,with=FALSE] 
num_test[,weeks_worked_in_year := NULL]

mvtr <- sapply(cat_train, function(x){sum(is.na(x))/length(x)})*100
mvte <- sapply(cat_test, function(x){sum(is.na(x)/length(x))}*100)
mvtr
mvte

cat_train <- subset(cat_train, select = mvtr < 5 )
cat_test <- subset(cat_test, select = mvte < 5)

cat_train <- cat_train[,names(cat_train) := lapply(.SD, as.character),.SDcols = names(cat_train)]
for (i in seq_along(cat_train)) set(cat_train, i=which(is.na(cat_train[[i]])), j=i, value="Unavailable")
cat_train <- cat_train[, names(cat_train) := lapply(.SD,factor), .SDcols = names(cat_train)]

cat_test <- cat_test[, (names(cat_test)) := lapply(.SD, as.character), .SDcols = names(cat_test)]
for (i in seq_along(cat_test)) set(cat_test, i=which(is.na(cat_test[[i]])), j=i, value="Unavailable")
cat_test <- cat_test[, (names(cat_test)) := lapply(.SD, factor), .SDcols = names(cat_test)]

for(i in names(cat_train)){
                  p <- 5/100
                  ld <- names(which(prop.table(table(cat_train[[i]])) < p))
                  levels(cat_train[[i]])[levels(cat_train[[i]]) %in% ld] <- "Other"
}

for(i in names(cat_test)){
                  p <- 5/100
                  ld <- names(which(prop.table(table(cat_test[[i]])) < p))
                  levels(cat_test[[i]])[levels(cat_test[[i]]) %in% ld] <- "Other"
}

library(mlr)
summarizeColumns(cat_train)[,"nlevs"]
summarizeColumns(cat_test)[,"nlevs"]

num_train[,.N,age][order(age)]
num_train[,.N,wage_per_hour][order(-N)]

num_train[,age:= cut(x = age,breaks = c(0,30,60,90),include.lowest = TRUE,labels = c("young","adult","old"))]
num_train[,age := factor(age)]

num_test[,age:= cut(x = age,breaks = c(0,30,60,90),include.lowest = TRUE,labels = c("young","adult","old"))]
num_test[,age := factor(age)]

num_train[,wage_per_hour := ifelse(wage_per_hour == 0,"Zero","MoreThanZero")][,wage_per_hour := as.factor(wage_per_hour)]
num_train[,capital_gains := ifelse(capital_gains == 0,"Zero","MoreThanZero")][,capital_gains := as.factor(capital_gains)]
num_train[,capital_losses := ifelse(capital_losses == 0,"Zero","MoreThanZero")][,capital_losses := as.factor(capital_losses)]
num_train[,dividend_from_Stocks := ifelse(dividend_from_Stocks == 0,"Zero","MoreThanZero")][,dividend_from_Stocks := as.factor(dividend_from_Stocks)]

num_test[,wage_per_hour := ifelse(wage_per_hour == 0,"Zero","MoreThanZero")][,wage_per_hour := as.factor(wage_per_hour)]
num_test[,capital_gains := ifelse(capital_gains == 0,"Zero","MoreThanZero")][,capital_gains := as.factor(capital_gains)]
num_test[,capital_losses := ifelse(capital_losses == 0,"Zero","MoreThanZero")][,capital_losses := as.factor(capital_losses)]
num_test[,dividend_from_Stocks := ifelse(dividend_from_Stocks == 0,"Zero","MoreThanZero")][,dividend_from_Stocks := as.factor(dividend_from_Stocks)]

num_train[,income_level := NULL]

d_train <- cbind(num_train,cat_train)
d_test <- cbind(num_test,cat_test)

rm(num_train,num_test,cat_train,cat_test) #save memory

library(mlr)

train.task <- makeClassifTask(data = d_train,target = "income_level")
test.task <- makeClassifTask(data=d_test,target = "income_level")

train.task <- removeConstantFeatures(train.task)
test.task <- removeConstantFeatures(test.task)

var_imp <- generateFilterValuesData(train.task, method = c("information.gain"))
plotFilterValues(var_imp,feat.type.cols = TRUE)

train.under <- undersample(train.task,rate = 0.1) #keep only 10% of majority class
table(getTaskTargets(train.under))

train.over <- oversample(train.task,rate=15) #make minority class 15 times
table(getTaskTargets(train.over))

train.smote <- smote(train.task,rate = 15,nn = 5)

system.time(
    train.smote <- smote(train.task,rate = 10,nn = 3) 
   )

table(getTaskTargets(train.smote))

listLearners("classif","twoclass")[c("class","package")]

naive_learner <- makeLearner("classif.naiveBayes",predict.type = "response")
naive_learner$par.vals <- list(laplace = 1)

folds <- makeResampleDesc("CV",iters=10,stratify = TRUE)

fun_cv <- function(a){
     crv_val <- resample(naive_learner,a,folds,measures = list(acc,tpr,tnr,fpr,fp,fn))
     crv_val$aggr
}

fun_cv (train.task)

fun_cv(train.under)
fun_cv(train.over)
fun_cv(train.smote)

nB_model <- train(naive_learner, train.smote)
nB_predict <- predict(nB_model,test.task)

nB_prediction <- nB_predict$data$response
dCM <- confusionMatrix(d_test$income_level,nB_prediction)
precision <- dCM$byClass['Pos Pred Value']
recall <- dCM$byClass['Sensitivity']

f_measure <- 2*((precision*recall)/(precision+recall))
f_measure 

set.seed(2002)
xgb_learner <- makeLearner("classif.xgboost",predict.type = "response")
xgb_learner$par.vals <- list(
                      objective = "binary:logistic",
                      eval_metric = "error",
                      nrounds = 150,
                      print.every.n = 50
)

xg_ps <- makeParamSet( 
                makeIntegerParam("max_depth",lower=3,upper=10),
                makeNumericParam("lambda",lower=0.05,upper=0.5),
                makeNumericParam("eta", lower = 0.01, upper = 0.5),
                makeNumericParam("subsample", lower = 0.50, upper = 1),
                makeNumericParam("min_child_weight",lower=2,upper=10),
                makeNumericParam("colsample_bytree",lower = 0.50,upper = 0.80)
)

rancontrol <- makeTuneControlRandom(maxit = 5L) #do 5 iterations

set_cv <- makeResampleDesc("CV",iters = 5L,stratify = TRUE)

xgb_tune <- tuneParams(learner = xgb_learner, task = train.task, resampling = set_cv, measures = list(acc,tpr,tnr,fpr,fp,fn), par.set = xg_ps, control = rancontrol)

xgb_new <- setHyperPars(learner = xgb_learner, par.vals = xgb_tune$x)

xgmodel <- train(xgb_new, train.task)

predict.xg <- predict(xgmodel, test.task)

xg_prediction <- predict.xg$data$response

xg_confused <- confusionMatrix(d_test$income_level,xg_prediction)
Accuracy : 0.948
Sensitivity : 0.9574
Specificity : 0.6585

precision <- xg_confused$byClass['Pos Pred Value']
recall <- xg_confused$byClass['Sensitivity']

f_measure <- 2*((precision*recall)/(precision+recall))
f_measure

filtered.data <- filterFeatures(train.task,method = "information.gain",abs = 20)
xgb_boost <- train(xgb_new,filtered.data)

predict.xg$threshold

xgb_prob <- setPredictType(learner = xgb_new,predict.type = "prob")

xgmodel_prob <- train(xgb_prob,train.task)
predict.xgprob <- predict(xgmodel_prob,test.task)

df <- generateThreshVsPerfData(predict.xgprob,measures = list(fpr,tpr))
plotROCCurves(df)

pred2 <- setThreshold(predict.xgprob,0.4)

confusionMatrix(d_test$income_level,pred2$data$response)

pred3 <- setThreshold(predict.xgprob,0.30)
confusionMatrix(d_test$income_level,pred3$data$response)

getParamSet("classif.svm")
svm_learner <- makeLearner("classif.svm",predict.type = "response")
svm_learner$par.vals<- list(class.weights = c("0"=1,"1"=10),kernel="radial")

svm_param <- makeParamSet(
            makeIntegerParam("cost",lower = 10^-1,upper = 10^2), 
            makeIntegerParam("gamma",lower= 0.5,upper = 2)
)

set_search <- makeTuneControlRandom(maxit = 5L) #5 times

set_cv <- makeResampleDesc("CV",iters=5L,stratify = TRUE)

svm_tune <- tuneParams(learner = svm_learner,task = train.task,measures = list(acc,tpr,tnr,fpr,fp,fn), par.set = svm_param,control = set_search,resampling = set_cv)
)

svm_new <- setHyperPars(learner = svm_learner, par.vals = svm_tune$x)

svm_model <- train(svm_new,train.task)
predict_svm <- predict(svm_model,test.task)

confusionMatrix(d_test$income_level,predict_svm$data$response)