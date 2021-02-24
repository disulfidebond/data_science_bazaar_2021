library(plyr)
library(dplyr)
library(zoo)
library(MASS)
library(glmnet)
library(pROC)
library(caret)
library(ROCR)
library(xgboost)


`%notin%` <- Negate(`%in%`)
# import dataset
df = read.csv('COVID-19_Historical_Data_by_State.orig.csv')
df = df[order(df$DATE),]
row.names(df) <- NULL
# add new column HOSP_NEW as NA
df$HOSP_NEW = NA
# add HOSP_NEW values as difference between current day and prev day
df = df %>% mutate(HOSP_NEW = HOSP_YES - lag(HOSP_YES, default = HOSP_YES[1]))

# repeat for the values: HOSP_NO, HOSP_UNK, POS_FEM, POS_MALE
# then, create the label by scanning the next day to see if there is an increase in POS_NEW
df$SCAN = NA
df = df %>% mutate(SCAN = POS_NEW - lead(POS_NEW, default = POS_NEW))
View(df[which(colnames(df) %in% c('HOSP_NEW', 'HOSP_YES', 'SCAN', 'POS_NEW', 'INCREASE'))])
df$INCREASE = ifelse(df$SCAN < 0, 1.0, 0.0)
# select columns and rows, then you're ready to build the XGBoost model
df_colFiltered = df[which(names(df) %in% c("POS_NEW", "POS_7DAYAVG", "NEG_NEW", "NEG_7DAYAVG", "DTH_NEW", "DTH_7DAYAVG", "TEST_NEW", "TEST_7DAYAVG", "HOSP_NEW", "HOSP_NO_NEW", "HOSP_UNK_NEW", "POS_FEM_NEW", "POS_MALE_NEW", "INCREASE"))]
df_colFiltered = tail(df_colFiltered, -20)
df_colFiltered = head(df_colFiltered, -1)
row.names(df_colFiltered) <- NULL

df = df_colFiltered
df$INCREASE = ifelse(df$INCREASE == 1, "increasing", "decreasing")


df_split = sample.int(n=nrow(df), size=floor(.75*nrow(df)), replace=FALSE)
df_training = df[df_split,]
df_testing = df[-df_split,]


cvCtrl <- trainControl(method = "repeatedcv", number = 5, repeats = 1, classProbs = TRUE, summaryFunction = twoClassSummary, allowParallel = TRUE)

xgbGrid<-expand.grid(nrounds=c(500, 750), max_depth=c(2, 5), eta=c(0.001, 0.01), gamma = c(0), colsample_bytree = c(1), min_child_weight = c(2, 4), subsample = c(0.5))
xgbGridtrain<- train(INCREASE~., method = "xgbTree", data = df_training, metric = "ROC", tree_method = "auto",trControl = cvCtrl, tuneGrid = xgbGrid, na.action = na.pass)
xgbGridProbs<-predict(xgbGridtrain, newdata=df_testing, type="prob", na.action = na.pass)
xgbGridProbsDF<-as.data.frame(xgbGridProbs)

xgbGridRoc <-roc(df_testing$INCREASE, round(xgbGridProbsDF$increasing, digits=4))

print('baseline AUC is')
print(xgbGridRoc)
