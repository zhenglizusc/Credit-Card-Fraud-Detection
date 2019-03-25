library(data.table)
library(dplyr)
library(tree)
library(dplyr)
library(stringr)
library(lubridate)
library('randomForest')
library (e1071)
library(ROCR)
library(leaps)
library(caret)
library(caTools)
library(tree)


# Data Preparation
## read data
load(file = "f.rda")
load(file="transaction.rda")
data=f
data1=df1

## merge two dataset
Fraud=ifelse(data1$Fraud==1,"Yes","No")
data2 =data.frame(data,Fraud)

## seperate the dataset into training , test and OOT
data3 = data2[1:84095,]
set.seed(1)
x=sample(1:84095,84095*0.7,replace=FALSE)
train=data3[x,]
test=data3[-x,]
OOT=data2[84095:96353,]
(96353-84095)/96353 ##12.7%


# Feature Selection/Dimensionality Reduction
## KS test
t=train%>%filter(Fraud=="Yes")
t1=train%>%filter(Fraud=="No")

variables=names(train%>%select(-Fraud))
ks=data.frame()

for (j in variables){
  kj=ks.test(t[,j],t1[,j],alternative="two.sided")
  print(kj)
  ks[j,'KS']=kj[["statistic"]][["D"]]
}

### select the top 20 variables with the highest score of Ks
ks
ks$name=rownames(ks)
k=(ks%>%arrange(-KS))[1:20,]
k

train2=train[,c("same_cardnum_fraud_30" ,"same_cardnum_fraud_14"  ,   "same_cardnum_fraud_7"  ,     
                "same_cardnum_fraud_3"     ,   "same_merchantdesc_fraud_30"     ,   "same_merchantdesc_fraud_14"       , 
                "same_merchantnum_fraud_14"  ,    "same_merchantdesc_fraud_7"   ,    "same_merchantnum_fraud_7"        ,
                "same_merchantnum_fraud_30" ,     "same_merchantnum_fraud_3"  ,     "same_merchantdesc_fraud_3"  , 
                "same_cardnum_fraud_1"  ,  "same_merchantnum_fraud_1" , "same_merchantdesc_fraud_1",   "same_merchantnum_fraud_1",
                "amount","same_cardnum_7"  ,"same_cardnum_3" ,"same_cardnum_14" ,"same_merchantdesc_3","Fraud")]
train2

## Forward Stepwise Selection
library(leaps)
regfit.fwd=regsubsets(Fraud~.,data =train2, nvmax=20,really.big=T, method="forward")
names(summary(regfit.fwd))
summary(regfit.fwd)$adjr2
which.max(summary(regfit.fwd)$adjr2)

### choose using Adjusted R square(adjr2)
a=coef(regfit.fwd, which.max(summary(regfit.fwd)$adjr2))
a=data.frame(a)
a

### We select 14 variables
train_apply=train2[,c("same_cardnum_fraud_30" ,"same_cardnum_fraud_14"  ,   "same_cardnum_fraud_7"  ,     
                      "same_cardnum_fraud_3"     ,   "same_merchantdesc_fraud_30"     ,   "same_merchantdesc_fraud_14"       , 
                      "same_merchantdesc_fraud_7"   ,  
                      "same_merchantnum_fraud_3"  ,     "same_merchantdesc_fraud_3"  , 
                      "same_cardnum_fraud_1"  ,  "same_merchantdesc_fraud_1",   
                      "same_cardnum_7"  ,"same_cardnum_3" ,"same_merchantdesc_3","Fraud")]

test_apply=test[,c("same_cardnum_fraud_30" ,"same_cardnum_fraud_14"  ,   "same_cardnum_fraud_7"  ,     
                   "same_cardnum_fraud_3"     ,   "same_merchantdesc_fraud_30"     ,   "same_merchantdesc_fraud_14"       , 
                   "same_merchantdesc_fraud_7"   ,  
                   "same_merchantnum_fraud_3"  ,     "same_merchantdesc_fraud_3"  , 
                   "same_cardnum_fraud_1"  ,  "same_merchantdesc_fraud_1",   
                   "same_cardnum_7"  ,"same_cardnum_3" ,"same_merchantdesc_3","Fraud")]

OOT_apply=OOT[,c("same_cardnum_fraud_30" ,"same_cardnum_fraud_14"  ,   "same_cardnum_fraud_7"  ,     
                 "same_cardnum_fraud_3"     ,   "same_merchantdesc_fraud_30"     ,   "same_merchantdesc_fraud_14"       , 
                 "same_merchantdesc_fraud_7"   ,  
                 "same_merchantnum_fraud_3"  ,     "same_merchantdesc_fraud_3"  , 
                 "same_cardnum_fraud_1"  ,  "same_merchantdesc_fraud_1",   
                 "same_cardnum_7"  ,"same_cardnum_3" ,"same_merchantdesc_3","Fraud")]


# Model Building
## 1. Decision Tree
### build decision tree model
tree.carseats =tree(Fraud~.,train_apply)

plot(tree.carseats)
text(tree.carseats,pretty = 0)

### confusion matrix for OOT dataset
tree.pred=predict(tree.carseats,OOT_apply,type="class")
table(tree.pred,OOT_apply$Fraud) 
## for OOT dataset
#tree.pred2    No   Yes
#No         11889   157
#Yes         32   181


## 2. Random Forest
### build random forest mondel
model_rf<-randomForest(Fraud~., data = train_apply)
model_rf

### confusion matrix for oot dataset
preds<-predict(model_rf,OOT_apply[,-15])
table(preds, OOT_apply$Fraud)
## preds    No   Yes
##   No  11899    63
##   Yes    22   275


## 3. Logistic Regression
glm.fit = glm(Fraud ~ ., data= train_apply, family='binomial')
glm.fit

### confusion matrix for OOT dataset
glm.prob = predict(glm.fit,OOT_apply, type = 'response') 
glm.predict = ifelse(glm.prob > 0.5, 'Yes', 'No')
table(glm.predict, OOT_apply$Fraud)
## glm.predict2    No   Yes
##           No 11870    87
##          Yes    51   251


## 4.SVM
### encode the response as a factor variable
train_apply$Fraud <- as.factor(train_apply$Fraud)

svmfit =svm(Fraud~., data=train_apply, probability=TRUE)

### confusion matrix for OOT data
pred_oot=predict(svmfit,OOT_apply,probability=TRUE)
oot_with_prob = attr(pred_oot, "probabilities")
confusionMatrix(pred_oot,OOT_apply$Fraud,positive = 'Yes')
## Prediction    No   Yes
##        No  11880    67
##        Yes    41   271

### calculate FDR of oot for top 2%
a = oot_with_prob[,2]
b = OOT_apply[,15]
b <- ifelse((b == "Yes"),1,0)
b = as.numeric(b)
combined_oot = data.frame(a,b)
combined_oot =combined_oot %>% 
  arrange(-a) 

sum(combined_oot[1:12259,]$b) #338
12259*0.02 #245
sum(combined_oot[1:245,]$b)/338
## 70.71%


