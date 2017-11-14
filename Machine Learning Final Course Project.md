---
title: "Machine Learning Course Project Final"
output: html_document
---

```{r setup, include=FALSE}
knitr::opts_chunk$set(echo = TRUE)
```

#### Introduction and Background

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement - a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website here: http://groupware.les.inf.puc-rio.br/har (see the section on the Weight Lifting Exercise Dataset).

#### Goal

The goal of your project is to predict the manner in which they did the exercise. This is the "classe" variable in the training set. You may use any of the other variables to predict with. You should create a report describing how you built your model, how you used cross validation, what you think the expected out of sample error is, and why you made the choices you did. You will also use your prediction model to predict 20 different test cases.

#### Data Processing

#### Training Dataset Link

The training data for this project are available here:
https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv

#### Testing Dataset Link

https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv

#### Data Loading and Cleaning
First load all the packages that we would need to analyse the data and to build a model to predict the test data.

```{r, echo = TRUE}
library(caret)
library(rpart)
library(randomForest)
library(rpart.plot)
library(rattle)
library(gbm)
library(survival)
set.seed(12345)
setwd("C:/Users/nvarshney/Desktop/R programming/Coursera/Git Hub and Programming assignments/Course 8/Course Project")
# Load and read the training and test data into R using, read.csv function
training  <- read.csv("pml-training.csv", na.strings = c("NA","#DIV/0!",""))
# Keep this testing data aside for prediction
testing20<- read.csv("pml-testing.csv", na.strings = c("NA","DIV/0!",""))
dim(training)
dim(testing20)
str(training)
# As we can see in this dataset there are so many NAs value. First I will remove all the NAs from the data and willcheck for near zero variance column.
```

#### Training Set
Clean variables with more than 75% NAs

```{r, echo = TRUE}
mytrain_NA <- training
for (i in 1:length(training)) {
  if (sum(is.na(training[ , i])) / nrow(training) >= .75) {
    for (j in 1:length(mytrain_NA)) {
      if (length(grep(names(training[i]), names(mytrain_NA)[j]))==1) {
        mytrain_NA <- mytrain_NA[ , -j]
      }
    }
  }
}

dim(mytrain_NA)
names(mytrain_NA)
```

#### Remove first few columns they don't seem like a predictor.

```{r, echo = TRUE}
mytrain_NA <- mytrain_NA[,-c(1:7)]
str(mytrain_NA)
# Changing name to original training data
training <- mytrain_NA
```

#### Remove variables with near zero variance.

```{r,echo = TRUE}
nzv <- nearZeroVar(training, saveMetrics = TRUE)
nzv
# There is no variables near to zero variance, all false none to remove.
dim(training)

```

#### Data spiliting
 
In order to get out of sample errors, we will spilt the training dataset into training(70%) and testing(30%). I will use cross validation within the training set to improve the modelfit and then do an out-of-sample test with the testing data sets. 

```{r, echo = TRUE}
inTrain <- createDataPartition(training$classe, p = .7, list = FALSE)
training <- training[inTrain,]
testing <- training[-inTrain,]
dim(training)
str(training)
dim(testing)
```

In training dataset we have 13737 observations and 53 variables, and in testing dataset 4110 oservations and 53 variables.

#### Finally Clean the testing20 set, that we will be using  to predict 20 test cases.

 Remove all the NA variables from thes testing20 data and check for if we have any zero variance variable in the data and  get the same numbers of variables as we have in the model.

```{r, echo = TRUE}
mytest_NA <- testing20
for (i in 1:length(testing20)) {
  if (sum(is.na(testing20[ , i])) / nrow(testing20) >= .75) {
    for (j in 1:length(mytest_NA)) {
      if (length(grep(names(testing20[i]), names(mytest_NA)[j]))==1) {
        mytest_NA <- mytest_NA[ , -j]
      }
    }
  }
}
dim(mytest_NA)
# Remove first few columns from the testing20data,they don't look like a predictor.
mytest_NA <- mytest_NA[,-c(1:7)]
# change name to original name
testing20 <- mytest_NA
# To check  in the test data if we have any variables to near zero variance. All are False, so no Variables to remove
nzv_test<- nearZeroVar(testing20, saveMetrics = TRUE)
nzv_test
dim(testing20)
str(testing20)
View(testing20)
```

#### Using Machine Learning Algorithim to build  different models for prediction.

We will use Random Forest, Decision Trees, and Generalized Boosted Regression Model and then find out which alogorithim provides the best out-of-sample accuracy.

#### Prediction with Random Forest

```{r}
set.seed(12345)
modfit1 <- train(classe~.,data = training, method = "rf", preProcess = c("center", "scale"), trControl=trainControl(method = "cv", number = 4))
modfit1
# Cross Validation on my testing data
predR <- predict(modfit1,newdata = testing)
RF <-confusionMatrix(predR, testing$classe)
RF$overall["Accuracy"]
```

#### Prediction with Decision Tree 

```{r, echo = TRUE}
set.seed(12345)
modfit2 <- rpart(classe ~., data = training, method = "class")
modfit2
fancyRpartPlot(modfit2)
# Cross Validation on my testing data
predD <- predict(modfit2, testing, type = "class")
DT <-confusionMatrix(testing$classe, predD)
DT
DT$overall["Accuracy"]
```

#### Prediction with Generalized Boosted Regression

```{r, echo = TRUE}
set.seed(12345)
modfit3 <- train(classe ~., data = training, method = "gbm",verbose =  FALSE,trControl=trainControl(method = "cv", number = 4))
modfit3$finalModel
# Cross Validation on my testing data
predG <- predict(modfit3, testing)
GBM <- confusionMatrix(testing$classe, predG)
GBM$overall["Accuracy"]
```

#### Model Assessment

```{r, echo = TRUE}
AccuracyResults <- data.frame(
Model = c('RF','CART','GBM'),
Accuracy = c(RF$overall[1],DT$overall[1], GBM$overall[1]))
print(AccuracyResults)
```

From the above results we can say that Random Forest is way better than classification tree method but close to generalized boosting method. In random forest accuracy is 100% and out-of-sample error is 0. This may be due to the fact that many predictors are highly corelated to each other.

#### Final Prediction on the test data with 20 cases

Now we will use the random forest model to predict the outcome variable Classe for the testing dataset.

```{r, echo = TRUE}
pred <-predict(modfit1, testing20)
pred

```

#### Generating files to submit as answer for the assignment

```{r,echo=TRUE}
pml_write_files = function(x){
  n = length(x)
  for(i in 1:n){
    filename = paste0("problem_id_",i,".txt")
    write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
  }
}

pml_write_files(pred)
```

####Out of Sample Error

```{r, echo = TRUE}
#Find out how much error we have in Random Forest Model
error <- 1-RF$overll["Accuracy"]
error
```

####Conclusion

From the resulting table we can see the Random Forest algorithm yields a better result.

####Appendices

Just checking the Variable Imporatance in the final Random Forest Model

```{r, echo = TRUE}
mod_imp <- varImp(modfit1)
mod_imp
```