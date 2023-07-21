##########################################################
# Install and Load all needed packages
##########################################################
if(!require(randomForest)) install.packages("randomForest")
if(!require(reldist)) install.packages("reldist")
if(!require(readxl)) install.packages("readxl")
if(!require(dplyr)) install.packages("dplyr")
if(!require(tidyr)) install.packages("tidyr")
if(!require(dslabs)) install.packages("dslabs")
if(!require(stringr)) install.packages("stringr")
if(!require(forcats)) install.packages("forcats")
if(!require(ggplot2)) install.packages("ggplot2")
if(!require(caTools)) install.packages("caTools")
if(!require(rpart.plot)) install.packages("rpart.plot")
if(!require(ISLR)) install.packages("ISLR")
if(!require(e1071)) install.packages("e1071")
if(!require(OneR)) install.packages("OneR")
if(!require(tidyverse)) install.packages("tidyverse", 
                                         repos = "http://cran.us.r-project.org")
if(!require(ggthemes)) install.packages("ggthemes", 
                                        repos="http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", 
                                     repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", 
                                          repos = "http://cran.us.r-project.org")
if(!require(rpart)) install.packages("rpart", 
                                     repos="http://cran.us.r-project.org")
if(!require(MLmetrics)) install.packages("MLmetrics", 
                                         repos="http://cran.us.r-project.org")
if(!require(haven)) install.packages("haven", 
                                     repos="http://cran.us.r-project.org")
library(randomForest)
library(reldist)
library(readxl)
library(dplyr)
library(tidyr)
library(dslabs)
library(stringr)
library(forcats)
library(ggplot2)
library(tidyverse)
library(OneR)
library(caret)
library(data.table)
library(ggthemes)
library(caTools)
library(rpart)
library(ISLR)
library(e1071)
library(MLmetrics)
library(haven)
library(hrbrthemes)
library(viridis)
library(rpart.plot)

##########################################################
# Data Loading
##########################################################
set.seed(1, sample.kind = "Rounding")
trainFileName = "adult.data"; testFileName = "adult.test"

if (!file.exists (trainFileName))
  download.file (
    url = "http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data", 
    destfile = trainFileName)

if (!file.exists (testFileName))
  download.file (
    url = "http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test",
    destfile = testFileName)
colNames = c ("age", "workclass", "fnlwgt", "education", 
              "educationnum", "maritalstatus", "occupation",
              "relationship", "race", "sex", "capitalgain",
              "capitalloss", "hoursperweek", "nativecountry",
              "incomelevel")

adult = read.table (trainFileName, header = FALSE, sep = ",",
                    strip.white = TRUE, col.names = colNames,
                    na.strings = "?", stringsAsFactors = TRUE)
##########################################################
# DATA EXPLORATION
##########################################################
str(adult) # We look the variables/columns and see their type of variable
na_v <- sapply(adult, function(x) sum(is.na(x))) # Create a function that shows the NA's of the database
na_v # Print the results
pna_v <- sapply(adult, function(adult){sum(is.na(adult)==T)*100/length(adult)}) # Create a function that shows the percentage of NA's of the database
round(pna_v, digits = 3) # Print the results

##########################################################
# DATA EXPLORATION
##########################################################
# And for the modeling, we remove all the NA's of the database
adult = adult[!is.na(adult$workclass) & !is.na(adult$occupation),]
adult = adult[!is.na(adult$nativecountry),]
adult$fnlwgt = NULL

##########################################################
# TRAIN_SET AND TEST_SET
##########################################################
# Sample, train and test sets for the models
sample.adult <- sample.split(adult$incomelevel, SplitRatio = 0.80)
train_set = subset(adult, sample.adult == TRUE)
test_set = subset(adult, sample.adult == FALSE)

##########################################################
# MODELING
##########################################################

#######################
# SVM (Support Vector Machine)
#######################
svm.adult = svm(incomelevel ~
                  age+education+occupation+race+sex,
                data = train_set)
# Prediction of data and Confusion Matrix
test_set$pred.value = predict(svm.adult, newdata=test_set, type="response")
model1 <- table(test_set$income, test_set$pred.value)
confusionMatrix(model1) # We see the results
F1_Score(test_set$income, test_set$pred.value) # And evaluate the model with the F1 Score

# And we add all the results to a database
results<- data.frame(
  Model="SVM (Support Vector Machine)",
  Accuracy= Accuracy(test_set$income, test_set$pred.value),
  F1Score= F1_Score(test_set$income, test_set$pred.value),
  Sensitivity= sensitivity(test_set$income, test_set$pred.value),
  Specificity= specificity(test_set$income, test_set$pred.value))
results # Print the table of the restults

#######################
# DECISION TREE
#######################
# Applying Decision Tree Model
detree <- rpart(incomelevel ~
                  age+education+occupation+race+sex,
                data = train_set)
# Prediction of data and Confusion Matrix
test_set$pred.value2 = predict(detree, newdata=test_set, type="class")
model2 <- table(test_set$income, test_set$pred.value2)
confusionMatrix(model2) # We see the results
F1_Score(test_set$income, test_set$pred.value2) # And evaluate the model with the F1 Score

# And we add all the results to a database
results<- bind_rows(
  results, data.frame(Model="Decision Tree",
                      Accuracy=Accuracy(test_set$income, test_set$pred.value2),
                      F1Score=F1_Score(test_set$income, test_set$pred.value2),
                      Sensitivity=sensitivity(test_set$income, test_set$pred.value2),
                      Specificity =specificity(test_set$income, test_set$pred.value2)))
results # Print the table of the restults

#######################
# RANDOM FOREST
#######################
set.seed(4543) # this is for reproducibility
# Applying Random Forest Model
rfmodel <- randomForest(incomelevel ~ 
                          age+education+occupation+race+sex,
                        data = train_set, importance = TRUE)
# Prediction of data and Confusion Matrix
test_set$pred.value3 = predict(rfmodel, newdata=test_set)
model3 <- table(test_set$income, test_set$pred.value3)
confusionMatrix(model3) # We see the results
F1_Score(test_set$income, test_set$pred.value3) # And evaluate the model with the F1 Score

# And we add all the results to a database
results<- bind_rows(
  results, data.frame(Model="Random Forest", 
                      Accuracy=Accuracy(test_set$income, test_set$pred.value3),
                      F1Score=F1_Score(test_set$income, test_set$pred.value3),
                      Sensitivity=sensitivity(test_set$income, test_set$pred.value3),
                      Specificity =specificity(test_set$income, test_set$pred.value3)))
results # Print the table of the restults

##########################################################
# RESULTS
##########################################################
results # Print the table of the restults



























