---
title: 'IST 707 HW4: Regression and Artificial Neural Network/Deep Learning '  
author: "Sanjeev Ramasamy Seenivasagamani"  
date: "April 27, 2019"  
output:  
  word_document: default  
  pdf_document: default  
  html_document: default  
---  

Load the required libraries:  

```{r message=FALSE, warning=FALSE}
require(foreign)
require(ggplot2)
require(MASS)
require(Hmisc)
require(reshape2)
library(rms)
library(data.table)
library(tidyverse)
library(tidytext)
library(caret)
library(rpart)
library("rpart.plot")
library(recipes)
library(randomForest)
library(keras)
```

## Introduction

### About:  

This is an educational data set which is collected from learning management system (LMS) called Kalboard 360. Kalboard 360 is a multi-agent LMS, which has been designed to facilitate learning through the use of leading-edge technology. Such system provides users with a synchronous access to educational resources from any device with Internet connection.  

The dataset consists of 305 males and 175 females. The students come from different origins such as 179 students are from Kuwait, 172 students are from Jordan, 28 students from Palestine, 22 students are from Iraq, 17 students from Lebanon, 12 students from Tunis, 11 students from Saudi Arabia, 9 students from Egypt, 7 students from Syria, 6 students from USA, Iran and Libya, 4 students from Morocco and one student from Venezuela.  

This dataset includes also a new category of features; this feature is parent parturition in the educational process. Parent participation feature have two sub features: Parent Answering Survey and Parent School Satisfaction. There are 270 of the parents answered survey and 210 are not, 292 of the parents are satisfied from the school and 188 are not.  

#### The students are classified into three numerical intervals based on their total grade/mark:  

Low-Level: interval includes values from 0 to 69,  

Middle-Level: interval includes values from 70 to 89,  

High-Level: interval includes values from 90-100.  


### Data load:  

```{r}
data <- read.csv("Students Academic Performance.csv", header = TRUE)
str(data)
```


## Exploratory Data Analysis  

Let us have a look at the data distribution through various plots.   

### Studentcount vs Birthplace  

```{r}
ggplot(data = data, aes(x = PlaceofBirth)) + geom_bar(aes(fill = NationalITy)) + 
    labs(x = "Birth Place", y = "Student Count") + coord_flip()
```

### Studentcount vs Gender  

```{r}
ggplot(data = data, aes(x = Class, fill = gender)) + geom_bar() + 
    labs(x = "Gender", y = "Student Count") + coord_flip()
```


### Studentcount vs GradeID  

```{r}
ggplot(data = data, aes(x = GradeID, fill = gender)) + geom_bar() + 
    labs(x = "Grade ID", y = "Student Count") + coord_flip()
```

### Studentcount vs Section ID  

```{r}
ggplot(data = data, aes(x = SectionID, fill = Topic)) + geom_bar() +
    labs(x = "Section ID", y = "Student Count") +
    coord_flip()
```

### Topic vs Class  

```{r}
ggplot(data = data, aes(x = Class, fill = Topic)) + geom_bar() +
    labs(x = "Class", y = "Topic") +
    coord_flip()
```

### Gender vs Topic  

```{r}
ggplot(data = data, aes(x = Topic, fill = gender)) + geom_bar() +
    labs(x = "Topic", y = "Student Count") +
    scale_y_continuous(breaks = seq(0,100,4)) + coord_flip()
```

### Topic vs Stage ID  
```{r}
ggplot(data = data, aes(x = Topic, fill = StageID)) + geom_bar() +
    labs(x = "Topic", y = "Stage ID") + coord_flip() +
    scale_y_continuous(breaks = seq(0,100,4))
```

## Topic vs SectionID   
```{r}
ggplot(data = data, aes(x = Topic, fill = SectionID)) + geom_bar() +
    labs(x = "Topic", y = "Section ID") + coord_flip() +
    scale_y_continuous(breaks = seq(0,100,4))
```

### Topic vs Student Count  

```{r}
ggplot(data = data, aes(x = Topic, fill = Class)) + geom_bar(position = "fill") +
    labs(x = "Topic", y = "Student Count") + coord_flip() +
    scale_y_continuous(breaks = seq(0,100,4))
```

### AnnouncementViews vs VisitedResources  
```{r}
ggplot(data = data, aes( x = VisITedResources, y = AnnouncementsView)) + geom_point() +
    geom_smooth(method = "lm")
```


## Ordinal Logistic Regression

### Data Manipulation:  

As you can see that some of the categorical variables have more than 9 levels. This can cause issues while training the Ordinal Regression Model. So, let us try to convert the ordinal data to numbered ranking.  

```{r}
#Converting the variables to character prior to ranking them
data$NationalITy<- as.character(data$NationalITy)
data$PlaceofBirth<- as.character(data$PlaceofBirth)
data$GradeID<- as.character(data$GradeID)
data$Topic<- as.character(data$Topic)

#Nationality
data$NationalITy[data$NationalITy == "Egypt"] <- 1
data$NationalITy[data$NationalITy == "Tunis"] <- 1
data$NationalITy[data$NationalITy == "Lybia"] <- 1
data$NationalITy[data$NationalITy == "Iran"] <- 2
data$NationalITy[data$NationalITy == "Iraq"] <- 2
data$NationalITy[data$NationalITy == "Jordan"] <- 2
data$NationalITy[data$NationalITy == "KW"] <- 2
data$NationalITy[data$NationalITy == "lebanon"] <- 2
data$NationalITy[data$NationalITy == "SaudiArabia"] <- 2
data$NationalITy[data$NationalITy == "Syria"] <- 2
data$NationalITy[data$NationalITy == "Palestine"] <- 2
data$NationalITy[data$NationalITy == "Morocco"] <- 3
data$NationalITy[data$NationalITy == "venzuela"] <- 4
data$NationalITy[data$NationalITy == "USA"] <- 5

#Place of Birth
data$PlaceofBirth[data$PlaceofBirth == "Egypt"] <- 1
data$PlaceofBirth[data$PlaceofBirth == "Tunis"] <- 1
data$PlaceofBirth[data$PlaceofBirth == "Lybia"] <- 1
data$PlaceofBirth[data$PlaceofBirth == "Iran"] <- 2
data$PlaceofBirth[data$PlaceofBirth == "Iraq"] <- 2
data$PlaceofBirth[data$PlaceofBirth == "Jordan"] <- 2
data$PlaceofBirth[data$PlaceofBirth == "KuwaIT"] <- 2
data$PlaceofBirth[data$PlaceofBirth == "lebanon"] <- 2
data$PlaceofBirth[data$PlaceofBirth == "SaudiArabia"] <- 2
data$PlaceofBirth[data$PlaceofBirth == "Syria"] <- 2
data$PlaceofBirth[data$PlaceofBirth == "Palestine"] <- 2
data$PlaceofBirth[data$PlaceofBirth == "Morocco"] <- 3
data$PlaceofBirth[data$PlaceofBirth == "venzuela"] <- 4
data$PlaceofBirth[data$PlaceofBirth == "USA"] <- 5

#Grade ID
data$GradeID[data$GradeID == "G-02"] <- 2
data$GradeID[data$GradeID == "G-04"] <- 4
data$GradeID[data$GradeID == "G-05"] <- 5
data$GradeID[data$GradeID == "G-06"] <- 6
data$GradeID[data$GradeID == "G-07"] <- 7
data$GradeID[data$GradeID == "G-08"] <- 8
data$GradeID[data$GradeID == "G-09"] <- 9
data$GradeID[data$GradeID == "G-10"] <- 10
data$GradeID[data$GradeID == "G-11"] <- 11
data$GradeID[data$GradeID == "G-12"] <- 12

#Subject Topic
data$Topic[data$Topic == "Arabic"] <- 1
data$Topic[data$Topic == "Biology"] <- 2
data$Topic[data$Topic == "Chemistry"] <- 3
data$Topic[data$Topic == "English"] <- 4
data$Topic[data$Topic == "French"] <- 5
data$Topic[data$Topic == "Geology"] <- 6
data$Topic[data$Topic == "History"] <- 7
data$Topic[data$Topic == "IT"] <- 8
data$Topic[data$Topic == "Math"] <- 9
data$Topic[data$Topic == "Quran"] <- 10
data$Topic[data$Topic == "Science"] <- 11
data$Topic[data$Topic == "Spanish"] <- 12

#Converting the variables back to integer instead of factors
data$NationalITy<- as.integer(data$NationalITy)
data$PlaceofBirth<- as.integer(data$PlaceofBirth)
data$GradeID<- as.integer(data$GradeID)
data$Topic<- as.integer(data$Topic)
```


### Splitting the data into train and test dataset:  

```{r }
train_index <- createDataPartition(data$Class, p = 0.6, list = FALSE)
train <- data[train_index, ]
test <- data[-train_index, ]
```

### Training the model:  

```{r }
m <- polr(Class ~., data = train, Hess=TRUE)
summary(m)
```

We see the usual regression output coefficient table including the value of each coefficient, standard errors, t values, estimates for the two intercepts, residual deviance and AIC. AIC is the information criteria. Lesser the better.  

Now we’ll calculate some essential metrics such as p-Value, CI, Odds ratio:  

```{r message=FALSE, warning=FALSE}
ctable <- coef(summary(m))
```

Calculating the p-value from the t-value:  
```{r}
p <- pnorm(abs(ctable[, "t value"]), lower.tail = FALSE) * 2
ctable <- cbind(ctable, "p value" = p)
ctable
```

Confidence Intervals:  
```{r}
ci <- confint(m)
```

```{r}
exp(coef(m))

## OR and CI
exp(cbind(OR = coef(m), ci))
```

### Interpretation:  

Since we have many independent variables we will be having a look at just three of them for interpretation purpose.  

1. The gender of the student being Male increases the odds of the student being in Low or Middle or High level grades combined by 1.326 than Female  

2. When the nnumber of raised hands go up by 1 unit, the odds of moving from Low level to Middle or High levels are multiplied by 0.987  

3. For every parent that views the announcement the odds of the student moving from the Low scoring bracket to Medium or High level increases by 1.008  


Let us now go ahead and predict the Class variable using the above model  
```{r}
predictedClass <- predict(m, test)  # predict the classes directly
head(predictedClass)
```

Confusion Matrix:  
```{r}
table(test$Class, predictedClass)
```

Miscassification Error:  
```{r}
mean(as.character(test$Class) != as.character(predictedClass))
```

## ARTIFICIAL NEURAL NETWORK  

As we have manipulated the data for performing Ordinal Logistic Regression, let us now read the data again and perform train-test split for the Multilayer Perceptron modelling.  

```{r}
data <- read.csv("Students Academic Performance.csv", header = TRUE)
train_index <- createDataPartition(data$Class, p = 0.6, list = FALSE)
train <- data[train_index, ]
test <- data[-train_index, ]
```

### Data Preprocessing:  

```{r}
rec_obj <- recipe(Class ~ ., data = train) %>%
  step_dummy(all_nominal(), -all_outcomes()) %>%
   step_center(all_predictors(), -all_outcomes()) %>% 
  step_scale(all_predictors(), -all_outcomes()) %>% 
  prep(data = train)
rec_obj
```

Splitting the dependent and the independent variables in the training and testing datasets
```{r}
x_train_tbl <- as.matrix(bake(rec_obj, new_data = train) %>% select(-Class))
x_test_tbl <- as.matrix(bake(rec_obj, new_data = test) %>% select(-Class))
y_train_vec <- ifelse(pull(train, Class) == "H", 2,ifelse(pull(train, Class) == "M", 1,0))
y_test_vec <-  ifelse(pull(test, Class) == "H", 2,ifelse(pull(test, Class) == "M", 1,0))
str(x_train_tbl)
dim(x_train_tbl)
```

```{r message=FALSE, warning=FALSE}
library(keras)
use_python("/usr/local/bin/python3")
model_keras <- keras_model_sequential()
```

```{r}
model_keras %>%
  layer_flatten(input_shape = ncol(rec_obj)) %>%
  layer_dense(units = 128, activation = 'relu') %>%
  layer_dense(units = 10, activation = 'softmax')
```


```{r}
model_keras %>% compile(
  optimizer = 'adam', 
  loss = 'sparse_categorical_crossentropy',
  metrics = c('accuracy')
)
```

```{r}
set.seed(100)
fit <- model_keras %>% fit(x_train_tbl, y_train_vec, epochs = 11)
```

Plot training & validation accuracy values  
```{r}
plot(fit)
```


We van see from the aboev plot that the accuracy saturates around 84% as the loss continues to drop.


Predicting the Class:  

```{r}
class_pred <- model_keras %>% predict_classes(x_test_tbl)
class_pred[1:20]
```


## NAIVE BAYES CLASSIFIER  

Since, this is a classification problem let us try few other algorithms straight out of the box.  

```{r message=FALSE, warning=FALSE}
model_nb <- train(Class ~., data = train, method = "nb",
                  trControl = trainControl(method = "repeatedcv",
                                             number = 6, repeats = 5),
                  tuneGrid = expand.grid(fL = 1, usekernel = T, adjust = 1))

model_nb
```

As we can see from the summry, that the model is having just 45% accuracy with a kappa score of 0.0284.  

Let us now go ahead and predict the Class variable using the above model and evaluate it through confusion matrix. 

```{r message=FALSE, warning=FALSE, paged.print=FALSE}
predict_nb <- predict(model_nb, newdata = test, type = "raw")
head(predict_nb)

# Confusion Matrix
confusionMatrix(predict_nb, test$Class)
```

## DECISION TREE  

Let us now implement the decision tree algorithm using the training dataset and see if it out performs Naive Bayes.

```{r}
model_dt <- train(Class ~ ., data = train, method = "rpart",
                       metric = "Accuracy",
                       tuneLength = 10,
                       trControl = trainControl(method = "repeatedcv", number = 10, repeats = 3))
```

Siimilar to the previous steps, we will go ahead and predict the Class variable using the decision tree model.  

```{r}
predict_dt <- predict(model_dt, newdata = test, type = "raw")
head(predict_dt)

# Confusion Matrix
confusionMatrix(predict_dt, test$Class)
```

As we can see, the Decision Tree algorithm performs really weell and we can see the same in the accuracy of 69.47%.  

Let us plot above model so that we can have a better idea about the conditions used for partitioning the data.  

```{r}
library(rattle)
fancyRpartPlot(model_dt$finalModel)
```

## RANDOM FOREST

```{r}
set.seed(10)

rf.model <- randomForest(Class ~ ., data = train, importance = TRUE,
                         ntree = 2000, nodesize = 20)

rf.predict <- predict(rf.model, test)
confusionMatrix(test$Class, rf.predict)
```

```{r}
varImpPlot(rf.model)
```

With an accuracy of 76.32%, Random Forest still lags behind the Multilayer Perceptron model.  

## Conclusion  

Based on the accuracy measures, the Multilayer Perceptron outperforms all the other models with an accuracy over 80%.
