---
title: "Machine Learning Project"
author: "S. Purvis"
date: "February 12, 2018"
output: 
    html_document: 
      keep_md: true
---



###Summary



###Introduction
Proper technique in weightlifting is important for effective training and has a positive impact on cardio-resipiratory fitness.  in 2013, Velloso et. al.[^1] used machine learning and pattern recognization techniques to quantify *how well* certain weight lifting activities were being performed.  Study participants completed simple curling execise using sensors mounted in the users' glove, armband, lumbar belt and dumbell.  Under the supervision of an experienced, 6 participants performed 10 sets of bicep curls according to specifications and with 4 specific mistakes that correspond to poor exercise execution.  
  
Fifty eight different measurements where captured for each participant.The investigators then chose 17 features in resulting data set (using created covariates, i.e. mean,variance, etc.) and random forest approach to see if they could build a model that accurately predicts.  In this analysis, we attempt to generate 


```r
library(dplyr)
library(tidyr)
library(ggplot2)
library(lubridate)
library(caret)
library(rattle)
library(cowplot)
```


### Data
Data was imported directly from the provide web address
  

```r
# data files were from site 
download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv",destfile="training.csv",method="libcurl")
download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv",destfile="testing.csv",method="libcurl")

#Create training dataframe
train.dl <- read.csv("training.csv", 
                     header = TRUE, 
                     na.strings = c(""," ","NA"))


#Create test dataframe
test.dl <- read.csv("testing.csv", 
                  header = TRUE, 
                  na.strings = c(""," ","NA"))
```

###Cleaning/Exploration


```r
dim(train.dl)
```

```
## [1] 19622   160
```
Using the dim command, we see a data set with 160 variables.  Review of the dataframe summary (not shown), reveals a large number of variables that are comprised of NAs.  These will be removed.


```r
# There are a large number of variables that have little or no data, remove
#variable "X" is removed as it represents the observation number
train.removeNAs<-train.dl %>% select(which(colMeans(is.na(.)) < 0.5))%>% select(-X)

dim(train.removeNAs)
```

```
## [1] 19622    59
```
This removes 101 variables.
  
Intial view is that all variables are assigned an appropriate class, so no need to make any changes
  
In absence of a data dictionary, it is impossible to draw any conclusions on value variables raw_timestamp_part_1:num_window, therefore, I chose to remove these variables.


```r
train.remove.vars1 <- train.removeNAs %>% select(-c(raw_timestamp_part_1:num_window))
dim(train.remove.vars1)
```

```
## [1] 19622    54
```
This leaves 54 variables
  
Repeat these cleaning steps on the test data set.

```r
test.removeNAs<-test.dl %>% select(which(colMeans(is.na(.)) < 0.5))%>% select(-X)
test.remove.vars1 <- test.removeNAs %>% select(-c(raw_timestamp_part_1:num_window))
```

####Creation of Dummy Variables
During initial analysis, it looked as if there was some differences in the sensor data between users.  To determine if there was, we looked at data from all user when they did the exercise to specification (classe == A) and then looked at a few variables to see if there was any difference between users.

```r
train.A <- train.remove.vars1 %>% filter(classe == "A")

# lets choose a single variable from each sensor that measures the same movement (roll)
belt<- ggplot(train.A, aes(x=user_name, y=roll_belt))+
  geom_boxplot()+
  ggtitle("Roll Belt vs User", subtitle = "Exercise Performed to Spec")

arm<-ggplot(train.A, aes(x=user_name, y=roll_arm))+
  geom_boxplot()+
  ggtitle("Roll Arm vs User", subtitle = "Exercise Performed to Spec")

dbell<- ggplot(train.A, aes(x=user_name, y=roll_dumbbell))+
  geom_boxplot()+
  ggtitle("Roll Dumbbell vs User", subtitle = "Exercise Performed to Spec")

fore<- ggplot(train.A, aes(x=user_name, y=roll_forearm))+
  geom_boxplot()+
  ggtitle("Roll Forearm vs User", subtitle = "Exercise Performed to Spec")

plot_grid(belt, arm, dbell, fore, label_size = 6)
```

![](MachLearnProject_files/figure-html/users-1.png)<!-- -->
  
So, even when the exercise is done to specification, there is signficant difference between users.  I chose to convert these users to dummy variables.

```r
##Convert user_name to dummy variables in training---------------------------------------------
dumMod <- dummyVars(~user_name, data=train.remove.vars1, fullRank=T)
dumtrain <- data.frame(predict(dumMod, newdata = train.remove.vars1))
train.final <- cbind(dumtrain, train.remove.vars1)
train.final <- train.final %>% select(-user_name)


# Repeat on the test data set-------------------------------------------------------
dumModTest <- dummyVars(~user_name, data=test.remove.vars1, fullRank=T)
dumtest <- data.frame(predict(dumModTest, newdata = test.remove.vars1))
test.final <- cbind(dumtest, test.remove.vars1)
test.final <- test.final %>% select(-user_name)
```

Before we move on to feature selection, let's partition our training data into training (used to build the model) and validation (used to determine accuracy as we build our model) data sets.

```r
trainparts <- createDataPartition(y=train.final$classe,
                                  p = 0.75,
                                  list = FALSE)
train.for.modeling <- train.final[trainparts,]
validation <- train.final[-trainparts,]
```


###Feature Selection

####Colinear Features
The first step in feature selection is to remove the highly correlated variables as these will increase bias.

```r
# calculate correlation matrix
set.seed(12345)
correlationMatrix <- cor(train.for.modeling[,6:57])
# find attributes that are highly corrected (ideally >0.75)
highlyCorrelated <- findCorrelation(correlationMatrix, cutoff=0.75)
# print indexes of highly correlated attributes that should be removed from the train dataframe
print(highlyCorrelated)
```

```
##  [1] 10  1  9 22  4 36  8  2 37 35 38 21 34 23 25 13 48 45 31 33 18
```

```r
# remove them 
train.final.cor <- train.final %>% select(-highlyCorrelated)

dim(train.final.cor)
```

```
## [1] 19622    37
```
This removes another 15-20 variables.
  
####Initial Modelling - Predicting with Trees
The goal of this step is to wade through the remaining 34 or so variables, so see if we can find the foundation of essential variables to move use in model building.  We will discuss preprocssing and cross-validation in the modelling section.

```r
# prepare training scheme
set.seed(12345)
#simple train control
control <- trainControl(method="cv", number=5)
# train the initial model
model.rpart <- train(classe~., data=train.final.cor, method="rpart", preProcess=c("center","scale"), trControl=control)
#explor the tree
fancyRpartPlot(model.rpart$finalModel)
```

![](MachLearnProject_files/figure-html/rpart-1.png)<!-- -->



```r
pred.rpart <- predict(model.rpart, validation)
confusionMatrix(pred.rpart,validation$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction   A   B   C   D   E
##          A 907 215 122 169  57
##          B  21 328  18 126 102
##          C 464 406 715 509 332
##          D   0   0   0   0   0
##          E   3   0   0   0 410
## 
## Overall Statistics
##                                           
##                Accuracy : 0.4812          
##                  95% CI : (0.4672, 0.4953)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.343           
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.6502  0.34563   0.8363   0.0000  0.45505
## Specificity            0.8396  0.93249   0.5774   1.0000  0.99925
## Pos Pred Value         0.6170  0.55126   0.2947      NaN  0.99274
## Neg Pred Value         0.8579  0.85588   0.9435   0.8361  0.89067
## Prevalence             0.2845  0.19352   0.1743   0.1639  0.18373
## Detection Rate         0.1850  0.06688   0.1458   0.0000  0.08361
## Detection Prevalence   0.2998  0.12133   0.4947   0.0000  0.08422
## Balanced Accuracy      0.7449  0.63906   0.7068   0.5000  0.72715
```
This simple prediction model uses only 4 variables and is 50% accurate.  Let's use these 4 variables moving forward to see how they perform.

###Modeling
#### Preprocessing and Cross-Validation
During data exploration, it became clear that many of the variables were highly skewed and non guassian.  To prepare these data for modeling, I used the preProcess argument of the train function to normalize the data (center and scale).

Accuracy measures on the training set are optimistic.  To minimize this overestimation of accuracy, we employed cross-validation techniques during modeling. Cross-validation is a powerful preventative measure against overfitting.  To balance computation time and performance, I chose to use the "repeatedcv" method, done 10x and repeated 3x.

####Model Selection

```r
# this code creates a formula object that we can use to pick out the best algorithm
ini.formula <- as.formula(classe~roll_belt+pitch_forearm+magnet_dumbbell_y+magnet_dumbbell_z)

set.seed(1234)
control <- trainControl(method="repeatedcv", number=10, repeats = 3)
#train random forest model
train.model.rf <- train(ini.formula, data=train.final.cor, method="rf", preProcess=c("center","scale"), trControl=control)
#train gradient boosting model
train.model.gbm <- train(ini.formula, data=train.final.cor, method="gbm", preProcess=c("center","scale"), trControl=control)
```

```
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1308
##      2        1.5237             nan     0.1000    0.0849
##      3        1.4674             nan     0.1000    0.0641
##      4        1.4244             nan     0.1000    0.0497
##      5        1.3907             nan     0.1000    0.0401
##      6        1.3644             nan     0.1000    0.0338
##      7        1.3417             nan     0.1000    0.0286
##      8        1.3228             nan     0.1000    0.0278
##      9        1.3043             nan     0.1000    0.0232
##     10        1.2894             nan     0.1000    0.0272
##     20        1.1679             nan     0.1000    0.0114
##     40        1.0588             nan     0.1000    0.0050
##     60        0.9954             nan     0.1000    0.0026
##     80        0.9526             nan     0.1000    0.0026
##    100        0.9169             nan     0.1000    0.0020
##    120        0.8893             nan     0.1000    0.0023
##    140        0.8641             nan     0.1000    0.0020
##    150        0.8537             nan     0.1000    0.0012
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1714
##      2        1.4974             nan     0.1000    0.1150
##      3        1.4220             nan     0.1000    0.0884
##      4        1.3647             nan     0.1000    0.0716
##      5        1.3185             nan     0.1000    0.0589
##      6        1.2804             nan     0.1000    0.0529
##      7        1.2463             nan     0.1000    0.0393
##      8        1.2200             nan     0.1000    0.0427
##      9        1.1935             nan     0.1000    0.0328
##     10        1.1718             nan     0.1000    0.0342
##     20        1.0307             nan     0.1000    0.0176
##     40        0.8743             nan     0.1000    0.0083
##     60        0.7813             nan     0.1000    0.0060
##     80        0.7132             nan     0.1000    0.0037
##    100        0.6611             nan     0.1000    0.0029
##    120        0.6245             nan     0.1000    0.0028
##    140        0.5925             nan     0.1000    0.0010
##    150        0.5786             nan     0.1000    0.0021
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.2061
##      2        1.4774             nan     0.1000    0.1358
##      3        1.3913             nan     0.1000    0.1136
##      4        1.3194             nan     0.1000    0.0859
##      5        1.2645             nan     0.1000    0.0694
##      6        1.2197             nan     0.1000    0.0557
##      7        1.1837             nan     0.1000    0.0563
##      8        1.1453             nan     0.1000    0.0568
##      9        1.1090             nan     0.1000    0.0389
##     10        1.0842             nan     0.1000    0.0362
##     20        0.9071             nan     0.1000    0.0167
##     40        0.7450             nan     0.1000    0.0092
##     60        0.6475             nan     0.1000    0.0035
##     80        0.5811             nan     0.1000    0.0027
##    100        0.5361             nan     0.1000    0.0045
##    120        0.4970             nan     0.1000    0.0017
##    140        0.4693             nan     0.1000    0.0016
##    150        0.4551             nan     0.1000    0.0015
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1297
##      2        1.5246             nan     0.1000    0.0843
##      3        1.4688             nan     0.1000    0.0650
##      4        1.4265             nan     0.1000    0.0474
##      5        1.3945             nan     0.1000    0.0412
##      6        1.3674             nan     0.1000    0.0339
##      7        1.3450             nan     0.1000    0.0308
##      8        1.3249             nan     0.1000    0.0265
##      9        1.3079             nan     0.1000    0.0256
##     10        1.2914             nan     0.1000    0.0252
##     20        1.1708             nan     0.1000    0.0121
##     40        1.0603             nan     0.1000    0.0067
##     60        0.9975             nan     0.1000    0.0038
##     80        0.9551             nan     0.1000    0.0016
##    100        0.9184             nan     0.1000    0.0014
##    120        0.8895             nan     0.1000    0.0025
##    140        0.8647             nan     0.1000    0.0011
##    150        0.8548             nan     0.1000    0.0018
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1666
##      2        1.4980             nan     0.1000    0.1143
##      3        1.4242             nan     0.1000    0.0853
##      4        1.3669             nan     0.1000    0.0709
##      5        1.3207             nan     0.1000    0.0524
##      6        1.2853             nan     0.1000    0.0477
##      7        1.2538             nan     0.1000    0.0473
##      8        1.2235             nan     0.1000    0.0440
##      9        1.1964             nan     0.1000    0.0358
##     10        1.1741             nan     0.1000    0.0311
##     20        1.0338             nan     0.1000    0.0157
##     40        0.8850             nan     0.1000    0.0133
##     60        0.7816             nan     0.1000    0.0053
##     80        0.7130             nan     0.1000    0.0042
##    100        0.6646             nan     0.1000    0.0028
##    120        0.6251             nan     0.1000    0.0021
##    140        0.5934             nan     0.1000    0.0018
##    150        0.5799             nan     0.1000    0.0020
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1980
##      2        1.4803             nan     0.1000    0.1464
##      3        1.3879             nan     0.1000    0.1036
##      4        1.3215             nan     0.1000    0.0792
##      5        1.2694             nan     0.1000    0.0806
##      6        1.2160             nan     0.1000    0.0617
##      7        1.1772             nan     0.1000    0.0558
##      8        1.1397             nan     0.1000    0.0455
##      9        1.1111             nan     0.1000    0.0486
##     10        1.0805             nan     0.1000    0.0393
##     20        0.9065             nan     0.1000    0.0238
##     40        0.7421             nan     0.1000    0.0086
##     60        0.6469             nan     0.1000    0.0054
##     80        0.5748             nan     0.1000    0.0037
##    100        0.5285             nan     0.1000    0.0033
##    120        0.4926             nan     0.1000    0.0015
##    140        0.4646             nan     0.1000    0.0019
##    150        0.4508             nan     0.1000    0.0014
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1306
##      2        1.5235             nan     0.1000    0.0875
##      3        1.4661             nan     0.1000    0.0640
##      4        1.4238             nan     0.1000    0.0504
##      5        1.3909             nan     0.1000    0.0400
##      6        1.3643             nan     0.1000    0.0340
##      7        1.3415             nan     0.1000    0.0285
##      8        1.3220             nan     0.1000    0.0252
##      9        1.3052             nan     0.1000    0.0246
##     10        1.2894             nan     0.1000    0.0252
##     20        1.1675             nan     0.1000    0.0135
##     40        1.0560             nan     0.1000    0.0056
##     60        0.9936             nan     0.1000    0.0024
##     80        0.9499             nan     0.1000    0.0033
##    100        0.9139             nan     0.1000    0.0023
##    120        0.8849             nan     0.1000    0.0009
##    140        0.8596             nan     0.1000    0.0014
##    150        0.8490             nan     0.1000    0.0010
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1754
##      2        1.4968             nan     0.1000    0.1147
##      3        1.4217             nan     0.1000    0.0885
##      4        1.3648             nan     0.1000    0.0706
##      5        1.3200             nan     0.1000    0.0590
##      6        1.2823             nan     0.1000    0.0526
##      7        1.2488             nan     0.1000    0.0463
##      8        1.2193             nan     0.1000    0.0379
##      9        1.1951             nan     0.1000    0.0379
##     10        1.1713             nan     0.1000    0.0344
##     20        1.0319             nan     0.1000    0.0154
##     40        0.8842             nan     0.1000    0.0090
##     60        0.7782             nan     0.1000    0.0045
##     80        0.7129             nan     0.1000    0.0044
##    100        0.6632             nan     0.1000    0.0034
##    120        0.6222             nan     0.1000    0.0021
##    140        0.5877             nan     0.1000    0.0008
##    150        0.5729             nan     0.1000    0.0020
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.2026
##      2        1.4767             nan     0.1000    0.1392
##      3        1.3871             nan     0.1000    0.1074
##      4        1.3187             nan     0.1000    0.0857
##      5        1.2634             nan     0.1000    0.0808
##      6        1.2111             nan     0.1000    0.0616
##      7        1.1718             nan     0.1000    0.0470
##      8        1.1414             nan     0.1000    0.0426
##      9        1.1146             nan     0.1000    0.0438
##     10        1.0851             nan     0.1000    0.0374
##     20        0.9079             nan     0.1000    0.0256
##     40        0.7332             nan     0.1000    0.0109
##     60        0.6379             nan     0.1000    0.0061
##     80        0.5727             nan     0.1000    0.0019
##    100        0.5256             nan     0.1000    0.0037
##    120        0.4888             nan     0.1000    0.0016
##    140        0.4604             nan     0.1000    0.0017
##    150        0.4478             nan     0.1000    0.0010
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1277
##      2        1.5241             nan     0.1000    0.0863
##      3        1.4671             nan     0.1000    0.0622
##      4        1.4253             nan     0.1000    0.0490
##      5        1.3922             nan     0.1000    0.0410
##      6        1.3653             nan     0.1000    0.0328
##      7        1.3432             nan     0.1000    0.0292
##      8        1.3242             nan     0.1000    0.0281
##      9        1.3055             nan     0.1000    0.0275
##     10        1.2862             nan     0.1000    0.0259
##     20        1.1687             nan     0.1000    0.0125
##     40        1.0594             nan     0.1000    0.0045
##     60        0.9972             nan     0.1000    0.0043
##     80        0.9529             nan     0.1000    0.0029
##    100        0.9174             nan     0.1000    0.0036
##    120        0.8887             nan     0.1000    0.0018
##    140        0.8638             nan     0.1000    0.0013
##    150        0.8533             nan     0.1000    0.0009
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1720
##      2        1.4972             nan     0.1000    0.1167
##      3        1.4209             nan     0.1000    0.0864
##      4        1.3651             nan     0.1000    0.0684
##      5        1.3215             nan     0.1000    0.0568
##      6        1.2845             nan     0.1000    0.0507
##      7        1.2520             nan     0.1000    0.0517
##      8        1.2193             nan     0.1000    0.0425
##      9        1.1921             nan     0.1000    0.0328
##     10        1.1709             nan     0.1000    0.0336
##     20        1.0333             nan     0.1000    0.0137
##     40        0.8711             nan     0.1000    0.0079
##     60        0.7738             nan     0.1000    0.0045
##     80        0.7103             nan     0.1000    0.0029
##    100        0.6625             nan     0.1000    0.0031
##    120        0.6217             nan     0.1000    0.0019
##    140        0.5908             nan     0.1000    0.0021
##    150        0.5777             nan     0.1000    0.0015
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.2008
##      2        1.4803             nan     0.1000    0.1393
##      3        1.3897             nan     0.1000    0.1082
##      4        1.3209             nan     0.1000    0.0887
##      5        1.2654             nan     0.1000    0.0706
##      6        1.2205             nan     0.1000    0.0567
##      7        1.1845             nan     0.1000    0.0552
##      8        1.1459             nan     0.1000    0.0534
##      9        1.1110             nan     0.1000    0.0462
##     10        1.0816             nan     0.1000    0.0429
##     20        0.9051             nan     0.1000    0.0237
##     40        0.7375             nan     0.1000    0.0074
##     60        0.6447             nan     0.1000    0.0040
##     80        0.5792             nan     0.1000    0.0040
##    100        0.5301             nan     0.1000    0.0022
##    120        0.4940             nan     0.1000    0.0018
##    140        0.4653             nan     0.1000    0.0015
##    150        0.4531             nan     0.1000    0.0014
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1278
##      2        1.5226             nan     0.1000    0.0833
##      3        1.4662             nan     0.1000    0.0621
##      4        1.4243             nan     0.1000    0.0501
##      5        1.3912             nan     0.1000    0.0409
##      6        1.3643             nan     0.1000    0.0326
##      7        1.3429             nan     0.1000    0.0265
##      8        1.3250             nan     0.1000    0.0295
##      9        1.3063             nan     0.1000    0.0281
##     10        1.2863             nan     0.1000    0.0244
##     20        1.1670             nan     0.1000    0.0127
##     40        1.0571             nan     0.1000    0.0055
##     60        0.9948             nan     0.1000    0.0041
##     80        0.9538             nan     0.1000    0.0031
##    100        0.9195             nan     0.1000    0.0019
##    120        0.8903             nan     0.1000    0.0011
##    140        0.8657             nan     0.1000    0.0011
##    150        0.8537             nan     0.1000    0.0005
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1715
##      2        1.4979             nan     0.1000    0.1211
##      3        1.4209             nan     0.1000    0.0864
##      4        1.3654             nan     0.1000    0.0688
##      5        1.3201             nan     0.1000    0.0617
##      6        1.2808             nan     0.1000    0.0523
##      7        1.2469             nan     0.1000    0.0414
##      8        1.2202             nan     0.1000    0.0412
##      9        1.1934             nan     0.1000    0.0342
##     10        1.1713             nan     0.1000    0.0344
##     20        1.0273             nan     0.1000    0.0138
##     40        0.8727             nan     0.1000    0.0060
##     60        0.7784             nan     0.1000    0.0053
##     80        0.7110             nan     0.1000    0.0041
##    100        0.6624             nan     0.1000    0.0020
##    120        0.6214             nan     0.1000    0.0022
##    140        0.5899             nan     0.1000    0.0027
##    150        0.5758             nan     0.1000    0.0010
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1997
##      2        1.4797             nan     0.1000    0.1380
##      3        1.3923             nan     0.1000    0.1134
##      4        1.3211             nan     0.1000    0.0919
##      5        1.2636             nan     0.1000    0.0710
##      6        1.2188             nan     0.1000    0.0599
##      7        1.1809             nan     0.1000    0.0577
##      8        1.1408             nan     0.1000    0.0504
##      9        1.1093             nan     0.1000    0.0423
##     10        1.0831             nan     0.1000    0.0414
##     20        0.9022             nan     0.1000    0.0209
##     40        0.7492             nan     0.1000    0.0110
##     60        0.6472             nan     0.1000    0.0055
##     80        0.5804             nan     0.1000    0.0046
##    100        0.5333             nan     0.1000    0.0036
##    120        0.4965             nan     0.1000    0.0018
##    140        0.4678             nan     0.1000    0.0019
##    150        0.4559             nan     0.1000    0.0021
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1281
##      2        1.5251             nan     0.1000    0.0819
##      3        1.4694             nan     0.1000    0.0655
##      4        1.4264             nan     0.1000    0.0492
##      5        1.3941             nan     0.1000    0.0412
##      6        1.3676             nan     0.1000    0.0343
##      7        1.3456             nan     0.1000    0.0300
##      8        1.3254             nan     0.1000    0.0264
##      9        1.3085             nan     0.1000    0.0239
##     10        1.2919             nan     0.1000    0.0285
##     20        1.1712             nan     0.1000    0.0154
##     40        1.0584             nan     0.1000    0.0053
##     60        0.9984             nan     0.1000    0.0042
##     80        0.9548             nan     0.1000    0.0028
##    100        0.9210             nan     0.1000    0.0024
##    120        0.8909             nan     0.1000    0.0019
##    140        0.8669             nan     0.1000    0.0013
##    150        0.8549             nan     0.1000    0.0011
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1703
##      2        1.4980             nan     0.1000    0.1185
##      3        1.4218             nan     0.1000    0.0851
##      4        1.3654             nan     0.1000    0.0679
##      5        1.3209             nan     0.1000    0.0590
##      6        1.2832             nan     0.1000    0.0501
##      7        1.2508             nan     0.1000    0.0384
##      8        1.2253             nan     0.1000    0.0460
##      9        1.1961             nan     0.1000    0.0361
##     10        1.1734             nan     0.1000    0.0339
##     20        1.0307             nan     0.1000    0.0128
##     40        0.8877             nan     0.1000    0.0108
##     60        0.7831             nan     0.1000    0.0061
##     80        0.7165             nan     0.1000    0.0062
##    100        0.6633             nan     0.1000    0.0042
##    120        0.6237             nan     0.1000    0.0028
##    140        0.5934             nan     0.1000    0.0020
##    150        0.5793             nan     0.1000    0.0006
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.2000
##      2        1.4790             nan     0.1000    0.1395
##      3        1.3899             nan     0.1000    0.1085
##      4        1.3211             nan     0.1000    0.0891
##      5        1.2657             nan     0.1000    0.0729
##      6        1.2190             nan     0.1000    0.0666
##      7        1.1752             nan     0.1000    0.0488
##      8        1.1440             nan     0.1000    0.0453
##      9        1.1155             nan     0.1000    0.0435
##     10        1.0865             nan     0.1000    0.0367
##     20        0.9064             nan     0.1000    0.0192
##     40        0.7432             nan     0.1000    0.0074
##     60        0.6454             nan     0.1000    0.0064
##     80        0.5784             nan     0.1000    0.0046
##    100        0.5303             nan     0.1000    0.0024
##    120        0.4935             nan     0.1000    0.0009
##    140        0.4639             nan     0.1000    0.0024
##    150        0.4495             nan     0.1000    0.0010
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1266
##      2        1.5242             nan     0.1000    0.0847
##      3        1.4679             nan     0.1000    0.0644
##      4        1.4257             nan     0.1000    0.0503
##      5        1.3927             nan     0.1000    0.0402
##      6        1.3661             nan     0.1000    0.0337
##      7        1.3436             nan     0.1000    0.0309
##      8        1.3235             nan     0.1000    0.0281
##      9        1.3052             nan     0.1000    0.0253
##     10        1.2886             nan     0.1000    0.0251
##     20        1.1683             nan     0.1000    0.0111
##     40        1.0569             nan     0.1000    0.0057
##     60        0.9940             nan     0.1000    0.0045
##     80        0.9513             nan     0.1000    0.0023
##    100        0.9175             nan     0.1000    0.0014
##    120        0.8882             nan     0.1000    0.0014
##    140        0.8646             nan     0.1000    0.0025
##    150        0.8535             nan     0.1000    0.0005
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1707
##      2        1.4979             nan     0.1000    0.1161
##      3        1.4223             nan     0.1000    0.0837
##      4        1.3683             nan     0.1000    0.0708
##      5        1.3227             nan     0.1000    0.0624
##      6        1.2835             nan     0.1000    0.0498
##      7        1.2514             nan     0.1000    0.0485
##      8        1.2205             nan     0.1000    0.0396
##      9        1.1942             nan     0.1000    0.0370
##     10        1.1706             nan     0.1000    0.0316
##     20        1.0293             nan     0.1000    0.0136
##     40        0.8776             nan     0.1000    0.0125
##     60        0.7760             nan     0.1000    0.0061
##     80        0.7109             nan     0.1000    0.0035
##    100        0.6626             nan     0.1000    0.0047
##    120        0.6215             nan     0.1000    0.0016
##    140        0.5907             nan     0.1000    0.0016
##    150        0.5768             nan     0.1000    0.0016
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.2027
##      2        1.4779             nan     0.1000    0.1440
##      3        1.3860             nan     0.1000    0.1040
##      4        1.3199             nan     0.1000    0.0857
##      5        1.2666             nan     0.1000    0.0756
##      6        1.2183             nan     0.1000    0.0588
##      7        1.1805             nan     0.1000    0.0498
##      8        1.1485             nan     0.1000    0.0520
##      9        1.1126             nan     0.1000    0.0402
##     10        1.0855             nan     0.1000    0.0367
##     20        0.9011             nan     0.1000    0.0223
##     40        0.7363             nan     0.1000    0.0063
##     60        0.6430             nan     0.1000    0.0051
##     80        0.5731             nan     0.1000    0.0034
##    100        0.5272             nan     0.1000    0.0024
##    120        0.4901             nan     0.1000    0.0016
##    140        0.4606             nan     0.1000    0.0014
##    150        0.4476             nan     0.1000    0.0014
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1306
##      2        1.5250             nan     0.1000    0.0848
##      3        1.4686             nan     0.1000    0.0651
##      4        1.4261             nan     0.1000    0.0511
##      5        1.3931             nan     0.1000    0.0401
##      6        1.3666             nan     0.1000    0.0345
##      7        1.3443             nan     0.1000    0.0305
##      8        1.3244             nan     0.1000    0.0250
##      9        1.3079             nan     0.1000    0.0280
##     10        1.2887             nan     0.1000    0.0222
##     20        1.1718             nan     0.1000    0.0119
##     40        1.0614             nan     0.1000    0.0062
##     60        0.9979             nan     0.1000    0.0028
##     80        0.9530             nan     0.1000    0.0024
##    100        0.9194             nan     0.1000    0.0029
##    120        0.8904             nan     0.1000    0.0015
##    140        0.8654             nan     0.1000    0.0011
##    150        0.8547             nan     0.1000    0.0011
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1706
##      2        1.4956             nan     0.1000    0.1139
##      3        1.4207             nan     0.1000    0.0869
##      4        1.3641             nan     0.1000    0.0701
##      5        1.3191             nan     0.1000    0.0555
##      6        1.2826             nan     0.1000    0.0462
##      7        1.2524             nan     0.1000    0.0473
##      8        1.2220             nan     0.1000    0.0385
##      9        1.1966             nan     0.1000    0.0360
##     10        1.1740             nan     0.1000    0.0319
##     20        1.0320             nan     0.1000    0.0123
##     40        0.8801             nan     0.1000    0.0079
##     60        0.7808             nan     0.1000    0.0095
##     80        0.7104             nan     0.1000    0.0035
##    100        0.6640             nan     0.1000    0.0040
##    120        0.6239             nan     0.1000    0.0022
##    140        0.5913             nan     0.1000    0.0017
##    150        0.5782             nan     0.1000    0.0020
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.2094
##      2        1.4748             nan     0.1000    0.1398
##      3        1.3869             nan     0.1000    0.1011
##      4        1.3214             nan     0.1000    0.0908
##      5        1.2616             nan     0.1000    0.0740
##      6        1.2148             nan     0.1000    0.0614
##      7        1.1755             nan     0.1000    0.0528
##      8        1.1418             nan     0.1000    0.0532
##      9        1.1067             nan     0.1000    0.0401
##     10        1.0812             nan     0.1000    0.0393
##     20        0.9028             nan     0.1000    0.0146
##     40        0.7433             nan     0.1000    0.0093
##     60        0.6451             nan     0.1000    0.0056
##     80        0.5774             nan     0.1000    0.0052
##    100        0.5306             nan     0.1000    0.0023
##    120        0.4940             nan     0.1000    0.0016
##    140        0.4635             nan     0.1000    0.0016
##    150        0.4506             nan     0.1000    0.0010
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1273
##      2        1.5246             nan     0.1000    0.0859
##      3        1.4678             nan     0.1000    0.0638
##      4        1.4264             nan     0.1000    0.0501
##      5        1.3927             nan     0.1000    0.0393
##      6        1.3666             nan     0.1000    0.0338
##      7        1.3447             nan     0.1000    0.0318
##      8        1.3239             nan     0.1000    0.0257
##      9        1.3074             nan     0.1000    0.0239
##     10        1.2919             nan     0.1000    0.0270
##     20        1.1691             nan     0.1000    0.0129
##     40        1.0596             nan     0.1000    0.0048
##     60        0.9964             nan     0.1000    0.0037
##     80        0.9531             nan     0.1000    0.0032
##    100        0.9183             nan     0.1000    0.0027
##    120        0.8891             nan     0.1000    0.0014
##    140        0.8658             nan     0.1000    0.0020
##    150        0.8540             nan     0.1000    0.0009
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1696
##      2        1.4973             nan     0.1000    0.1135
##      3        1.4228             nan     0.1000    0.0850
##      4        1.3666             nan     0.1000    0.0740
##      5        1.3198             nan     0.1000    0.0569
##      6        1.2829             nan     0.1000    0.0516
##      7        1.2498             nan     0.1000    0.0406
##      8        1.2227             nan     0.1000    0.0428
##      9        1.1959             nan     0.1000    0.0349
##     10        1.1728             nan     0.1000    0.0383
##     20        1.0335             nan     0.1000    0.0157
##     40        0.8856             nan     0.1000    0.0089
##     60        0.7831             nan     0.1000    0.0072
##     80        0.7135             nan     0.1000    0.0033
##    100        0.6605             nan     0.1000    0.0019
##    120        0.6233             nan     0.1000    0.0022
##    140        0.5883             nan     0.1000    0.0025
##    150        0.5734             nan     0.1000    0.0010
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.2069
##      2        1.4774             nan     0.1000    0.1432
##      3        1.3875             nan     0.1000    0.1058
##      4        1.3192             nan     0.1000    0.0902
##      5        1.2639             nan     0.1000    0.0663
##      6        1.2212             nan     0.1000    0.0556
##      7        1.1842             nan     0.1000    0.0614
##      8        1.1430             nan     0.1000    0.0441
##      9        1.1149             nan     0.1000    0.0457
##     10        1.0844             nan     0.1000    0.0389
##     20        0.9140             nan     0.1000    0.0221
##     40        0.7373             nan     0.1000    0.0083
##     60        0.6419             nan     0.1000    0.0048
##     80        0.5761             nan     0.1000    0.0025
##    100        0.5295             nan     0.1000    0.0033
##    120        0.4914             nan     0.1000    0.0015
##    140        0.4636             nan     0.1000    0.0018
##    150        0.4513             nan     0.1000    0.0016
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1278
##      2        1.5239             nan     0.1000    0.0813
##      3        1.4679             nan     0.1000    0.0650
##      4        1.4252             nan     0.1000    0.0483
##      5        1.3925             nan     0.1000    0.0388
##      6        1.3662             nan     0.1000    0.0312
##      7        1.3450             nan     0.1000    0.0332
##      8        1.3239             nan     0.1000    0.0249
##      9        1.3065             nan     0.1000    0.0245
##     10        1.2908             nan     0.1000    0.0284
##     20        1.1701             nan     0.1000    0.0125
##     40        1.0592             nan     0.1000    0.0061
##     60        0.9964             nan     0.1000    0.0029
##     80        0.9537             nan     0.1000    0.0024
##    100        0.9197             nan     0.1000    0.0018
##    120        0.8889             nan     0.1000    0.0024
##    140        0.8649             nan     0.1000    0.0019
##    150        0.8541             nan     0.1000    0.0010
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1695
##      2        1.4981             nan     0.1000    0.1181
##      3        1.4216             nan     0.1000    0.0837
##      4        1.3669             nan     0.1000    0.0677
##      5        1.3225             nan     0.1000    0.0593
##      6        1.2839             nan     0.1000    0.0527
##      7        1.2504             nan     0.1000    0.0458
##      8        1.2212             nan     0.1000    0.0419
##      9        1.1941             nan     0.1000    0.0350
##     10        1.1720             nan     0.1000    0.0341
##     20        1.0306             nan     0.1000    0.0133
##     40        0.8797             nan     0.1000    0.0106
##     60        0.7808             nan     0.1000    0.0060
##     80        0.7106             nan     0.1000    0.0053
##    100        0.6604             nan     0.1000    0.0030
##    120        0.6221             nan     0.1000    0.0023
##    140        0.5893             nan     0.1000    0.0014
##    150        0.5755             nan     0.1000    0.0011
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.2083
##      2        1.4770             nan     0.1000    0.1421
##      3        1.3857             nan     0.1000    0.1014
##      4        1.3203             nan     0.1000    0.0865
##      5        1.2659             nan     0.1000    0.0739
##      6        1.2189             nan     0.1000    0.0672
##      7        1.1737             nan     0.1000    0.0516
##      8        1.1404             nan     0.1000    0.0415
##      9        1.1103             nan     0.1000    0.0467
##     10        1.0817             nan     0.1000    0.0415
##     20        0.9127             nan     0.1000    0.0263
##     40        0.7392             nan     0.1000    0.0071
##     60        0.6458             nan     0.1000    0.0046
##     80        0.5791             nan     0.1000    0.0034
##    100        0.5303             nan     0.1000    0.0026
##    120        0.4963             nan     0.1000    0.0017
##    140        0.4653             nan     0.1000    0.0022
##    150        0.4510             nan     0.1000    0.0012
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1267
##      2        1.5250             nan     0.1000    0.0836
##      3        1.4692             nan     0.1000    0.0635
##      4        1.4277             nan     0.1000    0.0495
##      5        1.3946             nan     0.1000    0.0395
##      6        1.3681             nan     0.1000    0.0320
##      7        1.3460             nan     0.1000    0.0310
##      8        1.3249             nan     0.1000    0.0262
##      9        1.3074             nan     0.1000    0.0252
##     10        1.2910             nan     0.1000    0.0268
##     20        1.1709             nan     0.1000    0.0140
##     40        1.0605             nan     0.1000    0.0057
##     60        0.9985             nan     0.1000    0.0034
##     80        0.9547             nan     0.1000    0.0026
##    100        0.9208             nan     0.1000    0.0032
##    120        0.8906             nan     0.1000    0.0022
##    140        0.8673             nan     0.1000    0.0016
##    150        0.8551             nan     0.1000    0.0006
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1731
##      2        1.4981             nan     0.1000    0.1147
##      3        1.4241             nan     0.1000    0.0881
##      4        1.3670             nan     0.1000    0.0721
##      5        1.3209             nan     0.1000    0.0573
##      6        1.2837             nan     0.1000    0.0516
##      7        1.2500             nan     0.1000    0.0458
##      8        1.2208             nan     0.1000    0.0356
##      9        1.1974             nan     0.1000    0.0374
##     10        1.1733             nan     0.1000    0.0342
##     20        1.0288             nan     0.1000    0.0143
##     40        0.8793             nan     0.1000    0.0117
##     60        0.7817             nan     0.1000    0.0077
##     80        0.7133             nan     0.1000    0.0034
##    100        0.6628             nan     0.1000    0.0029
##    120        0.6240             nan     0.1000    0.0019
##    140        0.5933             nan     0.1000    0.0016
##    150        0.5780             nan     0.1000    0.0016
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.2018
##      2        1.4797             nan     0.1000    0.1349
##      3        1.3927             nan     0.1000    0.1041
##      4        1.3264             nan     0.1000    0.0921
##      5        1.2690             nan     0.1000    0.0711
##      6        1.2230             nan     0.1000    0.0532
##      7        1.1880             nan     0.1000    0.0571
##      8        1.1492             nan     0.1000    0.0524
##      9        1.1170             nan     0.1000    0.0405
##     10        1.0910             nan     0.1000    0.0477
##     20        0.9055             nan     0.1000    0.0174
##     40        0.7433             nan     0.1000    0.0096
##     60        0.6450             nan     0.1000    0.0038
##     80        0.5817             nan     0.1000    0.0047
##    100        0.5332             nan     0.1000    0.0034
##    120        0.4972             nan     0.1000    0.0017
##    140        0.4684             nan     0.1000    0.0016
##    150        0.4555             nan     0.1000    0.0016
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1274
##      2        1.5230             nan     0.1000    0.0853
##      3        1.4664             nan     0.1000    0.0625
##      4        1.4248             nan     0.1000    0.0531
##      5        1.3910             nan     0.1000    0.0403
##      6        1.3646             nan     0.1000    0.0339
##      7        1.3417             nan     0.1000    0.0278
##      8        1.3233             nan     0.1000    0.0269
##      9        1.3057             nan     0.1000    0.0254
##     10        1.2892             nan     0.1000    0.0287
##     20        1.1670             nan     0.1000    0.0132
##     40        1.0574             nan     0.1000    0.0059
##     60        0.9944             nan     0.1000    0.0037
##     80        0.9492             nan     0.1000    0.0027
##    100        0.9162             nan     0.1000    0.0014
##    120        0.8868             nan     0.1000    0.0016
##    140        0.8616             nan     0.1000    0.0014
##    150        0.8503             nan     0.1000    0.0020
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1724
##      2        1.4966             nan     0.1000    0.1155
##      3        1.4208             nan     0.1000    0.0885
##      4        1.3640             nan     0.1000    0.0681
##      5        1.3184             nan     0.1000    0.0592
##      6        1.2801             nan     0.1000    0.0496
##      7        1.2477             nan     0.1000    0.0468
##      8        1.2176             nan     0.1000    0.0418
##      9        1.1912             nan     0.1000    0.0369
##     10        1.1682             nan     0.1000    0.0331
##     20        1.0261             nan     0.1000    0.0151
##     40        0.8716             nan     0.1000    0.0072
##     60        0.7756             nan     0.1000    0.0064
##     80        0.7108             nan     0.1000    0.0025
##    100        0.6575             nan     0.1000    0.0040
##    120        0.6207             nan     0.1000    0.0028
##    140        0.5895             nan     0.1000    0.0009
##    150        0.5755             nan     0.1000    0.0015
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1973
##      2        1.4798             nan     0.1000    0.1413
##      3        1.3890             nan     0.1000    0.1079
##      4        1.3212             nan     0.1000    0.0904
##      5        1.2634             nan     0.1000    0.0711
##      6        1.2189             nan     0.1000    0.0665
##      7        1.1744             nan     0.1000    0.0521
##      8        1.1413             nan     0.1000    0.0470
##      9        1.1119             nan     0.1000    0.0507
##     10        1.0787             nan     0.1000    0.0385
##     20        0.8972             nan     0.1000    0.0245
##     40        0.7349             nan     0.1000    0.0129
##     60        0.6438             nan     0.1000    0.0050
##     80        0.5748             nan     0.1000    0.0041
##    100        0.5282             nan     0.1000    0.0040
##    120        0.4928             nan     0.1000    0.0017
##    140        0.4629             nan     0.1000    0.0019
##    150        0.4504             nan     0.1000    0.0014
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1282
##      2        1.5239             nan     0.1000    0.0849
##      3        1.4667             nan     0.1000    0.0614
##      4        1.4261             nan     0.1000    0.0502
##      5        1.3933             nan     0.1000    0.0401
##      6        1.3670             nan     0.1000    0.0333
##      7        1.3448             nan     0.1000    0.0308
##      8        1.3248             nan     0.1000    0.0252
##      9        1.3087             nan     0.1000    0.0257
##     10        1.2903             nan     0.1000    0.0223
##     20        1.1711             nan     0.1000    0.0122
##     40        1.0609             nan     0.1000    0.0055
##     60        0.9989             nan     0.1000    0.0040
##     80        0.9554             nan     0.1000    0.0016
##    100        0.9213             nan     0.1000    0.0022
##    120        0.8917             nan     0.1000    0.0012
##    140        0.8670             nan     0.1000    0.0019
##    150        0.8562             nan     0.1000    0.0007
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1724
##      2        1.4967             nan     0.1000    0.1158
##      3        1.4222             nan     0.1000    0.0841
##      4        1.3675             nan     0.1000    0.0702
##      5        1.3228             nan     0.1000    0.0594
##      6        1.2844             nan     0.1000    0.0530
##      7        1.2502             nan     0.1000    0.0433
##      8        1.2221             nan     0.1000    0.0426
##      9        1.1945             nan     0.1000    0.0344
##     10        1.1725             nan     0.1000    0.0363
##     20        1.0306             nan     0.1000    0.0157
##     40        0.8781             nan     0.1000    0.0086
##     60        0.7794             nan     0.1000    0.0047
##     80        0.7139             nan     0.1000    0.0066
##    100        0.6618             nan     0.1000    0.0032
##    120        0.6204             nan     0.1000    0.0046
##    140        0.5870             nan     0.1000    0.0014
##    150        0.5731             nan     0.1000    0.0014
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.2042
##      2        1.4784             nan     0.1000    0.1390
##      3        1.3899             nan     0.1000    0.1002
##      4        1.3255             nan     0.1000    0.0933
##      5        1.2660             nan     0.1000    0.0723
##      6        1.2206             nan     0.1000    0.0614
##      7        1.1787             nan     0.1000    0.0535
##      8        1.1445             nan     0.1000    0.0488
##      9        1.1113             nan     0.1000    0.0409
##     10        1.0857             nan     0.1000    0.0327
##     20        0.9045             nan     0.1000    0.0227
##     40        0.7401             nan     0.1000    0.0100
##     60        0.6448             nan     0.1000    0.0070
##     80        0.5785             nan     0.1000    0.0049
##    100        0.5282             nan     0.1000    0.0019
##    120        0.4925             nan     0.1000    0.0025
##    140        0.4657             nan     0.1000    0.0014
##    150        0.4521             nan     0.1000    0.0005
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1281
##      2        1.5234             nan     0.1000    0.0855
##      3        1.4670             nan     0.1000    0.0629
##      4        1.4247             nan     0.1000    0.0523
##      5        1.3905             nan     0.1000    0.0409
##      6        1.3632             nan     0.1000    0.0353
##      7        1.3408             nan     0.1000    0.0301
##      8        1.3209             nan     0.1000    0.0263
##      9        1.3039             nan     0.1000    0.0252
##     10        1.2876             nan     0.1000    0.0285
##     20        1.1674             nan     0.1000    0.0125
##     40        1.0543             nan     0.1000    0.0062
##     60        0.9934             nan     0.1000    0.0034
##     80        0.9506             nan     0.1000    0.0022
##    100        0.9164             nan     0.1000    0.0034
##    120        0.8871             nan     0.1000    0.0019
##    140        0.8614             nan     0.1000    0.0009
##    150        0.8508             nan     0.1000    0.0010
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1735
##      2        1.4962             nan     0.1000    0.1153
##      3        1.4212             nan     0.1000    0.0868
##      4        1.3640             nan     0.1000    0.0724
##      5        1.3176             nan     0.1000    0.0594
##      6        1.2794             nan     0.1000    0.0488
##      7        1.2469             nan     0.1000    0.0461
##      8        1.2173             nan     0.1000    0.0411
##      9        1.1910             nan     0.1000    0.0323
##     10        1.1697             nan     0.1000    0.0374
##     20        1.0261             nan     0.1000    0.0135
##     40        0.8799             nan     0.1000    0.0089
##     60        0.7804             nan     0.1000    0.0058
##     80        0.7093             nan     0.1000    0.0030
##    100        0.6624             nan     0.1000    0.0030
##    120        0.6214             nan     0.1000    0.0022
##    140        0.5908             nan     0.1000    0.0015
##    150        0.5765             nan     0.1000    0.0016
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.2020
##      2        1.4797             nan     0.1000    0.1368
##      3        1.3921             nan     0.1000    0.1065
##      4        1.3237             nan     0.1000    0.0919
##      5        1.2658             nan     0.1000    0.0762
##      6        1.2188             nan     0.1000    0.0655
##      7        1.1751             nan     0.1000    0.0525
##      8        1.1421             nan     0.1000    0.0503
##      9        1.1086             nan     0.1000    0.0439
##     10        1.0815             nan     0.1000    0.0376
##     20        0.9036             nan     0.1000    0.0147
##     40        0.7399             nan     0.1000    0.0058
##     60        0.6437             nan     0.1000    0.0054
##     80        0.5782             nan     0.1000    0.0053
##    100        0.5297             nan     0.1000    0.0015
##    120        0.4947             nan     0.1000    0.0028
##    140        0.4672             nan     0.1000    0.0018
##    150        0.4548             nan     0.1000    0.0013
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1278
##      2        1.5248             nan     0.1000    0.0863
##      3        1.4682             nan     0.1000    0.0630
##      4        1.4265             nan     0.1000    0.0508
##      5        1.3930             nan     0.1000    0.0402
##      6        1.3673             nan     0.1000    0.0346
##      7        1.3446             nan     0.1000    0.0276
##      8        1.3257             nan     0.1000    0.0302
##      9        1.3064             nan     0.1000    0.0277
##     10        1.2865             nan     0.1000    0.0245
##     20        1.1696             nan     0.1000    0.0132
##     40        1.0586             nan     0.1000    0.0039
##     60        0.9976             nan     0.1000    0.0029
##     80        0.9536             nan     0.1000    0.0036
##    100        0.9193             nan     0.1000    0.0018
##    120        0.8897             nan     0.1000    0.0010
##    140        0.8647             nan     0.1000    0.0008
##    150        0.8548             nan     0.1000    0.0008
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1713
##      2        1.4977             nan     0.1000    0.1198
##      3        1.4204             nan     0.1000    0.0839
##      4        1.3650             nan     0.1000    0.0689
##      5        1.3200             nan     0.1000    0.0610
##      6        1.2802             nan     0.1000    0.0493
##      7        1.2481             nan     0.1000    0.0420
##      8        1.2199             nan     0.1000    0.0431
##      9        1.1928             nan     0.1000    0.0326
##     10        1.1715             nan     0.1000    0.0325
##     20        1.0305             nan     0.1000    0.0135
##     40        0.8795             nan     0.1000    0.0103
##     60        0.7769             nan     0.1000    0.0062
##     80        0.7131             nan     0.1000    0.0056
##    100        0.6642             nan     0.1000    0.0032
##    120        0.6234             nan     0.1000    0.0035
##    140        0.5893             nan     0.1000    0.0016
##    150        0.5763             nan     0.1000    0.0013
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.2085
##      2        1.4732             nan     0.1000    0.1374
##      3        1.3845             nan     0.1000    0.1006
##      4        1.3189             nan     0.1000    0.0883
##      5        1.2631             nan     0.1000    0.0759
##      6        1.2128             nan     0.1000    0.0538
##      7        1.1783             nan     0.1000    0.0527
##      8        1.1453             nan     0.1000    0.0531
##      9        1.1105             nan     0.1000    0.0410
##     10        1.0846             nan     0.1000    0.0397
##     20        0.9072             nan     0.1000    0.0199
##     40        0.7404             nan     0.1000    0.0066
##     60        0.6417             nan     0.1000    0.0049
##     80        0.5756             nan     0.1000    0.0051
##    100        0.5288             nan     0.1000    0.0020
##    120        0.4926             nan     0.1000    0.0029
##    140        0.4641             nan     0.1000    0.0009
##    150        0.4532             nan     0.1000    0.0015
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1265
##      2        1.5256             nan     0.1000    0.0835
##      3        1.4697             nan     0.1000    0.0605
##      4        1.4286             nan     0.1000    0.0494
##      5        1.3959             nan     0.1000    0.0401
##      6        1.3691             nan     0.1000    0.0337
##      7        1.3463             nan     0.1000    0.0295
##      8        1.3260             nan     0.1000    0.0288
##      9        1.3073             nan     0.1000    0.0261
##     10        1.2886             nan     0.1000    0.0222
##     20        1.1709             nan     0.1000    0.0126
##     40        1.0606             nan     0.1000    0.0050
##     60        0.9970             nan     0.1000    0.0036
##     80        0.9528             nan     0.1000    0.0033
##    100        0.9199             nan     0.1000    0.0020
##    120        0.8911             nan     0.1000    0.0025
##    140        0.8652             nan     0.1000    0.0014
##    150        0.8553             nan     0.1000    0.0011
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1677
##      2        1.4975             nan     0.1000    0.1128
##      3        1.4236             nan     0.1000    0.0886
##      4        1.3658             nan     0.1000    0.0703
##      5        1.3198             nan     0.1000    0.0572
##      6        1.2827             nan     0.1000    0.0519
##      7        1.2491             nan     0.1000    0.0403
##      8        1.2230             nan     0.1000    0.0417
##      9        1.1960             nan     0.1000    0.0379
##     10        1.1721             nan     0.1000    0.0323
##     20        1.0330             nan     0.1000    0.0144
##     40        0.8841             nan     0.1000    0.0108
##     60        0.7788             nan     0.1000    0.0050
##     80        0.7182             nan     0.1000    0.0059
##    100        0.6638             nan     0.1000    0.0035
##    120        0.6219             nan     0.1000    0.0013
##    140        0.5926             nan     0.1000    0.0030
##    150        0.5785             nan     0.1000    0.0015
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.2000
##      2        1.4794             nan     0.1000    0.1376
##      3        1.3902             nan     0.1000    0.1039
##      4        1.3232             nan     0.1000    0.0898
##      5        1.2681             nan     0.1000    0.0665
##      6        1.2248             nan     0.1000    0.0626
##      7        1.1855             nan     0.1000    0.0573
##      8        1.1460             nan     0.1000    0.0519
##      9        1.1114             nan     0.1000    0.0434
##     10        1.0842             nan     0.1000    0.0355
##     20        0.9016             nan     0.1000    0.0169
##     40        0.7423             nan     0.1000    0.0084
##     60        0.6454             nan     0.1000    0.0060
##     80        0.5791             nan     0.1000    0.0031
##    100        0.5335             nan     0.1000    0.0032
##    120        0.4955             nan     0.1000    0.0022
##    140        0.4657             nan     0.1000    0.0014
##    150        0.4543             nan     0.1000    0.0011
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1282
##      2        1.5237             nan     0.1000    0.0839
##      3        1.4676             nan     0.1000    0.0653
##      4        1.4242             nan     0.1000    0.0499
##      5        1.3909             nan     0.1000    0.0393
##      6        1.3646             nan     0.1000    0.0351
##      7        1.3417             nan     0.1000    0.0282
##      8        1.3231             nan     0.1000    0.0273
##      9        1.3052             nan     0.1000    0.0271
##     10        1.2876             nan     0.1000    0.0255
##     20        1.1679             nan     0.1000    0.0120
##     40        1.0581             nan     0.1000    0.0058
##     60        0.9950             nan     0.1000    0.0033
##     80        0.9534             nan     0.1000    0.0039
##    100        0.9181             nan     0.1000    0.0005
##    120        0.8876             nan     0.1000    0.0028
##    140        0.8644             nan     0.1000    0.0007
##    150        0.8531             nan     0.1000    0.0017
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1730
##      2        1.4950             nan     0.1000    0.1157
##      3        1.4203             nan     0.1000    0.0855
##      4        1.3636             nan     0.1000    0.0685
##      5        1.3184             nan     0.1000    0.0572
##      6        1.2811             nan     0.1000    0.0535
##      7        1.2469             nan     0.1000    0.0431
##      8        1.2187             nan     0.1000    0.0407
##      9        1.1935             nan     0.1000    0.0347
##     10        1.1706             nan     0.1000    0.0326
##     20        1.0300             nan     0.1000    0.0139
##     40        0.8721             nan     0.1000    0.0099
##     60        0.7802             nan     0.1000    0.0091
##     80        0.7107             nan     0.1000    0.0056
##    100        0.6607             nan     0.1000    0.0045
##    120        0.6178             nan     0.1000    0.0017
##    140        0.5866             nan     0.1000    0.0023
##    150        0.5710             nan     0.1000    0.0008
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.2026
##      2        1.4786             nan     0.1000    0.1360
##      3        1.3915             nan     0.1000    0.1112
##      4        1.3219             nan     0.1000    0.0866
##      5        1.2672             nan     0.1000    0.0761
##      6        1.2193             nan     0.1000    0.0692
##      7        1.1739             nan     0.1000    0.0520
##      8        1.1411             nan     0.1000    0.0488
##      9        1.1086             nan     0.1000    0.0365
##     10        1.0842             nan     0.1000    0.0324
##     20        0.9062             nan     0.1000    0.0185
##     40        0.7408             nan     0.1000    0.0076
##     60        0.6407             nan     0.1000    0.0038
##     80        0.5770             nan     0.1000    0.0041
##    100        0.5293             nan     0.1000    0.0049
##    120        0.4917             nan     0.1000    0.0026
##    140        0.4637             nan     0.1000    0.0010
##    150        0.4520             nan     0.1000    0.0017
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1276
##      2        1.5237             nan     0.1000    0.0866
##      3        1.4673             nan     0.1000    0.0634
##      4        1.4252             nan     0.1000    0.0489
##      5        1.3927             nan     0.1000    0.0388
##      6        1.3666             nan     0.1000    0.0341
##      7        1.3441             nan     0.1000    0.0318
##      8        1.3234             nan     0.1000    0.0258
##      9        1.3069             nan     0.1000    0.0270
##     10        1.2880             nan     0.1000    0.0238
##     20        1.1695             nan     0.1000    0.0112
##     40        1.0578             nan     0.1000    0.0048
##     60        0.9986             nan     0.1000    0.0030
##     80        0.9537             nan     0.1000    0.0025
##    100        0.9175             nan     0.1000    0.0022
##    120        0.8894             nan     0.1000    0.0023
##    140        0.8647             nan     0.1000    0.0012
##    150        0.8541             nan     0.1000    0.0009
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1709
##      2        1.4980             nan     0.1000    0.1173
##      3        1.4229             nan     0.1000    0.0855
##      4        1.3673             nan     0.1000    0.0726
##      5        1.3213             nan     0.1000    0.0583
##      6        1.2838             nan     0.1000    0.0519
##      7        1.2498             nan     0.1000    0.0439
##      8        1.2215             nan     0.1000    0.0423
##      9        1.1941             nan     0.1000    0.0328
##     10        1.1730             nan     0.1000    0.0313
##     20        1.0309             nan     0.1000    0.0120
##     40        0.8812             nan     0.1000    0.0094
##     60        0.7800             nan     0.1000    0.0060
##     80        0.7143             nan     0.1000    0.0038
##    100        0.6629             nan     0.1000    0.0032
##    120        0.6217             nan     0.1000    0.0016
##    140        0.5895             nan     0.1000    0.0013
##    150        0.5758             nan     0.1000    0.0018
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.2066
##      2        1.4749             nan     0.1000    0.1344
##      3        1.3890             nan     0.1000    0.1135
##      4        1.3177             nan     0.1000    0.0848
##      5        1.2630             nan     0.1000    0.0684
##      6        1.2192             nan     0.1000    0.0678
##      7        1.1731             nan     0.1000    0.0503
##      8        1.1410             nan     0.1000    0.0418
##      9        1.1123             nan     0.1000    0.0416
##     10        1.0852             nan     0.1000    0.0379
##     20        0.9063             nan     0.1000    0.0206
##     40        0.7439             nan     0.1000    0.0080
##     60        0.6511             nan     0.1000    0.0067
##     80        0.5806             nan     0.1000    0.0045
##    100        0.5315             nan     0.1000    0.0030
##    120        0.4959             nan     0.1000    0.0018
##    140        0.4664             nan     0.1000    0.0014
##    150        0.4534             nan     0.1000    0.0014
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1311
##      2        1.5234             nan     0.1000    0.0868
##      3        1.4670             nan     0.1000    0.0630
##      4        1.4249             nan     0.1000    0.0514
##      5        1.3915             nan     0.1000    0.0396
##      6        1.3651             nan     0.1000    0.0339
##      7        1.3422             nan     0.1000    0.0271
##      8        1.3240             nan     0.1000    0.0298
##      9        1.3061             nan     0.1000    0.0284
##     10        1.2865             nan     0.1000    0.0249
##     20        1.1686             nan     0.1000    0.0123
##     40        1.0586             nan     0.1000    0.0062
##     60        0.9956             nan     0.1000    0.0032
##     80        0.9512             nan     0.1000    0.0024
##    100        0.9175             nan     0.1000    0.0015
##    120        0.8888             nan     0.1000    0.0016
##    140        0.8631             nan     0.1000    0.0008
##    150        0.8517             nan     0.1000    0.0008
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1702
##      2        1.4954             nan     0.1000    0.1156
##      3        1.4199             nan     0.1000    0.0866
##      4        1.3649             nan     0.1000    0.0668
##      5        1.3200             nan     0.1000    0.0622
##      6        1.2811             nan     0.1000    0.0517
##      7        1.2473             nan     0.1000    0.0469
##      8        1.2171             nan     0.1000    0.0379
##      9        1.1921             nan     0.1000    0.0356
##     10        1.1698             nan     0.1000    0.0288
##     20        1.0302             nan     0.1000    0.0137
##     40        0.8793             nan     0.1000    0.0061
##     60        0.7799             nan     0.1000    0.0085
##     80        0.7153             nan     0.1000    0.0041
##    100        0.6644             nan     0.1000    0.0032
##    120        0.6223             nan     0.1000    0.0027
##    140        0.5897             nan     0.1000    0.0010
##    150        0.5782             nan     0.1000    0.0017
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.2068
##      2        1.4770             nan     0.1000    0.1339
##      3        1.3901             nan     0.1000    0.1130
##      4        1.3192             nan     0.1000    0.0910
##      5        1.2624             nan     0.1000    0.0684
##      6        1.2183             nan     0.1000    0.0679
##      7        1.1729             nan     0.1000    0.0483
##      8        1.1419             nan     0.1000    0.0447
##      9        1.1134             nan     0.1000    0.0512
##     10        1.0807             nan     0.1000    0.0390
##     20        0.9049             nan     0.1000    0.0187
##     40        0.7395             nan     0.1000    0.0072
##     60        0.6435             nan     0.1000    0.0042
##     80        0.5802             nan     0.1000    0.0066
##    100        0.5285             nan     0.1000    0.0021
##    120        0.4945             nan     0.1000    0.0036
##    140        0.4633             nan     0.1000    0.0009
##    150        0.4504             nan     0.1000    0.0008
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1273
##      2        1.5236             nan     0.1000    0.0849
##      3        1.4674             nan     0.1000    0.0632
##      4        1.4260             nan     0.1000    0.0480
##      5        1.3934             nan     0.1000    0.0426
##      6        1.3654             nan     0.1000    0.0328
##      7        1.3438             nan     0.1000    0.0289
##      8        1.3243             nan     0.1000    0.0277
##      9        1.3062             nan     0.1000    0.0240
##     10        1.2907             nan     0.1000    0.0258
##     20        1.1707             nan     0.1000    0.0096
##     40        1.0602             nan     0.1000    0.0051
##     60        0.9977             nan     0.1000    0.0031
##     80        0.9540             nan     0.1000    0.0025
##    100        0.9197             nan     0.1000    0.0031
##    120        0.8913             nan     0.1000    0.0024
##    140        0.8667             nan     0.1000    0.0009
##    150        0.8555             nan     0.1000    0.0008
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1691
##      2        1.4985             nan     0.1000    0.1151
##      3        1.4237             nan     0.1000    0.0861
##      4        1.3677             nan     0.1000    0.0737
##      5        1.3210             nan     0.1000    0.0584
##      6        1.2831             nan     0.1000    0.0505
##      7        1.2506             nan     0.1000    0.0432
##      8        1.2223             nan     0.1000    0.0394
##      9        1.1969             nan     0.1000    0.0328
##     10        1.1758             nan     0.1000    0.0333
##     20        1.0299             nan     0.1000    0.0140
##     40        0.8820             nan     0.1000    0.0087
##     60        0.7801             nan     0.1000    0.0071
##     80        0.7120             nan     0.1000    0.0050
##    100        0.6616             nan     0.1000    0.0026
##    120        0.6221             nan     0.1000    0.0024
##    140        0.5906             nan     0.1000    0.0024
##    150        0.5768             nan     0.1000    0.0012
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.2021
##      2        1.4761             nan     0.1000    0.1367
##      3        1.3878             nan     0.1000    0.1011
##      4        1.3215             nan     0.1000    0.0825
##      5        1.2678             nan     0.1000    0.0764
##      6        1.2196             nan     0.1000    0.0694
##      7        1.1742             nan     0.1000    0.0480
##      8        1.1433             nan     0.1000    0.0514
##      9        1.1091             nan     0.1000    0.0334
##     10        1.0864             nan     0.1000    0.0470
##     20        0.9076             nan     0.1000    0.0203
##     40        0.7412             nan     0.1000    0.0061
##     60        0.6462             nan     0.1000    0.0050
##     80        0.5786             nan     0.1000    0.0038
##    100        0.5337             nan     0.1000    0.0040
##    120        0.4958             nan     0.1000    0.0019
##    140        0.4671             nan     0.1000    0.0013
##    150        0.4552             nan     0.1000    0.0012
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1285
##      2        1.5241             nan     0.1000    0.0849
##      3        1.4672             nan     0.1000    0.0630
##      4        1.4253             nan     0.1000    0.0507
##      5        1.3927             nan     0.1000    0.0394
##      6        1.3668             nan     0.1000    0.0323
##      7        1.3453             nan     0.1000    0.0323
##      8        1.3244             nan     0.1000    0.0305
##      9        1.3052             nan     0.1000    0.0224
##     10        1.2905             nan     0.1000    0.0250
##     20        1.1698             nan     0.1000    0.0124
##     40        1.0581             nan     0.1000    0.0071
##     60        0.9969             nan     0.1000    0.0051
##     80        0.9530             nan     0.1000    0.0025
##    100        0.9188             nan     0.1000    0.0029
##    120        0.8896             nan     0.1000    0.0017
##    140        0.8657             nan     0.1000    0.0013
##    150        0.8544             nan     0.1000    0.0006
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1695
##      2        1.4969             nan     0.1000    0.1114
##      3        1.4221             nan     0.1000    0.0848
##      4        1.3674             nan     0.1000    0.0753
##      5        1.3204             nan     0.1000    0.0549
##      6        1.2853             nan     0.1000    0.0542
##      7        1.2505             nan     0.1000    0.0482
##      8        1.2202             nan     0.1000    0.0416
##      9        1.1940             nan     0.1000    0.0340
##     10        1.1725             nan     0.1000    0.0311
##     20        1.0310             nan     0.1000    0.0152
##     40        0.8809             nan     0.1000    0.0100
##     60        0.7799             nan     0.1000    0.0063
##     80        0.7111             nan     0.1000    0.0036
##    100        0.6613             nan     0.1000    0.0022
##    120        0.6227             nan     0.1000    0.0032
##    140        0.5900             nan     0.1000    0.0015
##    150        0.5770             nan     0.1000    0.0012
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.2044
##      2        1.4762             nan     0.1000    0.1300
##      3        1.3920             nan     0.1000    0.1095
##      4        1.3215             nan     0.1000    0.0897
##      5        1.2651             nan     0.1000    0.0746
##      6        1.2150             nan     0.1000    0.0584
##      7        1.1781             nan     0.1000    0.0468
##      8        1.1476             nan     0.1000    0.0521
##      9        1.1132             nan     0.1000    0.0378
##     10        1.0885             nan     0.1000    0.0366
##     20        0.9082             nan     0.1000    0.0174
##     40        0.7405             nan     0.1000    0.0065
##     60        0.6406             nan     0.1000    0.0054
##     80        0.5783             nan     0.1000    0.0034
##    100        0.5262             nan     0.1000    0.0027
##    120        0.4896             nan     0.1000    0.0024
##    140        0.4619             nan     0.1000    0.0017
##    150        0.4505             nan     0.1000    0.0016
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1271
##      2        1.5243             nan     0.1000    0.0842
##      3        1.4673             nan     0.1000    0.0636
##      4        1.4250             nan     0.1000    0.0502
##      5        1.3919             nan     0.1000    0.0398
##      6        1.3653             nan     0.1000    0.0324
##      7        1.3437             nan     0.1000    0.0306
##      8        1.3241             nan     0.1000    0.0285
##      9        1.3054             nan     0.1000    0.0242
##     10        1.2897             nan     0.1000    0.0238
##     20        1.1682             nan     0.1000    0.0106
##     40        1.0588             nan     0.1000    0.0049
##     60        0.9957             nan     0.1000    0.0029
##     80        0.9505             nan     0.1000    0.0022
##    100        0.9168             nan     0.1000    0.0025
##    120        0.8874             nan     0.1000    0.0014
##    140        0.8630             nan     0.1000    0.0018
##    150        0.8522             nan     0.1000    0.0011
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1718
##      2        1.4980             nan     0.1000    0.1149
##      3        1.4227             nan     0.1000    0.0897
##      4        1.3644             nan     0.1000    0.0695
##      5        1.3189             nan     0.1000    0.0597
##      6        1.2805             nan     0.1000    0.0527
##      7        1.2468             nan     0.1000    0.0416
##      8        1.2197             nan     0.1000    0.0394
##      9        1.1937             nan     0.1000    0.0379
##     10        1.1689             nan     0.1000    0.0300
##     20        1.0280             nan     0.1000    0.0152
##     40        0.8802             nan     0.1000    0.0060
##     60        0.7803             nan     0.1000    0.0062
##     80        0.7121             nan     0.1000    0.0038
##    100        0.6601             nan     0.1000    0.0040
##    120        0.6196             nan     0.1000    0.0022
##    140        0.5896             nan     0.1000    0.0014
##    150        0.5772             nan     0.1000    0.0023
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.2036
##      2        1.4769             nan     0.1000    0.1449
##      3        1.3848             nan     0.1000    0.1041
##      4        1.3186             nan     0.1000    0.0857
##      5        1.2645             nan     0.1000    0.0710
##      6        1.2190             nan     0.1000    0.0659
##      7        1.1746             nan     0.1000    0.0553
##      8        1.1381             nan     0.1000    0.0486
##      9        1.1077             nan     0.1000    0.0378
##     10        1.0824             nan     0.1000    0.0394
##     20        0.9008             nan     0.1000    0.0257
##     40        0.7362             nan     0.1000    0.0113
##     60        0.6435             nan     0.1000    0.0058
##     80        0.5782             nan     0.1000    0.0021
##    100        0.5307             nan     0.1000    0.0021
##    120        0.4934             nan     0.1000    0.0020
##    140        0.4673             nan     0.1000    0.0015
##    150        0.4526             nan     0.1000    0.0006
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1279
##      2        1.5241             nan     0.1000    0.0869
##      3        1.4673             nan     0.1000    0.0632
##      4        1.4256             nan     0.1000    0.0487
##      5        1.3923             nan     0.1000    0.0409
##      6        1.3653             nan     0.1000    0.0328
##      7        1.3431             nan     0.1000    0.0276
##      8        1.3243             nan     0.1000    0.0282
##      9        1.3057             nan     0.1000    0.0255
##     10        1.2893             nan     0.1000    0.0243
##     20        1.1686             nan     0.1000    0.0127
##     40        1.0603             nan     0.1000    0.0060
##     60        0.9975             nan     0.1000    0.0035
##     80        0.9535             nan     0.1000    0.0021
##    100        0.9199             nan     0.1000    0.0031
##    120        0.8912             nan     0.1000    0.0008
##    140        0.8655             nan     0.1000    0.0007
##    150        0.8549             nan     0.1000    0.0008
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1710
##      2        1.4982             nan     0.1000    0.1190
##      3        1.4200             nan     0.1000    0.0864
##      4        1.3640             nan     0.1000    0.0678
##      5        1.3186             nan     0.1000    0.0585
##      6        1.2807             nan     0.1000    0.0519
##      7        1.2477             nan     0.1000    0.0438
##      8        1.2188             nan     0.1000    0.0386
##      9        1.1943             nan     0.1000    0.0386
##     10        1.1698             nan     0.1000    0.0322
##     20        1.0313             nan     0.1000    0.0157
##     40        0.8855             nan     0.1000    0.0211
##     60        0.7787             nan     0.1000    0.0081
##     80        0.7136             nan     0.1000    0.0033
##    100        0.6650             nan     0.1000    0.0048
##    120        0.6266             nan     0.1000    0.0018
##    140        0.5938             nan     0.1000    0.0021
##    150        0.5800             nan     0.1000    0.0010
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.2013
##      2        1.4783             nan     0.1000    0.1438
##      3        1.3856             nan     0.1000    0.1077
##      4        1.3169             nan     0.1000    0.0864
##      5        1.2614             nan     0.1000    0.0679
##      6        1.2175             nan     0.1000    0.0665
##      7        1.1726             nan     0.1000    0.0534
##      8        1.1361             nan     0.1000    0.0469
##      9        1.1068             nan     0.1000    0.0378
##     10        1.0828             nan     0.1000    0.0417
##     20        0.9040             nan     0.1000    0.0187
##     40        0.7406             nan     0.1000    0.0076
##     60        0.6452             nan     0.1000    0.0040
##     80        0.5846             nan     0.1000    0.0033
##    100        0.5364             nan     0.1000    0.0018
##    120        0.5021             nan     0.1000    0.0020
##    140        0.4699             nan     0.1000    0.0017
##    150        0.4557             nan     0.1000    0.0006
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1274
##      2        1.5231             nan     0.1000    0.0873
##      3        1.4656             nan     0.1000    0.0637
##      4        1.4224             nan     0.1000    0.0497
##      5        1.3900             nan     0.1000    0.0405
##      6        1.3632             nan     0.1000    0.0335
##      7        1.3409             nan     0.1000    0.0307
##      8        1.3207             nan     0.1000    0.0263
##      9        1.3032             nan     0.1000    0.0253
##     10        1.2868             nan     0.1000    0.0260
##     20        1.1646             nan     0.1000    0.0115
##     40        1.0555             nan     0.1000    0.0054
##     60        0.9937             nan     0.1000    0.0040
##     80        0.9503             nan     0.1000    0.0024
##    100        0.9160             nan     0.1000    0.0027
##    120        0.8868             nan     0.1000    0.0020
##    140        0.8631             nan     0.1000    0.0014
##    150        0.8522             nan     0.1000    0.0009
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1741
##      2        1.4970             nan     0.1000    0.1140
##      3        1.4221             nan     0.1000    0.0902
##      4        1.3638             nan     0.1000    0.0718
##      5        1.3179             nan     0.1000    0.0576
##      6        1.2807             nan     0.1000    0.0529
##      7        1.2474             nan     0.1000    0.0441
##      8        1.2192             nan     0.1000    0.0397
##      9        1.1935             nan     0.1000    0.0360
##     10        1.1702             nan     0.1000    0.0291
##     20        1.0302             nan     0.1000    0.0134
##     40        0.8817             nan     0.1000    0.0139
##     60        0.7752             nan     0.1000    0.0049
##     80        0.7098             nan     0.1000    0.0028
##    100        0.6628             nan     0.1000    0.0048
##    120        0.6245             nan     0.1000    0.0023
##    140        0.5918             nan     0.1000    0.0017
##    150        0.5788             nan     0.1000    0.0024
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.2054
##      2        1.4769             nan     0.1000    0.1456
##      3        1.3846             nan     0.1000    0.1034
##      4        1.3190             nan     0.1000    0.0862
##      5        1.2641             nan     0.1000    0.0676
##      6        1.2212             nan     0.1000    0.0660
##      7        1.1805             nan     0.1000    0.0579
##      8        1.1412             nan     0.1000    0.0446
##      9        1.1130             nan     0.1000    0.0355
##     10        1.0898             nan     0.1000    0.0431
##     20        0.9078             nan     0.1000    0.0157
##     40        0.7391             nan     0.1000    0.0078
##     60        0.6429             nan     0.1000    0.0043
##     80        0.5813             nan     0.1000    0.0030
##    100        0.5357             nan     0.1000    0.0044
##    120        0.4996             nan     0.1000    0.0018
##    140        0.4672             nan     0.1000    0.0013
##    150        0.4555             nan     0.1000    0.0008
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1273
##      2        1.5242             nan     0.1000    0.0835
##      3        1.4687             nan     0.1000    0.0640
##      4        1.4271             nan     0.1000    0.0503
##      5        1.3935             nan     0.1000    0.0381
##      6        1.3678             nan     0.1000    0.0336
##      7        1.3455             nan     0.1000    0.0306
##      8        1.3253             nan     0.1000    0.0259
##      9        1.3084             nan     0.1000    0.0279
##     10        1.2891             nan     0.1000    0.0236
##     20        1.1718             nan     0.1000    0.0106
##     40        1.0616             nan     0.1000    0.0062
##     60        0.9979             nan     0.1000    0.0030
##     80        0.9532             nan     0.1000    0.0020
##    100        0.9193             nan     0.1000    0.0013
##    120        0.8905             nan     0.1000    0.0020
##    140        0.8650             nan     0.1000    0.0011
##    150        0.8540             nan     0.1000    0.0011
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1703
##      2        1.4979             nan     0.1000    0.1190
##      3        1.4206             nan     0.1000    0.0858
##      4        1.3652             nan     0.1000    0.0729
##      5        1.3191             nan     0.1000    0.0544
##      6        1.2834             nan     0.1000    0.0496
##      7        1.2509             nan     0.1000    0.0446
##      8        1.2227             nan     0.1000    0.0436
##      9        1.1948             nan     0.1000    0.0362
##     10        1.1711             nan     0.1000    0.0317
##     20        1.0312             nan     0.1000    0.0111
##     40        0.8787             nan     0.1000    0.0088
##     60        0.7841             nan     0.1000    0.0089
##     80        0.7152             nan     0.1000    0.0034
##    100        0.6639             nan     0.1000    0.0034
##    120        0.6246             nan     0.1000    0.0024
##    140        0.5922             nan     0.1000    0.0032
##    150        0.5785             nan     0.1000    0.0019
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.2023
##      2        1.4774             nan     0.1000    0.1421
##      3        1.3883             nan     0.1000    0.1046
##      4        1.3223             nan     0.1000    0.0909
##      5        1.2654             nan     0.1000    0.0753
##      6        1.2153             nan     0.1000    0.0625
##      7        1.1770             nan     0.1000    0.0506
##      8        1.1445             nan     0.1000    0.0461
##      9        1.1162             nan     0.1000    0.0425
##     10        1.0893             nan     0.1000    0.0404
##     20        0.9120             nan     0.1000    0.0256
##     40        0.7467             nan     0.1000    0.0094
##     60        0.6478             nan     0.1000    0.0086
##     80        0.5782             nan     0.1000    0.0043
##    100        0.5309             nan     0.1000    0.0032
##    120        0.4926             nan     0.1000    0.0018
##    140        0.4656             nan     0.1000    0.0031
##    150        0.4533             nan     0.1000    0.0013
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1281
##      2        1.5237             nan     0.1000    0.0863
##      3        1.4664             nan     0.1000    0.0633
##      4        1.4243             nan     0.1000    0.0498
##      5        1.3922             nan     0.1000    0.0398
##      6        1.3664             nan     0.1000    0.0345
##      7        1.3438             nan     0.1000    0.0301
##      8        1.3236             nan     0.1000    0.0266
##      9        1.3064             nan     0.1000    0.0234
##     10        1.2908             nan     0.1000    0.0266
##     20        1.1686             nan     0.1000    0.0119
##     40        1.0594             nan     0.1000    0.0051
##     60        0.9965             nan     0.1000    0.0053
##     80        0.9518             nan     0.1000    0.0027
##    100        0.9178             nan     0.1000    0.0021
##    120        0.8876             nan     0.1000    0.0012
##    140        0.8634             nan     0.1000    0.0014
##    150        0.8519             nan     0.1000    0.0010
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1709
##      2        1.4979             nan     0.1000    0.1159
##      3        1.4234             nan     0.1000    0.0837
##      4        1.3684             nan     0.1000    0.0710
##      5        1.3225             nan     0.1000    0.0589
##      6        1.2841             nan     0.1000    0.0487
##      7        1.2520             nan     0.1000    0.0477
##      8        1.2220             nan     0.1000    0.0415
##      9        1.1950             nan     0.1000    0.0335
##     10        1.1738             nan     0.1000    0.0292
##     20        1.0332             nan     0.1000    0.0159
##     40        0.8808             nan     0.1000    0.0138
##     60        0.7774             nan     0.1000    0.0048
##     80        0.7095             nan     0.1000    0.0026
##    100        0.6627             nan     0.1000    0.0019
##    120        0.6236             nan     0.1000    0.0026
##    140        0.5912             nan     0.1000    0.0008
##    150        0.5784             nan     0.1000    0.0009
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.2064
##      2        1.4785             nan     0.1000    0.1411
##      3        1.3891             nan     0.1000    0.1089
##      4        1.3195             nan     0.1000    0.0900
##      5        1.2633             nan     0.1000    0.0789
##      6        1.2112             nan     0.1000    0.0586
##      7        1.1741             nan     0.1000    0.0533
##      8        1.1387             nan     0.1000    0.0416
##      9        1.1108             nan     0.1000    0.0453
##     10        1.0813             nan     0.1000    0.0351
##     20        0.9074             nan     0.1000    0.0196
##     40        0.7365             nan     0.1000    0.0050
##     60        0.6439             nan     0.1000    0.0045
##     80        0.5787             nan     0.1000    0.0037
##    100        0.5292             nan     0.1000    0.0015
##    120        0.4960             nan     0.1000    0.0010
##    140        0.4659             nan     0.1000    0.0015
##    150        0.4530             nan     0.1000    0.0011
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1262
##      2        1.5253             nan     0.1000    0.0861
##      3        1.4697             nan     0.1000    0.0648
##      4        1.4273             nan     0.1000    0.0502
##      5        1.3949             nan     0.1000    0.0396
##      6        1.3683             nan     0.1000    0.0347
##      7        1.3458             nan     0.1000    0.0312
##      8        1.3254             nan     0.1000    0.0269
##      9        1.3079             nan     0.1000    0.0254
##     10        1.2896             nan     0.1000    0.0240
##     20        1.1719             nan     0.1000    0.0122
##     40        1.0621             nan     0.1000    0.0057
##     60        0.9982             nan     0.1000    0.0048
##     80        0.9536             nan     0.1000    0.0023
##    100        0.9201             nan     0.1000    0.0022
##    120        0.8910             nan     0.1000    0.0007
##    140        0.8658             nan     0.1000    0.0012
##    150        0.8552             nan     0.1000    0.0014
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1664
##      2        1.4987             nan     0.1000    0.1165
##      3        1.4225             nan     0.1000    0.0882
##      4        1.3657             nan     0.1000    0.0693
##      5        1.3203             nan     0.1000    0.0570
##      6        1.2836             nan     0.1000    0.0516
##      7        1.2505             nan     0.1000    0.0510
##      8        1.2181             nan     0.1000    0.0370
##      9        1.1937             nan     0.1000    0.0357
##     10        1.1710             nan     0.1000    0.0283
##     20        1.0343             nan     0.1000    0.0150
##     40        0.8856             nan     0.1000    0.0110
##     60        0.7813             nan     0.1000    0.0078
##     80        0.7118             nan     0.1000    0.0052
##    100        0.6615             nan     0.1000    0.0036
##    120        0.6202             nan     0.1000    0.0021
##    140        0.5890             nan     0.1000    0.0012
##    150        0.5764             nan     0.1000    0.0019
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1997
##      2        1.4782             nan     0.1000    0.1415
##      3        1.3882             nan     0.1000    0.1071
##      4        1.3190             nan     0.1000    0.0876
##      5        1.2632             nan     0.1000    0.0699
##      6        1.2205             nan     0.1000    0.0607
##      7        1.1810             nan     0.1000    0.0602
##      8        1.1403             nan     0.1000    0.0447
##      9        1.1100             nan     0.1000    0.0396
##     10        1.0849             nan     0.1000    0.0341
##     20        0.9125             nan     0.1000    0.0184
##     40        0.7477             nan     0.1000    0.0084
##     60        0.6489             nan     0.1000    0.0078
##     80        0.5786             nan     0.1000    0.0030
##    100        0.5333             nan     0.1000    0.0010
##    120        0.4984             nan     0.1000    0.0024
##    140        0.4675             nan     0.1000    0.0019
##    150        0.4552             nan     0.1000    0.0009
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1283
##      2        1.5237             nan     0.1000    0.0841
##      3        1.4676             nan     0.1000    0.0643
##      4        1.4253             nan     0.1000    0.0497
##      5        1.3920             nan     0.1000    0.0385
##      6        1.3658             nan     0.1000    0.0331
##      7        1.3439             nan     0.1000    0.0281
##      8        1.3252             nan     0.1000    0.0257
##      9        1.3079             nan     0.1000    0.0256
##     10        1.2911             nan     0.1000    0.0281
##     20        1.1684             nan     0.1000    0.0124
##     40        1.0594             nan     0.1000    0.0069
##     60        0.9978             nan     0.1000    0.0043
##     80        0.9528             nan     0.1000    0.0021
##    100        0.9194             nan     0.1000    0.0020
##    120        0.8899             nan     0.1000    0.0014
##    140        0.8653             nan     0.1000    0.0021
##    150        0.8541             nan     0.1000    0.0015
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1727
##      2        1.4972             nan     0.1000    0.1206
##      3        1.4197             nan     0.1000    0.0842
##      4        1.3648             nan     0.1000    0.0688
##      5        1.3195             nan     0.1000    0.0589
##      6        1.2823             nan     0.1000    0.0489
##      7        1.2503             nan     0.1000    0.0492
##      8        1.2198             nan     0.1000    0.0389
##      9        1.1945             nan     0.1000    0.0369
##     10        1.1710             nan     0.1000    0.0321
##     20        1.0284             nan     0.1000    0.0181
##     40        0.8730             nan     0.1000    0.0063
##     60        0.7780             nan     0.1000    0.0067
##     80        0.7137             nan     0.1000    0.0037
##    100        0.6636             nan     0.1000    0.0036
##    120        0.6234             nan     0.1000    0.0023
##    140        0.5913             nan     0.1000    0.0010
##    150        0.5763             nan     0.1000    0.0016
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1991
##      2        1.4798             nan     0.1000    0.1456
##      3        1.3877             nan     0.1000    0.1065
##      4        1.3213             nan     0.1000    0.0813
##      5        1.2690             nan     0.1000    0.0760
##      6        1.2225             nan     0.1000    0.0673
##      7        1.1782             nan     0.1000    0.0533
##      8        1.1439             nan     0.1000    0.0413
##      9        1.1168             nan     0.1000    0.0372
##     10        1.0906             nan     0.1000    0.0463
##     20        0.9051             nan     0.1000    0.0151
##     40        0.7411             nan     0.1000    0.0080
##     60        0.6439             nan     0.1000    0.0042
##     80        0.5778             nan     0.1000    0.0042
##    100        0.5309             nan     0.1000    0.0021
##    120        0.4964             nan     0.1000    0.0027
##    140        0.4662             nan     0.1000    0.0005
##    150        0.4532             nan     0.1000    0.0010
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1293
##      2        1.5241             nan     0.1000    0.0859
##      3        1.4675             nan     0.1000    0.0605
##      4        1.4265             nan     0.1000    0.0522
##      5        1.3926             nan     0.1000    0.0394
##      6        1.3664             nan     0.1000    0.0349
##      7        1.3439             nan     0.1000    0.0307
##      8        1.3240             nan     0.1000    0.0263
##      9        1.3068             nan     0.1000    0.0274
##     10        1.2874             nan     0.1000    0.0249
##     20        1.1692             nan     0.1000    0.0116
##     40        1.0584             nan     0.1000    0.0050
##     60        0.9932             nan     0.1000    0.0032
##     80        0.9518             nan     0.1000    0.0025
##    100        0.9167             nan     0.1000    0.0021
##    120        0.8876             nan     0.1000    0.0010
##    140        0.8634             nan     0.1000    0.0012
##    150        0.8526             nan     0.1000    0.0013
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1734
##      2        1.4979             nan     0.1000    0.1146
##      3        1.4232             nan     0.1000    0.0855
##      4        1.3675             nan     0.1000    0.0709
##      5        1.3217             nan     0.1000    0.0605
##      6        1.2833             nan     0.1000    0.0506
##      7        1.2504             nan     0.1000    0.0449
##      8        1.2216             nan     0.1000    0.0455
##      9        1.1927             nan     0.1000    0.0342
##     10        1.1710             nan     0.1000    0.0340
##     20        1.0291             nan     0.1000    0.0145
##     40        0.8785             nan     0.1000    0.0089
##     60        0.7762             nan     0.1000    0.0055
##     80        0.7074             nan     0.1000    0.0031
##    100        0.6550             nan     0.1000    0.0035
##    120        0.6158             nan     0.1000    0.0016
##    140        0.5840             nan     0.1000    0.0018
##    150        0.5716             nan     0.1000    0.0009
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.2032
##      2        1.4786             nan     0.1000    0.1472
##      3        1.3853             nan     0.1000    0.1008
##      4        1.3207             nan     0.1000    0.0866
##      5        1.2645             nan     0.1000    0.0703
##      6        1.2203             nan     0.1000    0.0671
##      7        1.1758             nan     0.1000    0.0522
##      8        1.1422             nan     0.1000    0.0504
##      9        1.1082             nan     0.1000    0.0381
##     10        1.0838             nan     0.1000    0.0426
##     20        0.9030             nan     0.1000    0.0179
##     40        0.7350             nan     0.1000    0.0074
##     60        0.6424             nan     0.1000    0.0067
##     80        0.5774             nan     0.1000    0.0041
##    100        0.5326             nan     0.1000    0.0030
##    120        0.4945             nan     0.1000    0.0021
##    140        0.4658             nan     0.1000    0.0010
##    150        0.4528             nan     0.1000    0.0018
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1276
##      2        1.5244             nan     0.1000    0.0849
##      3        1.4676             nan     0.1000    0.0625
##      4        1.4256             nan     0.1000    0.0497
##      5        1.3931             nan     0.1000    0.0404
##      6        1.3670             nan     0.1000    0.0317
##      7        1.3457             nan     0.1000    0.0315
##      8        1.3250             nan     0.1000    0.0253
##      9        1.3086             nan     0.1000    0.0293
##     10        1.2885             nan     0.1000    0.0250
##     20        1.1700             nan     0.1000    0.0122
##     40        1.0596             nan     0.1000    0.0045
##     60        0.9973             nan     0.1000    0.0031
##     80        0.9537             nan     0.1000    0.0033
##    100        0.9203             nan     0.1000    0.0020
##    120        0.8910             nan     0.1000    0.0015
##    140        0.8654             nan     0.1000    0.0011
##    150        0.8538             nan     0.1000    0.0013
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1707
##      2        1.4988             nan     0.1000    0.1162
##      3        1.4237             nan     0.1000    0.0904
##      4        1.3667             nan     0.1000    0.0671
##      5        1.3218             nan     0.1000    0.0630
##      6        1.2828             nan     0.1000    0.0487
##      7        1.2509             nan     0.1000    0.0471
##      8        1.2201             nan     0.1000    0.0413
##      9        1.1943             nan     0.1000    0.0342
##     10        1.1717             nan     0.1000    0.0334
##     20        1.0280             nan     0.1000    0.0138
##     40        0.8804             nan     0.1000    0.0140
##     60        0.7785             nan     0.1000    0.0082
##     80        0.7128             nan     0.1000    0.0055
##    100        0.6622             nan     0.1000    0.0024
##    120        0.6212             nan     0.1000    0.0021
##    140        0.5891             nan     0.1000    0.0013
##    150        0.5756             nan     0.1000    0.0026
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1998
##      2        1.4802             nan     0.1000    0.1473
##      3        1.3881             nan     0.1000    0.1041
##      4        1.3226             nan     0.1000    0.0937
##      5        1.2653             nan     0.1000    0.0809
##      6        1.2123             nan     0.1000    0.0573
##      7        1.1753             nan     0.1000    0.0494
##      8        1.1433             nan     0.1000    0.0469
##      9        1.1117             nan     0.1000    0.0434
##     10        1.0840             nan     0.1000    0.0424
##     20        0.8973             nan     0.1000    0.0187
##     40        0.7347             nan     0.1000    0.0089
##     60        0.6451             nan     0.1000    0.0063
##     80        0.5783             nan     0.1000    0.0035
##    100        0.5306             nan     0.1000    0.0038
##    120        0.4943             nan     0.1000    0.0028
##    140        0.4655             nan     0.1000    0.0011
##    150        0.4540             nan     0.1000    0.0008
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.2093
##      2        1.4768             nan     0.1000    0.1368
##      3        1.3896             nan     0.1000    0.1052
##      4        1.3220             nan     0.1000    0.0863
##      5        1.2674             nan     0.1000    0.0722
##      6        1.2216             nan     0.1000    0.0672
##      7        1.1758             nan     0.1000    0.0507
##      8        1.1435             nan     0.1000    0.0431
##      9        1.1150             nan     0.1000    0.0472
##     10        1.0841             nan     0.1000    0.0399
##     20        0.9089             nan     0.1000    0.0219
##     40        0.7463             nan     0.1000    0.0110
##     60        0.6493             nan     0.1000    0.0053
##     80        0.5845             nan     0.1000    0.0043
##    100        0.5355             nan     0.1000    0.0017
##    120        0.4981             nan     0.1000    0.0023
##    140        0.4661             nan     0.1000    0.0012
##    150        0.4537             nan     0.1000    0.0005
```

```r
# collect resamples
results <- resamples(list(RF=train.model.rf, GBM=train.model.gbm))
# summarize the distributions
summary(results)
```

```
## 
## Call:
## summary.resamples(object = results)
## 
## Models: RF, GBM 
## Number of resamples: 30 
## 
## Accuracy 
##          Min.   1st Qu.    Median      Mean   3rd Qu.      Max. NA's
## RF  0.9184090 0.9253693 0.9301735 0.9298411 0.9346414 0.9424057    0
## GBM 0.8273051 0.8376416 0.8433120 0.8418785 0.8467218 0.8527013    0
## 
## Kappa 
##          Min.   1st Qu.    Median      Mean   3rd Qu.      Max. NA's
## RF  0.8967258 0.9056170 0.9117173 0.9112467 0.9173342 0.9271318    0
## GBM 0.7813589 0.7945683 0.8017709 0.7998676 0.8059681 0.8135898    0
```

```r
# boxplots of results
bwplot(results)
```

![](MachLearnProject_files/figure-html/ModSelect-1.png)<!-- -->

```r
rf.pred <- predict(train.model.rf, validation)
confusionMatrix(rf.pred, validation$classe)$overall[1]
```

```
##  Accuracy 
## 0.9932708
```


[^1]: Velloso, E.; Bulling, A.; Gellersen, H.; Ugulino, W.; Fuks, H. Qualitative Activity Recognition of Weight Lifting Exercises. Proceedings of 4th International Conference in Cooperation with SIGCHI (Augmented Human '13) . Stuttgart, Germany: ACM SIGCHI, 2013.

