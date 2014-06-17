Prediction of Activity Classes from the Weight Lifting Exercise Dataset (HAR)
========================================================

The Weight Lifting Exercise Dataset is used for Human Activity Recognition (HAR) 
and contains data from accelerometeres attached to the test subjects forearm, 
arm, and belt. The subjects were asked to perform dumbbell curls in five different
manners: exactly according to the specification (Class A), throwing the elbows to the front (Class B), lifting the dumbbell only halfway (Class C), lowering the dumbbell only halfway (Class D) and throwing the hips to the front (Class E). This information is contained in the `classe` variable of the dataset.

Goal of this analysis is to predict the class from the available data. The data 
was downloaded from the [Coursera page](https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv). 

Prerequisites
------------------------------------------------------------

In order to perform the analysis, some `R` libraries need to be loaded. Most of the
analyses are carried out with the `caret` package:


```r
library(caret)
```

```
## Loading required package: lattice
## Loading required package: ggplot2
```

```r
library(randomForest)
```

```
## randomForest 4.6-7
## Type rfNews() to see new features/changes/bug fixes.
```

```r
library(nnet)
library(e1071)
```


Preprocessing the data
------------------------------------------------------------

First, the downloaded data is read into `R` via the `read.csv()` function:


```r
path.raw <- "C:/data-sync/coursera/data-science-08-machine-learning/assignment01/"
path.dat <- paste0(path.raw, "")
setwd(path.dat)
dat.all.raw <- read.csv("pml-training.csv")
```


The data seems to contain individual measurements as well as summary measurements,
as indicated by the fact that some variables are missing when the variable `new_window`
has the value `"no"` (not shown here for brevity). These missing variables relate to 
skew and kurtosis of measurements, which can only be calulated as a summary 
on multiple measurements. These measurements that are summary measures (as indicated by `new_window=="yes"`) are removed from the data.


```r
dat.all <- subset(dat.all.raw, dat.all.raw$new_window != "yes")
```



```r
## vars:
vars.all <- c("classe", "roll_belt", "pitch_belt", "yaw_belt", "total_accel_belt", 
    "gyros_belt_x", "gyros_belt_y", "gyros_belt_z", "accel_belt_x", "accel_belt_y", 
    "accel_belt_z", "magnet_belt_x", "magnet_belt_y", "magnet_belt_z", "roll_arm", 
    "pitch_arm", "yaw_arm", "total_accel_arm", "gyros_arm_x", "gyros_arm_y", 
    "gyros_arm_z", "accel_arm_x", "accel_arm_y", "accel_arm_z", "magnet_arm_x", 
    "magnet_arm_y", "magnet_arm_z", "roll_dumbbell", "pitch_dumbbell", "yaw_dumbbell", 
    "gyros_dumbbell_x", "gyros_dumbbell_y", "gyros_dumbbell_z", "accel_dumbbell_x", 
    "accel_dumbbell_y", "accel_dumbbell_z", "magnet_dumbbell_x", "magnet_dumbbell_y", 
    "magnet_dumbbell_z", "roll_forearm", "pitch_forearm", "yaw_forearm", "total_accel_forearm", 
    "gyros_forearm_x", "gyros_forearm_y", "gyros_forearm_z", "accel_forearm_x", 
    "accel_forearm_y", "accel_forearm_z", "magnet_forearm_x", "magnet_forearm_y", 
    "magnet_forearm_z")

dat.all.clean <- dat.all[vars.all]
```


Splitting the data in training, validation, and test set
------------------------------------------------------------

In order to train the machine learning models, the data is split into three parts. 
The training set includes a third of the data, a validation set consists of 
another third, and finally a test set consisting of the final third of the data. 
The validation set will 
be used to evaluate the out-of-sample error of the machine learning models and 
to train a stacked model.

The fraction of a third of the data for the training set was chosen
due to the fact that computing the models (especially the random forest models,
see below) takes quite some time on my machine (about 30 min).


```r
set.seed(42)
wch.fold <- createFolds(dat.all.clean$classe, k = 3, list = FALSE)
dat.train <- dat.all.clean[wch.fold == 1, ]
dat.val <- dat.all.clean[wch.fold == 2, ]
dat.test <- dat.all.clean[wch.fold == 3, ]
```


Training individual models
------------------------------------------------------------

This section describes the different models that were used. All of the models
were trained on the training data set, and then used to predict the classes
of the validation data set to get a first hint at the out-of-sample error. 

### Random forest model

Random forest models are widely used in machine learning because they often 
give very good out-of-the box predictions without knowing much about the data,
as it is not influenced by outliers, and also takes into account 
interactions by design.
Hence, a random forest model is used as a starting point here. 

It is fitted with the standard options of the `caret` package, which uses
25 iterations of bootstrapping within the training data set to select 
the best model. 


```r
set.seed(12)
caret.rf <- train(classe ~ roll_belt + pitch_belt + yaw_belt + total_accel_belt + 
    gyros_belt_x + gyros_belt_y + gyros_belt_z + accel_belt_x + accel_belt_y + 
    accel_belt_z + magnet_belt_x + magnet_belt_y + magnet_belt_z + roll_arm + 
    pitch_arm + yaw_arm + total_accel_arm + gyros_arm_x + gyros_arm_y + gyros_arm_z + 
    accel_arm_x + accel_arm_y + accel_arm_z + magnet_arm_x + magnet_arm_y + 
    magnet_arm_z + roll_dumbbell + pitch_dumbbell + yaw_dumbbell + gyros_dumbbell_x + 
    gyros_dumbbell_y + gyros_dumbbell_z + accel_dumbbell_x + accel_dumbbell_y + 
    accel_dumbbell_z + magnet_dumbbell_x + magnet_dumbbell_y + magnet_dumbbell_z + 
    roll_forearm + pitch_forearm + yaw_forearm + total_accel_forearm + gyros_forearm_x + 
    gyros_forearm_y + gyros_forearm_z + accel_forearm_x + accel_forearm_y + 
    accel_forearm_z + magnet_forearm_x + magnet_forearm_y + magnet_forearm_z, 
    data = dat.train, method = "rf")
```

```
## Warning: package 'e1071' was built under R version 3.0.3
```


The training time of this algorithm is actually quite long, on my machine this
took 33 minutes. After training the model, we can predict the classes in the
training data set and in the validation data set:


```r
pred.caret.rf.train <- predict(caret.rf$finalModel)
pred.caret.rf.val <- predict(caret.rf$finalModel, newdata = dat.val)
```


The performance in the training set is really good:


```r
confusionMatrix(pred.caret.rf.train, dat.train$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1817   24    0    0    0
##          B    3 1201   21    1    4
##          C    2   12 1090   21    3
##          D    0    0    6 1026    6
##          E    1    2    0    1 1163
## 
## Overall Statistics
##                                        
##                Accuracy : 0.983        
##                  95% CI : (0.98, 0.986)
##     No Information Rate : 0.285        
##     P-Value [Acc > NIR] : <2e-16       
##                                        
##                   Kappa : 0.979        
##  Mcnemar's Test P-Value : NA           
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             0.997    0.969    0.976    0.978    0.989
## Specificity             0.995    0.994    0.993    0.998    0.999
## Pos Pred Value          0.987    0.976    0.966    0.988    0.997
## Neg Pred Value          0.999    0.993    0.995    0.996    0.998
## Prevalence              0.285    0.193    0.174    0.164    0.184
## Detection Rate          0.284    0.188    0.170    0.160    0.182
## Detection Prevalence    0.287    0.192    0.176    0.162    0.182
## Balanced Accuracy       0.996    0.982    0.984    0.988    0.994
```


Actually, the performance is so good that it seems likely we have overfitted
our model to the data. To test this assumption, we check on the performance
of the model in the validation data set.


```r
confusionMatrix(pred.caret.rf.val, dat.val$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1809   16    0    0    0
##          B    3 1206   16    1    2
##          C   10   18 1100   23    5
##          D    1    0    1 1024    5
##          E    1    0    0    1 1164
## 
## Overall Statistics
##                                         
##                Accuracy : 0.984         
##                  95% CI : (0.981, 0.987)
##     No Information Rate : 0.285         
##     P-Value [Acc > NIR] : < 2e-16       
##                                         
##                   Kappa : 0.98          
##  Mcnemar's Test P-Value : 1.22e-07      
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             0.992    0.973    0.985    0.976    0.990
## Specificity             0.997    0.996    0.989    0.999    1.000
## Pos Pred Value          0.991    0.982    0.952    0.993    0.998
## Neg Pred Value          0.997    0.993    0.997    0.995    0.998
## Prevalence              0.285    0.194    0.174    0.164    0.184
## Detection Rate          0.282    0.188    0.172    0.160    0.182
## Detection Prevalence    0.285    0.192    0.180    0.161    0.182
## Balanced Accuracy       0.994    0.984    0.987    0.987    0.995
```


Given that the performance of the prediction algorithm remains almost unchanged
in the validation data set, we seem not to have overfitted the data. 

Regarding the variable importance gives some clue to which variables 
are most important:


```r
varImp(caret.rf)
```

```
## rf variable importance
## 
##   only 20 most important variables shown (out of 51)
## 
##                   Overall
## roll_belt          100.00
## pitch_forearm       64.13
## yaw_belt            60.32
## magnet_dumbbell_z   50.53
## pitch_belt          44.84
## magnet_dumbbell_y   42.91
## roll_forearm        36.30
## accel_dumbbell_y    25.60
## roll_dumbbell       18.96
## accel_forearm_x     17.84
## magnet_dumbbell_x   17.46
## accel_dumbbell_z    15.64
## magnet_belt_z       15.49
## accel_belt_z        15.27
## magnet_belt_y       14.09
## magnet_forearm_z    13.54
## yaw_arm             13.29
## gyros_belt_z        11.08
## magnet_belt_x       10.79
## gyros_dumbbell_y     9.57
```


The roll sensor from the belt is by far the most important sensor regarding
to the random forest model, followed by the pitch sensor on the forearm and the 
yaw sensor on the belt.

There performance of the random forest model actually is so good that
training additional models (stacking,
ensemble learning) is probably superfluous, but just for the sake of practice,
I will also fit two more models.

### Multinomial regression model

A multinomial regression model is much more rigorous and doesn't take into
account interactions between variables by default. Including all interactions
in the model formula resulted in the algorithm being unable to calculate
initial values for the parameters, so we will also fit a model with only 
main effects here.


```r
set.seed(21)
caret.mn <- train(classe ~ roll_belt + pitch_belt + yaw_belt + total_accel_belt + 
    gyros_belt_x + gyros_belt_y + gyros_belt_z + accel_belt_x + accel_belt_y + 
    accel_belt_z + magnet_belt_x + magnet_belt_y + magnet_belt_z + roll_arm + 
    pitch_arm + yaw_arm + total_accel_arm + gyros_arm_x + gyros_arm_y + gyros_arm_z + 
    accel_arm_x + accel_arm_y + accel_arm_z + magnet_arm_x + magnet_arm_y + 
    magnet_arm_z + roll_dumbbell + pitch_dumbbell + yaw_dumbbell + gyros_dumbbell_x + 
    gyros_dumbbell_y + gyros_dumbbell_z + accel_dumbbell_x + accel_dumbbell_y + 
    accel_dumbbell_z + magnet_dumbbell_x + magnet_dumbbell_y + magnet_dumbbell_z + 
    roll_forearm + pitch_forearm + yaw_forearm + total_accel_forearm + gyros_forearm_x + 
    gyros_forearm_y + gyros_forearm_z + accel_forearm_x + accel_forearm_y + 
    accel_forearm_z + magnet_forearm_x + magnet_forearm_y + magnet_forearm_z, 
    data = dat.train, method = "multinom", trace = FALSE)
```


Training is much faster (time: 5 minutes on my machine). After we have the model,
we can predict how the dumbbel curls were performed (classes)...


```r
pred.caret.mn.train <- predict(caret.mn$finalModel)
pred.caret.mn.val <- predict(caret.mn$finalModel, newdata = dat.val)
```


... and inspect predcition accuracy in the training data:


```r
confusionMatrix(pred.caret.mn.train, dat.train$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1452  204  111   97  105
##          B   63  562   67    9  100
##          C  165  227  685  184   93
##          D   79   32  144  645   91
##          E   64  214  110  114  787
## 
## Overall Statistics
##                                         
##                Accuracy : 0.645         
##                  95% CI : (0.633, 0.657)
##     No Information Rate : 0.285         
##     P-Value [Acc > NIR] : <2e-16        
##                                         
##                   Kappa : 0.55          
##  Mcnemar's Test P-Value : <2e-16        
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             0.796   0.4536    0.613    0.615    0.669
## Specificity             0.887   0.9537    0.873    0.935    0.904
## Pos Pred Value          0.737   0.7016    0.506    0.651    0.611
## Neg Pred Value          0.916   0.8792    0.914    0.925    0.924
## Prevalence              0.285   0.1935    0.174    0.164    0.184
## Detection Rate          0.227   0.0878    0.107    0.101    0.123
## Detection Prevalence    0.307   0.1251    0.211    0.155    0.201
## Balanced Accuracy       0.842   0.7037    0.743    0.775    0.787
```


The performance is actually much lower than compared with the random forest model.
We of course also have to look at the performance in the validation data set:


```r
confusionMatrix(pred.caret.mn.val, dat.val$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1435  208  107  119  105
##          B   67  567   72    9  107
##          C  170  226  681  167  108
##          D   88   33  150  635  104
##          E   64  206  107  119  752
## 
## Overall Statistics
##                                         
##                Accuracy : 0.635         
##                  95% CI : (0.623, 0.647)
##     No Information Rate : 0.285         
##     P-Value [Acc > NIR] : <2e-16        
##                                         
##                   Kappa : 0.538         
##  Mcnemar's Test P-Value : <2e-16        
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             0.787   0.4573    0.610   0.6053    0.639
## Specificity             0.882   0.9506    0.873   0.9300    0.905
## Pos Pred Value          0.727   0.6898    0.504   0.6287    0.603
## Neg Pred Value          0.912   0.8795    0.914   0.9233    0.918
## Prevalence              0.285   0.1936    0.174   0.1638    0.184
## Detection Rate          0.224   0.0885    0.106   0.0991    0.117
## Detection Prevalence    0.308   0.1283    0.211   0.1577    0.195
## Balanced Accuracy       0.835   0.7039    0.741   0.7677    0.772
```


In the validation data set, performance is a little worse, but not by a lot.

In terms of variable importance, we see some differences compared to the 
random forest model:


```r
varImp(caret.mn)
```

```
## multinom variable importance
## 
##   only 20 most important variables shown (out of 51)
## 
##                     Overall
## total_accel_belt     100.00
## pitch_belt            58.88
## roll_belt             56.68
## accel_belt_y          36.73
## yaw_belt              36.12
## gyros_arm_x           31.15
## accel_belt_x          21.38
## accel_belt_z          17.90
## gyros_dumbbell_y      16.74
## total_accel_arm       14.57
## gyros_arm_y           14.08
## gyros_forearm_x       13.89
## total_accel_forearm   13.68
## magnet_belt_x         13.03
## magnet_dumbbell_z     12.63
## gyros_forearm_y       12.61
## pitch_forearm         11.23
## magnet_belt_z          8.82
## yaw_dumbbell           8.49
## accel_dumbbell_x       8.27
```

For the multinomial model (which assumes linear relationships and only 
main effects, in this case), the total acceleration as measured by the belt
sensor is the most important variable, by far. It is followed by the pitch
sensor and the roll sensor of the belt. The latter was identified as 
the most important variable by the random forest model.

### K nearest neighbor

As a third (and final) algorithm, we will be training a k-nearest-neighbor
algorithm. Again, we will be using the standard settings from `caret`'s 
`train()` function, which also figures out an acceptable parameter value
for the number of neighbors *k* that will be used for prediction later.


```r
set.seed(55)
caret.knn <- train(classe ~ roll_belt + pitch_belt + yaw_belt + total_accel_belt + 
    gyros_belt_x + gyros_belt_y + gyros_belt_z + accel_belt_x + accel_belt_y + 
    accel_belt_z + magnet_belt_x + magnet_belt_y + magnet_belt_z + roll_arm + 
    pitch_arm + yaw_arm + total_accel_arm + gyros_arm_x + gyros_arm_y + gyros_arm_z + 
    accel_arm_x + accel_arm_y + accel_arm_z + magnet_arm_x + magnet_arm_y + 
    magnet_arm_z + roll_dumbbell + pitch_dumbbell + yaw_dumbbell + gyros_dumbbell_x + 
    gyros_dumbbell_y + gyros_dumbbell_z + accel_dumbbell_x + accel_dumbbell_y + 
    accel_dumbbell_z + magnet_dumbbell_x + magnet_dumbbell_y + magnet_dumbbell_z + 
    roll_forearm + pitch_forearm + yaw_forearm + total_accel_forearm + gyros_forearm_x + 
    gyros_forearm_y + gyros_forearm_z + accel_forearm_x + accel_forearm_y + 
    accel_forearm_z + magnet_forearm_x + magnet_forearm_y + magnet_forearm_z, 
    data = dat.train, method = "knn")
```


The final value for *k* chosen by the algorithm was *k=5*.
Training time was about 2 minutes.

In order to predict the class of the exercise (`classe` variable), we need some
additional code. The knn algorithm returns the probability for each class, so we
need to select the class that has the highest probability. In case two classes
have the same probability, we randomly choose a class (in order not to 
make a systematic error.)


```r
pred.caret.knn.train.raw <- predict(caret.knn$finalModel, newdata = dat.train[-1])
pred.caret.knn.train <- apply(pred.caret.knn.train.raw, 1, function(i) colnames(pred.caret.knn.train.raw)[sample(which(i == 
    max(i)), size = 1)])
pred.caret.knn.val.raw <- predict(caret.knn$finalModel, newdata = dat.val[-1])
pred.caret.knn.val <- apply(pred.caret.knn.val.raw, 1, function(i) colnames(pred.caret.knn.val.raw)[sample(which(i == 
    max(i)), size = 1)])
```


Now we can inspect the performance in the training set:


```r
confusionMatrix(pred.caret.knn.train, dat.train$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1762  614  371  270  258
##          B   29  578  386  249  248
##          C   16   28  339  274  226
##          D   11   12   16  252  212
##          E    5    7    5    4  232
## 
## Overall Statistics
##                                         
##                Accuracy : 0.494         
##                  95% CI : (0.482, 0.506)
##     No Information Rate : 0.285         
##     P-Value [Acc > NIR] : <2e-16        
##                                         
##                   Kappa : 0.339         
##  Mcnemar's Test P-Value : <2e-16        
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             0.967   0.4665   0.3035   0.2402   0.1973
## Specificity             0.670   0.8234   0.8971   0.9531   0.9960
## Pos Pred Value          0.538   0.3879   0.3839   0.5010   0.9170
## Neg Pred Value          0.981   0.8655   0.8591   0.8649   0.8465
## Prevalence              0.285   0.1935   0.1744   0.1638   0.1836
## Detection Rate          0.275   0.0903   0.0529   0.0394   0.0362
## Detection Prevalence    0.511   0.2327   0.1379   0.0785   0.0395
## Balanced Accuracy       0.818   0.6450   0.6003   0.5967   0.5966
```


Accuracy is even lower than for the multinomial model above already in the
training set, and things don't look any better in the validation set:


```r
confusionMatrix(pred.caret.knn.val, dat.val$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1713  596  428  297  270
##          B   45  524  323  253  280
##          C   36   59  331  241  218
##          D   22   37   28  250  195
##          E    8   24    7    8  213
## 
## Overall Statistics
##                                         
##                Accuracy : 0.473         
##                  95% CI : (0.461, 0.485)
##     No Information Rate : 0.285         
##     P-Value [Acc > NIR] : <2e-16        
##                                         
##                   Kappa : 0.311         
##  Mcnemar's Test P-Value : <2e-16        
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             0.939   0.4226   0.2963    0.238   0.1811
## Specificity             0.653   0.8256   0.8953    0.947   0.9910
## Pos Pred Value          0.518   0.3677   0.3740    0.470   0.8192
## Neg Pred Value          0.964   0.8563   0.8576    0.864   0.8433
## Prevalence              0.285   0.1936   0.1744    0.164   0.1836
## Detection Rate          0.267   0.0818   0.0517    0.039   0.0333
## Detection Prevalence    0.516   0.2224   0.1382    0.083   0.0406
## Balanced Accuracy       0.796   0.6241   0.5958    0.593   0.5861
```


As a next step, we will do the stacking of the models by training the
meta-learner.

Training the meta-learner (stacking the models)
------------------------------------------------------------

Although the performance of the initial random forest model above
was quite impressive, and the performance of the other models was 
significantly worse, we still use all the models for building an 
ensemble model by stacking the predictions together 
(basically, just for the sake of doing it). 
For that purpose, we will create a new data frame containing the 
predictions of the individual models *on the training set*, as well
as the correct classes of the training set. Note that this is not the way it was
shown in the lectures, but on other sources I consulted. It allows to train
the stacked model on the training data alone, and evaluate the
performance of the stacked model on the validation data set.


```r
dat.stack.train <- data.frame(classe = dat.train$classe, pred.caret.rf = pred.caret.rf.train, 
    pred.caret.mn = pred.caret.mn.train, pred.caret.knn = pred.caret.knn.train)  #,'pred.fit.svm'=pred.fit.svm.train)
```


Additionally, for getting a (kind of) out-of-sample error rate,
we build a data frame of the predictions of the indivdual models
on the validation set, together with the true classes of the 
validation data set.


```r
dat.stack.val <- data.frame(classe = dat.val$classe, pred.caret.rf = pred.caret.rf.val, 
    pred.caret.mn = pred.caret.mn.val, pred.caret.knn = pred.caret.knn.val)  #,'pred.fit.svm'=pred.fit.svm.val1)
```


In order to use this for predicting the classes with the meta leaner,
we actually need to generate dummy variables for all of the predictors,
which is achieved by the `model.matrix()` function:


```r
## convert to model matrix (for prediction of new data):
dat.stack.val.mm <- model.matrix(classe ~ ., data = dat.stack.val)
head(dat.stack.val.mm)
```

```
##    (Intercept) pred.caret.rfB pred.caret.rfC pred.caret.rfD pred.caret.rfE
## 6            1              0              0              0              0
## 9            1              0              0              0              0
## 11           1              0              0              0              0
## 15           1              0              0              0              0
## 26           1              0              0              0              0
## 33           1              0              0              0              0
##    pred.caret.mnB pred.caret.mnC pred.caret.mnD pred.caret.mnE
## 6               0              0              0              0
## 9               0              0              0              0
## 11              0              0              0              0
## 15              0              0              0              0
## 26              0              0              0              0
## 33              0              0              0              0
##    pred.caret.knnB pred.caret.knnC pred.caret.knnD pred.caret.knnE
## 6                0               0               0               0
## 9                0               0               0               0
## 11               0               0               0               0
## 15               0               0               0               0
## 26               0               0               0               0
## 33               0               0               0               0
```


So now we have the data to train (and also test) the meta learner,
so we can start with training it on the data frame that contains 
only the training data. We will use a random forest model as the meta
learner in this case.


```r
set.seed(78)
fit.stack.rf <- train(classe ~ ., data = dat.stack.train, method = "rf")
```


Now we create a vector of predictions of the ensemble model, both for the
training data (which was used to train the meta learner), as well as
for the validation data:


```r
pred.fit.stack.rf.train <- predict(fit.stack.rf$finalModel)
pred.fit.stack.rf.val <- predict(fit.stack.rf$finalModel, newdata = dat.stack.val.mm)
```


Now let's look at the in-sample performance of the meta learner in the
training data:


```r
confusionMatrix(pred.fit.stack.rf.train, dat.stack.train$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1816   24    0    0    0
##          B    4 1201   21    1    4
##          C    2   12 1090   21    3
##          D    0    0    6 1026    6
##          E    1    2    0    1 1163
## 
## Overall Statistics
##                                        
##                Accuracy : 0.983        
##                  95% CI : (0.98, 0.986)
##     No Information Rate : 0.285        
##     P-Value [Acc > NIR] : <2e-16       
##                                        
##                   Kappa : 0.979        
##  Mcnemar's Test P-Value : NA           
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             0.996    0.969    0.976    0.978    0.989
## Specificity             0.995    0.994    0.993    0.998    0.999
## Pos Pred Value          0.987    0.976    0.966    0.988    0.997
## Neg Pred Value          0.998    0.993    0.995    0.996    0.998
## Prevalence              0.285    0.193    0.174    0.164    0.184
## Detection Rate          0.284    0.188    0.170    0.160    0.182
## Detection Prevalence    0.287    0.192    0.176    0.162    0.182
## Balanced Accuracy       0.995    0.982    0.984    0.988    0.994
```


The performance is quite impressive, and it is higher than each of the 
individual models (even the random forest model). But in order to really
judge the performance, we need to look at the predictions of the meta
learner for the validation data:


```r
confusionMatrix(pred.fit.stack.rf.val, dat.stack.val$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1809   16    0    0    0
##          B    3 1206   16    1    2
##          C   10   18 1100   23    5
##          D    1    0    1 1024    5
##          E    1    0    0    1 1164
## 
## Overall Statistics
##                                         
##                Accuracy : 0.984         
##                  95% CI : (0.981, 0.987)
##     No Information Rate : 0.285         
##     P-Value [Acc > NIR] : < 2e-16       
##                                         
##                   Kappa : 0.98          
##  Mcnemar's Test P-Value : 1.22e-07      
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             0.992    0.973    0.985    0.976    0.990
## Specificity             0.997    0.996    0.989    0.999    1.000
## Pos Pred Value          0.991    0.982    0.952    0.993    0.998
## Neg Pred Value          0.997    0.993    0.997    0.995    0.998
## Prevalence              0.285    0.194    0.174    0.164    0.184
## Detection Rate          0.282    0.188    0.172    0.160    0.182
## Detection Prevalence    0.285    0.192    0.180    0.161    0.182
## Balanced Accuracy       0.994    0.984    0.987    0.987    0.995
```


The performance measures show a really good performance of the stacked
model even in the validation data. Accuracy is as high as .984, which 
again is higher as for each of the individual models.

Testing the meta learner: Out-of-sample error
-------------------------------------

As a final estimate for the out-of-sample error, we now will look at the
performance of the meta learner in the testing data, which we have not 
touched yet.

First, we will create vectors of predictions for each of the individual
models, just like above:


```r
pred.caret.rf.test <- predict(caret.rf$finalModel, newdata = dat.test)

pred.caret.mn.test <- predict(caret.mn$finalModel, newdata = dat.test)

pred.caret.knn.test.raw <- predict(caret.knn$finalModel, newdata = dat.test[-1])
pred.caret.knn.test <- apply(pred.caret.knn.test.raw, 1, function(i) colnames(pred.caret.knn.test.raw)[sample(which(i == 
    max(i)), size = 1)])
```


Then we will just build a data frame containing these predictions of the 
individual model in the testing data, together with the `classe` variable
of the testing data. We again need to build dummy variables for the meta
learner to be able to use the data.


```r
dat.stack.test <- data.frame(classe = dat.test$classe, pred.caret.rf = pred.caret.rf.test, 
    pred.caret.mn = pred.caret.mn.test, pred.caret.knn = pred.caret.knn.test)

## convert to model matrix (for prediction of new data):
dat.stack.test.mm <- model.matrix(classe ~ ., data = dat.stack.test)
```


Now we can make the predictions for the testing data:


```r
pred.fit.stack.rf.test <- predict(fit.stack.rf$finalModel, newdata = dat.stack.test.mm)
```


To finally get an estimate of the out-of-sample error, with data that we
didn't touch yet, we compute the performance measures on the testing data:


```r
confusionMatrix(pred.fit.stack.rf.test, dat.stack.test$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1813   21    0    0    0
##          B    2 1209    9    1    1
##          C    8    9 1109   20    1
##          D    0    0    0 1027    6
##          E    1    0    0    1 1168
## 
## Overall Statistics
##                                        
##                Accuracy : 0.988        
##                  95% CI : (0.984, 0.99)
##     No Information Rate : 0.285        
##     P-Value [Acc > NIR] : <2e-16       
##                                        
##                   Kappa : 0.984        
##  Mcnemar's Test P-Value : NA           
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             0.994    0.976    0.992    0.979    0.993
## Specificity             0.995    0.997    0.993    0.999    1.000
## Pos Pred Value          0.989    0.989    0.967    0.994    0.998
## Neg Pred Value          0.998    0.994    0.998    0.996    0.998
## Prevalence              0.285    0.193    0.175    0.164    0.184
## Detection Rate          0.283    0.189    0.173    0.160    0.182
## Detection Prevalence    0.286    0.191    0.179    0.161    0.183
## Balanced Accuracy       0.995    0.987    0.992    0.989    0.996
```


Even in the testing data, the stacked model achieves an accuracy of .988,
which is even higher than the accuracy both in the training and 
in the validation data set (almost certainly a random fluctuation).


