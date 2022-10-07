UniversalBank-MLProject
================
Yeshil Bangera
06/10/2022

## Loading all required Packages

``` r
library(caret)
```

    ## Warning: package 'caret' was built under R version 4.1.3

    ## Loading required package: ggplot2

    ## Warning: package 'ggplot2' was built under R version 4.1.3

    ## Loading required package: lattice

``` r
library(lattice)
library(ggplot2)
library(FNN)
```

    ## Warning: package 'FNN' was built under R version 4.1.3

``` r
loan_df <- read.csv("UniversalBank.csv")
loan_df <- subset(loan_df, select = -5)
loan_df <- subset(loan_df, select = -1)
#loan.df$Education <-as.factor(loan.df$Education)
#loan.df$Securities.Account<-as.factor(loan.df$Securities.Account)
#loan.df$CD.Account <-as.factor(loan.df$CD.Account)
#loan.df$Online <-as.factor(loan.df$Online)
#loan.df$CreditCard <-as.factor(loan.df$CreditCard)
#loan.df$Personal.Loan<-as.factor(loan.df$Personal.Loan)
head(loan_df)
```

    ##   Age Experience Income Family CCAvg Education Mortgage Personal.Loan
    ## 1  25          1     49      4   1.6         1        0             0
    ## 2  45         19     34      3   1.5         1        0             0
    ## 3  39         15     11      1   1.0         1        0             0
    ## 4  35          9    100      1   2.7         2        0             0
    ## 5  35          8     45      4   1.0         2        0             0
    ## 6  37         13     29      4   0.4         2      155             0
    ##   Securities.Account CD.Account Online CreditCard
    ## 1                  1          0      0          0
    ## 2                  1          0      0          0
    ## 3                  0          0      0          0
    ## 4                  0          0      0          0
    ## 5                  0          0      0          1
    ## 6                  0          0      1          0

``` r
View(loan_df)
```

``` r
set.seed(111)
train.index <- sample(row.names(loan_df), 0.75*dim(loan_df)[1])
valid.index <- setdiff(row.names(loan_df), train.index)
train.df <- loan_df[train.index, ]
valid.df <- loan_df[valid.index, ]
## new customer
#Age = 40, Experience = 10, Income = 84, Family = 2, CCAvg = 2, Education_1 = 0, Education_2 = 1,
#Education_3 = 0, Mortgage = 0, Securities Account = 0, CD Account = 0, Online = 1, and Credit Card = 1.
new.df <- data.frame(Age = 40, Experience = 10, Income = 84, Family = 2, CCAvg = 2, Education=2, Mortgage = 0, Securities.Account = 0, CD.Account = 0, Online = 1, CreditCard = 1)
new.df
```

    ##   Age Experience Income Family CCAvg Education Mortgage Securities.Account
    ## 1  40         10     84      2     2         2        0                  0
    ##   CD.Account Online CreditCard
    ## 1          0      1          1

``` r
# Initialize normalization of training , validation data
train.norm.df <- train.df
valid.norm.df <- valid.df
loan.norm.df <- loan_df
new.norm.df <- new.df
head(loan.norm.df)
```

    ##   Age Experience Income Family CCAvg Education Mortgage Personal.Loan
    ## 1  25          1     49      4   1.6         1        0             0
    ## 2  45         19     34      3   1.5         1        0             0
    ## 3  39         15     11      1   1.0         1        0             0
    ## 4  35          9    100      1   2.7         2        0             0
    ## 5  35          8     45      4   1.0         2        0             0
    ## 6  37         13     29      4   0.4         2      155             0
    ##   Securities.Account CD.Account Online CreditCard
    ## 1                  1          0      0          0
    ## 2                  1          0      0          0
    ## 3                  0          0      0          0
    ## 4                  0          0      0          0
    ## 5                  0          0      0          1
    ## 6                  0          0      1          0

``` r
str(loan.norm.df)
```

    ## 'data.frame':    5000 obs. of  12 variables:
    ##  $ Age               : int  25 45 39 35 35 37 53 50 35 34 ...
    ##  $ Experience        : int  1 19 15 9 8 13 27 24 10 9 ...
    ##  $ Income            : int  49 34 11 100 45 29 72 22 81 180 ...
    ##  $ Family            : int  4 3 1 1 4 4 2 1 3 1 ...
    ##  $ CCAvg             : num  1.6 1.5 1 2.7 1 0.4 1.5 0.3 0.6 8.9 ...
    ##  $ Education         : int  1 1 1 2 2 2 2 3 2 3 ...
    ##  $ Mortgage          : int  0 0 0 0 0 155 0 0 104 0 ...
    ##  $ Personal.Loan     : int  0 0 0 0 0 0 0 0 0 1 ...
    ##  $ Securities.Account: int  1 1 0 0 0 0 0 0 0 0 ...
    ##  $ CD.Account        : int  0 0 0 0 0 0 0 0 0 0 ...
    ##  $ Online            : int  0 0 0 0 0 1 1 0 1 0 ...
    ##  $ CreditCard        : int  0 0 0 0 1 0 0 1 0 0 ...

``` r
#using preProcess from the caret library to normalize the dataset
norm.values <- preProcess(train.df[,c(1,2,3,4,5,7)], method = c("center", "scale"))
train.norm.df[, c(1,2,3,4,5,7)] <- predict(norm.values, train.df[,c(1,2,3,4,5,7)])
valid.norm.df[, c(1,2,3,4,5,7)] <- predict(norm.values, valid.df[, c(1,2,3,4,5,7)]) 
loan.norm.df[, c(1,2,3,4,5,7)] <- predict(norm.values, loan_df[,c(1,2,3,4,5,7)]) 
new.norm.df[,c(1,2,3,4,5,7)] <- predict(norm.values, new.df[,c(1,2,3,4,5,7)])
```

``` r
#install.packages('e1071', dependencies=TRUE)
# initialize a data frame with two columns: k, and accuracy.
accuracy.df <- data.frame(k = seq(1,20,1), accuracy = rep(0,20))
# compute knn for different k on validation.
for(i in 1:20) 
  { knn.pred <- knn(train.norm.df[, -8], valid.norm.df[, -8], 
                    cl = train.norm.df[,8], k = i)
  
accuracy.df[i,2] <- confusionMatrix(as.factor(knn.pred), as.factor(valid.norm.df[,8]))$overall[1]
}
print(accuracy.df)
```

    ##     k accuracy
    ## 1   1   0.9616
    ## 2   2   0.9536
    ## 3   3   0.9656
    ## 4   4   0.9584
    ## 5   5   0.9616
    ## 6   6   0.9584
    ## 7   7   0.9640
    ## 8   8   0.9568
    ## 9   9   0.9592
    ## 10 10   0.9528
    ## 11 11   0.9552
    ## 12 12   0.9528
    ## 13 13   0.9544
    ## 14 14   0.9496
    ## 15 15   0.9528
    ## 16 16   0.9480
    ## 17 17   0.9528
    ## 18 18   0.9488
    ## 19 19   0.9520
    ## 20 20   0.9472

``` r
#Find the best k
best.k <- accuracy.df$k[which.max(accuracy.df$accuracy)] 
best.k
```

    ## [1] 3

``` r
# use knn() to compute knn.
#Knn documentation:https://www.rdocumentation.org/packages/class/versions/7.3-17/topics/knn
# knn() is available in library FNN (provides a list of the nearest neighbors)
# and library class (allows a numerical output variable).
library(FNN)
#unlist(train.norm.df)
#as.numeric(new.norm.df)
#nn2 <- knn(train = train.norm.df[, -8], test = new.norm.df,
# cl = train.norm.df[, 8], k = 3)
nn2 <- knn(train = train.norm.df[, -8], test = new.norm.df, 
           cl = train.norm.df[,8], k = best.k) 
nn2
```

    ## [1] 0
    ## attr(,"nn.index")
    ##      [,1] [,2] [,3]
    ## [1,] 2432 2219 1344
    ## attr(,"nn.dist")
    ##           [,1]      [,2]     [,3]
    ## [1,] 0.4802746 0.4986191 0.638729
    ## Levels: 0

``` r
#Classification of loans in validation data
knn.pred.new <- knn(train.norm.df[, -8], test = valid.norm.df[,-8],
                    cl = train.norm.df[,8], k = best.k)
#Confustion matrix of validation
cf.val <- confusionMatrix(as.factor(knn.pred.new),as.factor(valid.norm.df[,8]))
cf.val
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    0    1
    ##          0 1125   40
    ##          1    3   82
    ##                                          
    ##                Accuracy : 0.9656         
    ##                  95% CI : (0.9539, 0.975)
    ##     No Information Rate : 0.9024         
    ##     P-Value [Acc > NIR] : < 2.2e-16      
    ##                                          
    ##                   Kappa : 0.7742         
    ##                                          
    ##  Mcnemar's Test P-Value : 4.021e-08      
    ##                                          
    ##             Sensitivity : 0.9973         
    ##             Specificity : 0.6721         
    ##          Pos Pred Value : 0.9657         
    ##          Neg Pred Value : 0.9647         
    ##              Prevalence : 0.9024         
    ##          Detection Rate : 0.9000         
    ##    Detection Prevalence : 0.9320         
    ##       Balanced Accuracy : 0.8347         
    ##                                          
    ##        'Positive' Class : 0              
    ## 

``` r
# ## Explanation:
# The data is been trained using 75% of the data set and the rest 25% is used as validation. K-NN algorithm is used to predict if the loan is accepted or not (0 OR 1). Later preprocess function is used to normalize the data. The "center" argument in the preprocess function subtracts the mean of the predictor's data from the predictor values while "scale" divides by the standard deviation.A for loop is used for implementing different values of k to identify the k with best accuracy. Then the best k value which is 3 in this case is used with the new data and a confusion matrix is created.
```

\##Conclusion: A) Level: nn2 returned value 0 which means loan not
accepted B) Best value of K is 3 C) Confusion matrix of validation has
an Accuracy : 0.9656 i.e. 96.56%
