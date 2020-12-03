
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(ggplot2)) install.packages("ggplot2", repos = "http://cran.us.r-project.org")
if(!require(dplyr)) install.packages("dplyr", repos = "http://cran.us.r-project.org")
if(!require(gbm)) install.packages("gbm", repos = "http://cran.us.r-project.org")
if(!require(e1071)) install.packages('e1071', dependencies=TRUE, repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(data.table)
library(ggplot2)
library(dplyr)

#===================DATA SOURCING AND CLEANING====================
#import data set

db <- tempfile() 


  download.file ("http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data", db)

   colNames = c ("Age", "Workclass", "Final_Weight", "Education", 
                "Education_num", "Marital_Status", "Occupation",
                "Relationship", "Race", "Sex", "Capital_Gain",
                "Capital_Loss", "Hours_Per_week", "Native_Country",
                "Income_Level")
  
  raw_data = read.table (db, header = FALSE, sep = ",",
                      strip.white = TRUE, col.names = colNames,
                      na.strings = "?", stringsAsFactors = TRUE)
  
#Pre-precessing or Data cleaning
  summary (raw_data)
  str (raw_data)   #expores the data type and data structure
  
  # Summary of all data sets without missing data
  table (complete.cases (raw_data))
  summary(raw_data [complete.cases (raw_data),])

  # Summary of all data sets with NAs or missing data
  table(!complete.cases(raw_data))
  summary  (raw_data [!complete.cases(raw_data),])  
  
  new_data = raw_data [!is.na (raw_data$Workclass) & !is.na (raw_data$Occupation) & 
                             !is.na (raw_data$Native_Country), ]
  
  # income_level variable proportion in new data
  table(new_data$Income_Level)
  
#create data partition 
  set.seed(2, sample.kind="Rounding")
  test_index <- createDataPartition(y = new_data$Income_Level, times = 1, p = 0.5, list = FALSE)
  train_set <- new_data[-test_index,]
  test_set <- new_data[test_index,]

#===================EXPLORATORY DATA ANALYSIS====================== 
  head(train_set)
  
#Gender analysis  
  male_gender <- train_set %>% filter (Sex == "Male")
  female_gender <- train_set %>% filter (Sex == "Female")
  table (train_set[,c("Sex", "Income_Level")])
  table (train_set[,c("Sex", "Education")]) 

  #gender proportion: gprop
  gender_prop <-  (sum(train_set$Sex == "Male")/sum(train_set$Sex == "Female"))
  gender_prop
  
  #male gender distribution using age in prediction outcomes
  boxplot (Age ~ Income_Level, data = male_gender, 
           main = "Age distribution for Male Gender income levels",
           xlab = "Income Levels", ylab = "Age", col = "Blue") 
  
  #female gender distribution using age in prediction outcomes
  boxplot (Age ~ Income_Level, data = female_gender, 
           main = "Age distribution for Female Gender income levels",
           xlab = "Income Levels", ylab = "Age", col = "Pink") 
  
  # Age distribution along income levels
    boxplot (Age ~ Income_Level, data = train_set, 
           main = "Age distribution for different income levels",
           xlab = "Income Levels", ylab = "Age", col = "Green")  
  
  #Age distribution for different levels of income 
  incomeBelow50K <- (train_set$Income_Level == "<=50K")
  xlimit <- c (min (train_set$Age), max (train_set$Age))
  ylimit <- c (0, 1000)
  
  hist1 <- qplot (Age, data = train_set[incomeBelow50K,], margins = TRUE, 
                 binwidth = 2, xlim = xlimit, ylim = ylimit, colour = Income_Level,
                 main = "Age distribution for income level <=50K")
  
  hist2 <- qplot (Age, data = train_set[!incomeBelow50K,], margins = TRUE, 
                 binwidth = 2, xlim = xlimit, ylim = ylimit, colour = Income_Level,
                 main = "Age distribution for income level >50K",)
  
  #Income distribution by Education level
  boxplot (Education_num ~ Income_Level, data = train_set, 
           main = "Income Distribution by Education Level",
           xlab = "Income Levels", ylab = "Education level", col = "white")
  
  #Job distribution by Education level
  boxplot (Education_num ~ Workclass, data = train_set,
           main = "Job Distribution by Education Levels",
           xlab = " ", ylab = "Education Number", col = "Dark Gray",
           las = 2)
  
  #income level distribution by work duration
  boxplot (Hours_Per_week ~ Income_Level, data = train_set,
           main = "Income Level Distribution by Hours worked per week",
           xlab = "Income Level ", ylab = "Hours Per Week", col = "violet",
           las = 2 )
  
  qplot (Income_Level, data = train_set, fill = Workclass) + 
    theme(axis.text.x = element_text(angle=90)) +
    facet_grid (. ~ Workclass)
  
  qplot (Income_Level, data = train_set, fill = Occupation) + 
    theme(axis.text.x = element_text(angle=90)) +
    facet_grid (. ~ Occupation)
  
  #=========================MODELLING ============================== 
  #Income level predicting by gender 
  y <- factor(train_set$Sex, c("Female", "Male"))
  x <- train_set$Income_Level
  
  y_hat <- ifelse(x == ">50K", "Male", "Female") %>% 
    factor(levels = levels(y))
  
  Acc <- mean(y_hat==y) 
  Sen <- sensitivity(y_hat, y)
  spe <- specificity(y_hat, y)  
  prev <- mean(y == "Male")
  tibble(Acc,Sen,spe, prev)
  
  #plot showing that income level is not gender based
  qplot (Sex, data = train_set, fill = Income_Level) + 
    theme(axis.text.x = element_text(angle=90)) +
    facet_grid (. ~ Income_Level)

  #Income Level prediction with Education level  
  a <- factor(train_set$Income_Level, c("<=50K", ">50K"))
  b <- train_set$Education_num
  a_hat <- ifelse(train_set$Education_num > (mean(train_set$Education_num)), "<=50K", ">50K") %>% 
  factor(levels = levels(train_set$Income_Level))
  acc_Ed <- mean(a == a_hat)
  sen_Ed <- sensitivity(a_hat, a)
  spe_Ed <- specificity(a_hat, a) 
  prev_Ed <- mean(a == "<=50K")
  tibble(acc_Ed, sen_Ed, spe_Ed, prev_Ed)
   
  cutoff <- seq(10, 16)
  accuracy <- map_dbl(cutoff, function(b){
    a_hat_hat <- ifelse(train_set$Education_num > b, "<=50K", ">50K") %>% 
      factor(levels = levels(test_set$Income_Level))
    mean(a_hat_hat == train_set$Income_Level)
  })
  
  best_cutoff <- cutoff[which.max(accuracy)]
  best_cutoff
  
  plot(cutoff, accuracy)
  
  #accuracy of the best cutoff for predicting income level with education level
  c_hat <- ifelse(test_set$Education_num > best_cutoff, "<=50K", ">50K") %>% 
    factor(levels = levels(test_set$Income_Level))
  c_hat <- factor(c_hat)
  mean(c_hat == test_set$Income_Level)
  
  #plot showing that higher education attract higher income  
  qplot (Income_Level, data = train_set, fill = Education) + 
    theme(axis.text.x = element_text(angle=90)) +
    facet_grid (. ~ Education)
  
  
  #Demographic Model(Dem_model) using all categories of data
   set.seed(2, sample.kind="Rounding")
  trCtrl <- trainControl(method = "cv", number = 10)
  
   boostFit <- train(Income_Level ~ Age + Workclass + Education + Education_num +
                      Marital_Status + Occupation + Relationship + Sex +
                      Race + Capital_Gain + Capital_Loss + Hours_Per_week +
                      Native_Country, trControl = trCtrl, 
                    method = "gbm", data = train_set, verbose = FALSE)
  
  confusionMatrix (train_set$Income_Level, predict (boostFit, train_set))

  #Demographic Model (Dem_model) Testing  
  test_set$predicted <- predict (boostFit, test_set)
  confusionMatrix (test_set$Income_Level, test_set$predicted)
  
  #Human Capital Model (HC_model) using significant data categories
  
  set.seed(3, sample.kind="Rounding")
  trCtrl_2 <- trainControl(method = "cv", number = 10)
  
  boostFit_2 <- train(Income_Level ~ Age + Education + Education_num + Sex +
                    Workclass + Occupation + Hours_Per_week, trControl = trCtrl_2, 
                    method = "gbm", data = train_set, verbose = FALSE)
  
  confusionMatrix (train_set$Income_Level, predict (boostFit_2, train_set))
  
  #HC_Model Testing  
  test_set$predicted_2 <- predict (boostFit_2, test_set)
  confusionMatrix (test_set$Income_Level, test_set$predicted_2)
  
  #Table of correlation between continuous data
  corTab <- cor (train_set[, c("Age", "Education_num", "Hours_Per_week")])
  corTab %>% knitr::kable() 
  
  #=====================END OF PROGRAM=========================