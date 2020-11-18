##########################################################
# Create edx set, validation set (final hold-out test set)
##########################################################

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(data.table)

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")

# if using R 3.6 or earlier:
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
                                           title = as.character(title),
                                           genres = as.character(genres))
movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding") # if using R 3.5 or earlier, use `set.seed(1)`
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>%
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)

#------CODE FOR BUILDING THE RECOMMENDATION SYSTEM FROM EDX DATASET BEGINS HERE------
#create training and test datasets from the edx dataset

set.seed(1, sample.kind="Rounding")
test_index <- createDataPartition(y = edx$rating, times = 1, p = 0.1, list = FALSE)
train_data <- edx[-test_index,]
temp <- edx[test_index,]

# Make sure userId and movieId in test set are also in train set
test_data <- temp %>%
  semi_join(train_data, by = "movieId") %>%
  semi_join(train_data, by = "userId")

# Add rows removed from test set back into train set
removed <- anti_join(temp, test_data)
train_data <- rbind(train_data, removed)

rm(test_index, temp, removed)

#measure of dispersion and central tendencies using the training dataset
avg_rating <- mean(train_data$rating)
sd_rating <- sd(train_data$rating)
quartiles <- quantile(train_data$rating)

# Define Mean Absolute Error (MAE)
MAE <- function(true_ratings, predicted_ratings){
  mean(abs(true_ratings - predicted_ratings))
}

# Define Mean Squared Error (MSE)
MSE <- function(true_ratings, predicted_ratings){
  mean((true_ratings - predicted_ratings)^2)
}

# Define Root Mean Squared Error (RMSE)
RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}

#EXPLORATORY DATA ANALYSIS OF THE TRAINING DATASET A SUBSET OF THE EDX DATASET
head(train_data) #excerpt of the data
train_data %>% group_by(rating) %>% summarize(n=n())#shows rating value range
boxplot(train_data$rating) #reveals rating pattern

#User relationsip towards movie using a scatterplot
plot(train_data$userId, train_data$movieId)

#The User distribution
train_data %>% group_by(userId) %>%
  summarise(n=n()) %>%
  ggplot(aes(n)) +
  geom_histogram(color = "red") +
  scale_x_log10() +
  ggtitle("Users Distribution") +
  xlab("Number of Ratings") +
  ylab("Number of Users")

#The Movie Distribition
train_data %>% group_by(movieId) %>%
  summarise(n=n()) %>%
  ggplot(aes(n)) +
  geom_histogram(color = "blue") +
  scale_x_log10() +
  ggtitle("Distribution of Movies") +
  xlab("Number of Ratings") +
  ylab("Number of Movies")

#Heatmap of the User and Movie Relationship
users <- sample(unique(train_data$userId), 100)
train_data %>% filter(userId %in% users) %>%
  select(userId, movieId, rating) %>%
  mutate(rating = 1) %>%
  spread(movieId, rating) %>%
  select(sample(ncol(.), 100)) %>%
  as.matrix() %>% t(.) %>%
  image(1:100, 1:100,. , xlab="Movies", ylab="Users")+
abline(h=0:100+0.5, v=0:100+0.5, col = "grey")+
title("Heatmap of User vs Movie")

#MODELING=====================================

#MODEL_RESULT_1
# naive (rmse, mse, mae) using random prediction
set.seed(4321, sample.kind = "Rounding")

# Create the probability of each rating
p <- function(x, y) mean(y == x)
rating <- seq(0.5,5,0.5)

# Estimate the probability of each rating with Monte Carlo simulation
B <- 10^4
M <- replicate(B, {
  s <- sample(train_data$rating, 100, replace = TRUE)
  sapply(rating, p, y= s)
})
prob <- sapply(1:nrow(M), function(x) mean(M[x,]))

# Predict random ratings
y_hat_random <- sample(rating, size = nrow(test_data),
                       replace = TRUE, prob = prob)

# Create a table with the error results
result <- tibble(Method = "Initial Assumption for Prediction Error", RMSE = 0000, MSE = 0000, MAE = 0000)
result <- bind_rows(result,
                    tibble(Method = "Model_Result_1 (Random Prediction)",
                           RMSE = RMSE(test_data$rating, y_hat_random),
                           MSE  = MSE(test_data$rating, y_hat_random),
                           MAE  = MAE(test_data$rating, y_hat_random)))
result %>% knitr::kable()

#MODEL_RESULT_2
#naive (rmse, mse, mae) using the training dataset
# Update the error table
result <- bind_rows(result,
                    tibble(Method = "Model_Result_2 (Training Data)",
                           RMSE = RMSE(test_data$rating, train_data$rating),
                           MSE  = MSE(test_data$rating, train_data$rating),
                           MAE  = MAE(test_data$rating, train_data$rating)))
result %>% knitr::kable()

#MODEL_RESULT_3
#naive (rmse, mse, mae) using lower limit of spread: avergae minus one sd
lower_lim <- avg_rating - sd_rating

# Update the error estimate table
result <- bind_rows(result,
                    tibble(Method = "Model_result_3 (Lower Limit of Spread)",
                           RMSE = RMSE(test_data$rating, lower_lim),
                           MSE  = MSE(test_data$rating, lower_lim),
                           MAE  = MAE(test_data$rating, lower_lim)))
result %>% knitr::kable()

#MODEL_RESULT_4
#naive (rmse, mse, mae) using upper limit of spread: avergae plus one sd
upper_lim <- avg_rating + sd_rating

# Update the error estimate table
result <- bind_rows(result,
                    tibble(Method = "Model_result_4 (Upper Limit of Spread)",
                           RMSE = RMSE(test_data$rating, upper_lim),
                           MSE  = MSE(test_data$rating, upper_lim),
                           MAE  = MAE(test_data$rating, upper_lim)))
result %>% knitr::kable()

#MODEL_RESULT_5
#naive (rmse, mse, mae) using first quartile
quantile(train_data$rating)

# Update the error estimate table
result <- bind_rows(result,
                    tibble(Method = "Model_result_5 (First Quartile)",
                           RMSE = RMSE(test_data$rating, 3),
                           MSE  = MSE(test_data$rating, 3),
                           MAE  = MAE(test_data$rating, 3)))
result %>% knitr::kable()

#MODEL_RESULT_6
#naive (rmse, mse, mae) using third quartile
quantile(train_data$rating)

# Update the error estimate table
result <- bind_rows(result,
                    tibble(Method = "Model_Result_6 (Third Quartile)",
                           RMSE = RMSE(test_data$rating, 4),
                           MSE  = MSE(test_data$rating, 4),
                           MAE  = MAE(test_data$rating, 4)))
result %>% knitr::kable()

#MODEL_RESULT_7
# naive (rmse, mse, mae) using Mean of observed values
mu <- mean(train_data$rating)

# Update the error estimate table
result <- bind_rows(result,
                    tibble(Method = "Model_Result_7 (Rating Mean)",
                           RMSE = RMSE(test_data$rating, mu),
                           MSE  = MSE(test_data$rating, mu),
                           MAE  = MAE(test_data$rating, mu)))
result %>% knitr::kable()

#MODEL_RESULT_8
# Add bias due to Movie effects (bi)TO MODEL 7
bi <- train_data %>%
  group_by(movieId) %>%
  summarize(b_i = mean(rating - mu))
head(bi)

bi %>% ggplot(aes(x = b_i)) +
  geom_histogram(bins=10, col = I("black")) +
  ggtitle("Movie Effect Distribution") +
  xlab("Movie effect") +
  ylab("Count")

# Predict the rating with mean + bi
y_hat_bi <- mu + test_data %>%
  left_join(bi, by = "movieId") %>%
  .$b_i

# Calculate the RMSE
result <- bind_rows(result,
                    tibble(Method = "Model_Result_8 (Mean + bi)",
                           RMSE = RMSE(test_data$rating, y_hat_bi),
                           MSE  = MSE(test_data$rating, y_hat_bi),
                           MAE  = MAE(test_data$rating, y_hat_bi)))
result %>% knitr::kable()

#MODEL_RESULT_9
# Add bias due to User effect (bu) Model_8
bu <- train_data %>%
  left_join(bi, by = 'movieId') %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu - b_i))

head(bu)

bu %>% ggplot(aes(x = b_u)) +
  geom_histogram(bins=10, col = I("black")) +
  ggtitle("User Effect Distribution") +
  xlab("User effect") +
  ylab("Count")

# Prediction
y_hat_bi_bu <- test_data %>%
  left_join(bi, by='movieId') %>%
  left_join(bu, by='userId') %>%
  mutate(pred = mu + b_i + b_u) %>%
  .$pred

# Update the results table
result <- bind_rows(result,
                    tibble(Method = "Model_Result_9 (Mean + bi + bu)",
                           RMSE = RMSE(test_data$rating, y_hat_bi_bu),
                           MSE  = MSE(test_data$rating, y_hat_bi_bu),
                           MAE  = MAE(test_data$rating, y_hat_bi_bu)))
result %>% knitr::kable()

#REGULARIZATION TO CONTROL USER AND MOVIE BIAS
lambdas <- seq(0, 10, 0.25)
rmses <- sapply(lambdas, function(l){
  mu <- mean(train_data$rating)

# Movie effect (bi)
  b_i <- train_data %>%
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+l))

# User effect (bu)
    b_u <- train_data %>%
    left_join(b_i, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n()+l))

#prediction: mu + bi + bu
  predicted_ratings <- test_data %>%
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    mutate(pred = mu + b_i + b_u) %>%
    .$pred
  return(RMSE(predicted_ratings, test_data$rating))
})

# select the lambda that returns the best RMSE.
lambda <- lambdas[which.min(rmses)]

# Plot to determine lambda with best RMSE
tibble(Lambda = lambdas, RMSE = rmses) %>%
  ggplot(aes(x = Lambda, y = RMSE)) +
  geom_point() +
  ggtitle("Regularization",
          subtitle = "penalizations showing best regularized RMSE.")

# PREDICTION MODELING USING BEST RESULT FROM REGULARIZATION (lambda).
mu <- mean(train_data$rating)

# Movie effect (bi)
b_i <- train_data %>%
  group_by(movieId) %>%
  summarize(b_i = sum(rating - mu)/(n()+lambda))

# User effect (bu)
b_u <- train_data %>%
  left_join(b_i, by="movieId") %>%
  group_by(userId) %>%
  summarize(b_u = sum(rating - b_i - mu)/(n()+lambda))

# Prediction
y_hat_reg <- test_data %>%
  left_join(b_i, by = "movieId") %>%
  left_join(b_u, by = "userId") %>%
  mutate(pred = mu + b_i + b_u) %>%
  pull(pred)

# Update the result table
result <- bind_rows(result,
                    tibble(Method = "Regularized model (bi and bu)",
                           RMSE = RMSE(test_data$rating, y_hat_reg),
                           MSE  = MSE(test_data$rating, y_hat_reg),
                           MAE  = MAE(test_data$rating, y_hat_reg)))

result %>% knitr::kable()


#FINAL TESTING USING FINAL HOLD OUT SET (vALIDATION)

mu_edx <- mean(edx$rating)

# Movie effect (bi)
b_i_edx <- edx %>%
  group_by(movieId) %>%
  summarize(b_i = sum(rating - mu_edx)/(n()+lambda))

# User effect (bu)
b_u_edx <- edx %>%
  left_join(b_i_edx, by="movieId") %>%
  group_by(userId) %>%
  summarize(b_u = sum(rating - b_i - mu_edx)/(n()+lambda))

# Prediction
y_hat_edx <- validation %>%
  left_join(b_i_edx, by = "movieId") %>%
  left_join(b_u_edx, by = "userId") %>%
  mutate(pred = mu_edx + b_i + b_u) %>%
  pull(pred)

# Update the results table
result <- bind_rows(result,
                    tibble(Method = "Final Model Testing (edx vs validation)",
                           RMSE = RMSE(validation$rating, y_hat_edx),
                           MSE  = MSE(validation$rating, y_hat_edx),
                           MAE  = MAE(validation$rating, y_hat_edx)))

result %>% knitr::kable()

#EVALUATION OF RESULT USING VALIDATION SET

#The best ten(10) rated movies
validation %>%
  left_join(b_i_edx, by = "movieId") %>%
  left_join(b_u_edx, by = "userId") %>%
  mutate(pred = mu_edx + b_i + b_u) %>%
  arrange(-pred) %>%
  group_by(title) %>%
  select(title) %>%
  head(10)

#The worst ten(10) rated movies
validation %>%
  left_join(b_i_edx, by = "movieId") %>%
  left_join(b_u_edx, by = "userId") %>%
  mutate(pred = mu_edx + b_i + b_u) %>%
  arrange(pred) %>%
  group_by(title) %>%
  select(title) %>%
  head(10)


#EVALUATION OF RESULT USING TEST SET, A SUBSET OF EDX DATASET

#The best ten(10) rated movies
test_data %>%
  left_join(b_i, by = "movieId") %>%
  left_join(b_u, by = "userId") %>%
  mutate(pred = mu + b_i + b_u) %>%
  arrange(-pred) %>%
  group_by(title) %>%
  select(title) %>%
  head(10)

#The worst ten(10) rated movies
test_data %>%
  left_join(b_i, by = "movieId") %>%
  left_join(b_u, by = "userId") %>%
  mutate(pred = mu + b_i + b_u) %>%
  arrange(pred) %>%
  group_by(title) %>%
  select(title) %>%
  head(10)

#END OF CODE===================================

