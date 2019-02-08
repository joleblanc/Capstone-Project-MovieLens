#############################################################
# Created by Johannes Le Blanc
# Februaray 8, 2019
#############################################################

library(dslabs)
library(tidyverse)
library(tibble)
library(dplyr)
library(tidyr)
library(devtools)
library(ggplot2)
library(statsr)
library(ggrepel)
library(SLICER)
library(stringr)
library(htmlwidgets)
library(purrr)
library(lubridate)
library(ggthemes)
library(bindrcpp)
library(gridExtra)
library(MASS)
library(caret)
library(purrr)
library(randomForest)
library(e1071)
library(rpart)

## Analysis

#############################################################
# Create edx set, validation set, and submission file
#############################################################

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- read.table(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                      col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
                                           title = as.character(title),
                                           genres = as.character(genres))

movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data

set.seed(1)
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

######################################################
## Characteristics of the data

# summarize the data
summary(edx)
summary(validation)

# number of unique users and unique movies
edx %>% 
  summarize(n_users = n_distinct(userId),
            n_movies = n_distinct(movieId))

# number of movie ratings
edx %>% 
  count(movieId) %>% 
  ggplot(aes(n)) + 
  geom_histogram(bins = 30, color = "black") + 
  scale_x_log10() + 
  ggtitle("Movies")

# activity of users
edx %>% 
  count(userId) %>% 
  ggplot(aes(n)) + 
  geom_histogram(bins = 30, color = "black") + 
  scale_x_log10() + 
  ggtitle("Users")

# check for NAs in train-set
sum(is.na(edx$title))

#check for NAs in test-set
sum(is.na(validation$title))

######################################################
### The first naive model

mu_hat <- mean(edx$rating)
mu_hat

# naive RMSE
naive_rmse <- RMSE(validation$rating, mu_hat)
naive_rmse

# results table for the naive approach
rmse_results <- data_frame(method = "Just the average", RMSE = naive_rmse)

######################################################
### The movie effect

mu <- mean(edx$rating) 
movie_avgs <- edx %>% 
  group_by(movieId) %>% 
  summarize(b_i = mean(rating - mu))

# plot of estimates
movie_avgs %>% qplot(b_i, geom ="histogram", bins = 30, data = ., color = I("black"))

# new prediction to show the difference to the naive approach
predicted_ratings <- mu + validation %>% 
  left_join(movie_avgs, by='movieId') %>%
  .$b_i

model_1_rmse <- RMSE(predicted_ratings, validation$rating)
rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Movie Effect Model",  
                                     RMSE = model_1_rmse ))
rmse_results %>% knitr::kable()

######################################################
### The user effect 

# plot the user ratings
 edx %>% 
  group_by(userId) %>% 
  summarize(b_u = mean(rating)) %>% 
  filter(n()>=100) %>%
  ggplot(aes(b_u)) + 
  geom_histogram(bins = 30, color = "black")

# compute approximation of user effects
user_avgs <- validation %>% 
  left_join(movie_avgs, by='movieId') %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu - b_i))

# create predictors to show improvement of RMSE
predicted_ratings <- validation %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  mutate(pred = mu + b_i + b_u) %>%
  .$pred


model_2_rmse <- RMSE(predicted_ratings, validation$rating)
rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Movie + User Effects Model",  
                                     RMSE = model_2_rmse ))
rmse_results %>% knitr::kable()

######################################################
### Refining the model

# check for the 10 largest mistakes to improve the model
validation %>% 
  left_join(movie_avgs, by="movieId") %>%
  mutate(residual = rating - (mu + b_i)) %>%
  arrange(desc(abs(residual))) %>% 
  dplyr::select(title, residual) %>% 
  distinct() %>%
  slice(1:10) %>% 
  knitr::kable()

# connect movieID with titles
movie_titles <- movielens %>% 
  filter(!is.na(title)) %>%
  dplyr::select(movieId, title) %>%
  distinct()

# 10 best rated movies
movie_avgs %>% left_join(movie_titles, by="movieId") %>%
  arrange(desc(b_i)) %>% 
  filter(!is.na(title)) %>%
  dplyr::select(title, b_i) %>% 
  slice(1:10) %>%   knitr::kable() 

# 10 worst movies
movie_avgs %>% left_join(movie_titles, by="movieId") %>%
  arrange(b_i) %>% 
  filter(!is.na(title)) %>%
  dplyr::select(title, b_i) %>% 
  slice(1:10) %>%  
  knitr::kable()

# how often are the best movies rated 
edx %>% count(movieId) %>% 
  left_join(movie_avgs) %>%
  left_join(movie_titles, by="movieId") %>%
  arrange(desc(b_i)) %>% 
  filter(!is.na(title)) %>%
  dplyr::select(title, b_i, n) %>% 
  slice(1:10) %>% 
  knitr::kable()

# how often are the worst movies rated
edx %>% count(movieId) %>% 
  left_join(movie_avgs) %>%
  left_join(movie_titles, by="movieId") %>%
  arrange(b_i) %>% 
  filter(!is.na(title)) %>%
  dplyr::select(title, b_i, n) %>% 
  slice(1:10) %>% 
  knitr::kable()

######################################################
### Regularization

# regularized estimates - exclude movies with less than 3 ratings
lambda <- 3
mu <- mean(edx$rating)
movie_reg_avgs <- edx %>% 
  group_by(movieId) %>% 
  summarize(b_i = sum(rating - mu)/(n()+lambda), n_i = n()) 

#plot of regularization to show regularized estimates vs. least square estimates
data_frame(original = movie_avgs$b_i, 
           regularlized = movie_reg_avgs$b_i, 
           n = movie_reg_avgs$n_i) %>%
  ggplot(aes(original, regularlized, size=sqrt(n))) + 
  geom_point(shape=1, alpha=0.5)

# top 10 based on lambda
edx %>%
  count(movieId) %>% 
  left_join(movie_reg_avgs) %>%
  left_join(movie_titles, by="movieId") %>%
  arrange(desc(b_i)) %>% 
  filter(!is.na(title)) %>%
  dplyr::select(title, b_i, n) %>% 
  slice(1:10) %>% 
  knitr::kable()

# 10 worst movies based on lambda
edx %>%
  count(movieId) %>% 
  left_join(movie_reg_avgs) %>%
  left_join(movie_titles, by="movieId") %>%
  arrange(b_i) %>% 
  filter(!is.na(title)) %>%
  dplyr::select(title, b_i, n) %>% 
  slice(1:10) %>% 
  knitr::kable()

# change of results compared to previous estimates of RMSE
predicted_ratings <- validation %>% 
  left_join(movie_reg_avgs, by='movieId') %>%
  mutate(pred = mu + b_i) %>%
  .$pred

model_3_rmse <- RMSE(predicted_ratings, validation$rating)
rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Regularized Movie Effect Model",  
                                     RMSE = model_2_rmse ))
rmse_results %>% 
  knitr::kable()

# choose lambda
lambdas <- seq(0, 10, 0.25)

mu <- mean(edx$rating)
just_the_sum <- edx %>% 
  group_by(movieId) %>% 
  summarize(s = sum(rating - mu), n_i = n())

rmses <- sapply(lambdas, function(l){
  predicted_ratings <- validation %>% 
    left_join(just_the_sum, by='movieId') %>% 
    mutate(b_i = s/(n_i+l)) %>%
    mutate(pred = mu + b_i) %>%
    .$pred
  return(RMSE(predicted_ratings, validation$rating))
})
qplot(lambdas, rmses)  
lambdas[which.min(rmses)]

# use cross-validation to pick a lambda
lambdas <- seq(0, 10, 0.25)

rmses <- sapply(lambdas, function(l){
  
  mu <- mean(edx$rating)
  
  b_i <- edx %>% 
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+l))
  
  b_u <- edx %>% 
    left_join(b_i, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n()+l))
  
  predicted_ratings <- 
    validation %>% 
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    mutate(pred = mu + b_i + b_u) %>%
    .$pred
  
  return(RMSE(predicted_ratings, validation$rating))
})

qplot(lambdas, rmses)  

# optimal lambda for the whole model 
lambda <- lambdas[which.min(rmses)]
lambda

######################################################
##Results

# show results of the model
rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Regularized Movie + User Effect Model",  
                                     RMSE = min(rmses)))
rmse_results %>% 
  knitr::kable()