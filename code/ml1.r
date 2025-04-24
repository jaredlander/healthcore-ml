# Packages ####

library(ggplot2)
library(dplyr)
library(gglander)
library(rsample)
library(yardstick)
library(recipes)

# Setttings ####
options(tidymodels.dark=TRUE)

# Data ####

data(credit_data, package = 'modeldata')
credit <- credit_data |> tibble::as_tibble()
credit
?modeldata::credit_data

# EDA ####

# {ggplot2} and {dplyr}

credit |> count(Marital)
credit |> count(Home)
credit |> count(Job)
credit |> count(Status)

ggplot(credit, aes(x=Status, fill=Status)) + geom_bar()

ggplot(credit, aes(x=Status, y=Amount)) + geom_violin()

ggplot(credit, aes(x=Income, y=Amount)) + geom_point()
ggplot(credit, aes(x=log(Income), y=log(Amount))) + geom_point()
ggplot(credit, aes(x=log(Income), y=log(Amount))) + geom_point() + geom_smooth()
ggplot(credit, aes(x=log(Income), y=log(Amount), color=Job)) + geom_point() + geom_smooth()
ggplot(credit, aes(x=log(Income), y=log(Amount), color=Job)) + 
    geom_point(show.legend = FALSE) + 
    geom_smooth(show.legend = FALSE) + 
    facet_wrap(~Job)
ggplot(credit, aes(x=log(Income), y=log(Amount), color=Job)) + 
    geom_jitter(shape=10, size=1, alpha=2/3, show.legend = FALSE) + 
    geom_smooth(show.legend = FALSE) + 
    facet_wrap(~Job)

# ML Process Setup ####

# - Create a train/test split
# - Setup cross-validation
# - Choose a loss function (performance metric like RMSE, MAE, Accuracy)

## Split the Data ####

# {rsample}

set.seed(1234)
credit_split <- initial_split(credit, prop = 0.8, strata = 'Status')
credit_split
credit_split$data
credit_split$in_id
credit_split$out_id
credit_split$id
credit[credit_split$in_id, ]

train <- training(credit_split)
test <- testing(credit_split)

initial_validation_split(credit_data)

## Cross-Validation ####

# {rsample}

cv_split <- vfold_cv(train, v=5, repeats=2, strata='Status')
cv_split
cv_split$splits[[1]]
cv_split$splits[[1]]

## Choose Loss Function ####

# {yardstick}

loss_fn <- metric_set(accuracy, roc_auc, mn_log_loss)
loss_fn

# Penlized Regression ####

# regularized regression
# L1 (lasso) regression
# L2 (ridge) regression
# elastic net

# Feature Engineering ####

# {recipes}

credit
rec_glm_1 <- recipe(Status ~ ., data=train) |> 
    themis::step_downsample(Status, under_ratio = 1.2) |> 
    step_nzv(all_predictors()) |> 
    # step_naomit(all_predictors()) |> 
    step_impute_knn(all_numeric_predictors()) |> 
    # step_mutate(age_orig=Age) |> 
    step_cut(Age, breaks=c(5, 18, 35, 45)) |> 
    step_normalize(all_numeric_predictors()) |> 
    # step_discretize(Age, min_unique=5) |> 
    step_unknown(all_nominal_predictors(), new_level='missing') |> 
    step_other(all_nominal_predictors(), other='misc') |> 
    step_novel(all_nominal_predictors(), new_level='unseen') |> 
    step_dummy(all_nominal_predictors(), one_hot = TRUE)
rec_glm_1

rec_glm_1 |> prep() |> bake(new_data = NULL)
