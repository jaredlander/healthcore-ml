# Packages ####

library(ggplot2)
library(dplyr)
library(gglander)
library(rsample)
library(yardstick)
library(recipes)
library(parsnip)
library(workflows)
library(dials)
library(tune)
library(tictoc)
library(coefplot)
library(future)

# only available in R 4.5+
use('themis', c('step_downsample'))

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

# Feature Engineering for Penalized Regression ####

# {recipes}

credit
rec_glm_1 <- recipe(Status ~ ., data=train) |> 
    themis::step_downsample(Status, under_ratio = 1.2) |> 
    # themis::step_upsample()
    step_nzv(all_predictors()) |> 
    # step_naomit(all_predictors()) |> 
    step_impute_knn(all_numeric_predictors()) |> 
    # step_mutate(age_orig=Age) |> 
    step_cut(Age, breaks=c(5, 18, 35, 45)) |> 
    step_normalize(all_numeric_predictors()) |> 
    # step_discretize(Age, min_unique=5) |> 
    step_unknown(all_nominal_predictors(), new_level='missing') |> 
    step_other(all_nominal_predictors(), other='misc', threshold = 0.05) |> 
    step_novel(all_nominal_predictors(), new_level='unseen') |> 
    step_dummy(all_nominal_predictors(), one_hot = TRUE)
rec_glm_1

rec_glm_1 |> prep() |> bake(new_data = NULL)
rec_glm_1 |> prep() |> bake(new_data = NULL, -Status, composition = 'dgCMatrix')

# Model Specification for Penalized Regression ####

# {parsnip}

# lm(y ~ x, data)
# glmnet::glmnet(x, y)

linear_reg()
linear_reg(engine='lm')
linear_reg(engine='glmnet')
linear_reg(engine='stan')
linear_reg(engine='spark')
linear_reg(engine='keras')
linear_reg(engine='brulee')

rand_forest()
boost_tree()

logistic_reg()
logistic_reg(engine='glmnet')

survival_reg()
censored::survival_prob_coxnet()

gen_additive_mod()
# library(multilevelmod)

poisson_reg()

spec_glm_1 <- logistic_reg(engine='glmnet', penalty = tune(), mixture = tune())
spec_glm_1

# Combine Feature Engineering and Model Spec for Penalized Regression ####

# {workflows}

flow_1 <- workflow(preprocessor = rec_glm_1, spec = spec_glm_1)
flow_1

# Tuning Parameters for Penalized Regression ####

# {dials}

params_glm_1 <- flow_1 |> extract_parameter_set_dials()
params_glm_1$object

grid_glm_1 <- params_glm_1 |> 
    grid_random(size=100) |> 
    dplyr::mutate(mixture=round(mixture, digits=1))

# Tune the Penalized Regression Model ####

# {tune} and {tictoc}

tic(msg = 'glm1')
tune_glm_1 <- tune_grid(
    flow_1,
    resamples = cv_split,
    grid=grid_glm_1,
    metrics = loss_fn,
    control = control_grid(verbose = TRUE)
)
toc(log=TRUE)

tic.log()

tune_glm_1
tune_glm_1$.metrics[[1]]

tune_glm_1 |> collect_metrics()
tune_glm_1 |> collect_metrics() |> dplyr::filter(.metric == 'roc_auc')
tune_glm_1 |> autoplot()
tune_glm_1 |> autoplot(metric='roc_auc')

tune_glm_1 |> show_best(metric='roc_auc', n = 10)
tune_glm_1 |> select_best(metric = 'roc_auc')
tune_glm_1 |> select_by_one_std_err(metric = 'roc_auc', -penalty)
best_params_glm_1 <- tune_glm_1 |> select_by_one_std_err(metric = 'roc_auc', -penalty)
best_params_glm_1$penalty

tune_glm_1 |> show_best(metric='accuracy', n = 10)

flow_1

mod_glm_1 <- flow_1 |> 
    finalize_workflow(parameters = best_params_glm_1)
mod_glm_1

fitted_glm_1 <- fit(mod_glm_1, data=train)
fitted_glm_1

# visualize with {coefplot}
class(fitted_glm_1)
class(fitted_glm_1$fit$fit)
fitted_glm_1 |> 
    extract_fit_engine() |> 
    coefplot(lambda=best_params_glm_1$penalty, sort='magnitude', trans = exp)

# America/New_York

# Feature Engineering for Boosted Trees ####

# {recipes}

rec_xg_1 <- recipe(Status ~ ., data=train) |> 
    step_nzv(all_predictors()) |> 
    step_unknown(all_nominal_predictors(), new_level = 'missing') |> 
    step_other(all_nominal_predictors(), other = 'misc') |> 
    step_novel(all_nominal_predictors(), new_level = 'unseen') |> 
    step_dummy(all_nominal_predictors(), one_hot = TRUE)
rec_xg_1

# Model Spec for Boosted Trees ####

# {parsnip}

train |> dplyr::count(Status, sort = TRUE)
2560/1003

spec_xg_1 <- boost_tree(
    mode='classification', 
    trees=tune(),
    tree_depth = tune(),
    sample_size = 0.7,
    mtry=1,
    # mtry=0.7,
    stop_iter=30
) |> 
    set_engine(
        engine='xgboost',
        # num_parallel_trees=5
        scale_pos_weight=2.55,
        nthread=4
        # tree_method='gpu_hist',
        # device='cuda'
    )

spec_xg_1

# Workflows for Boosted Trees ####


# {workflows}

flow_xg_1 <- workflow(preprocessor = rec_xg_1, spec = spec_xg_1)
flow_xg_1

# Parameters for Boosted Trees ####

# {dials}

params_xg_1 <- flow_xg_1 |> 
    extract_parameter_set_dials() |> 
    update(
        trees=trees(range = c(5, 200)),
        tree_depth=tree_depth(range = c(2, 12))
    )

grid_xg_1 <- grid_space_filling(params_xg_1, type = 'max_entropy', size=100)
grid_xg_1

grid_xg_1 |> dplyr::count(tree_depth)

grid_xg_1 |> 
    dplyr::summarize(
        count=dplyr::n(), min_trees=min(trees), max_trees=max(trees),
        .by=tree_depth
    ) |> 
    dplyr::arrange(tree_depth)

# Tuning Boosted Trees ####

# {tune} and {future}

parallelly::availableWorkers()
parallelly::availableCores()

plan(multisession, workers=4)

tic(msg='xg1')
tune_xg_1 <- tune_grid(
    flow_xg_1,
    resamples = cv_split,
    grid=grid_xg_1,
    metrics=loss_fn,
    control = control_grid(verbose=TRUE, allow_par = FALSE, parallel_over = 'everything')
)
toc(log=TRUE)

plan(sequential)

tune_xg_1
tune_xg_1 |> show_notes()

tune_xg_1 |> collect_metrics()
tune_xg_1 |> autoplot(metric='roc_auc')
tune_xg_1 |> show_best(metric='roc_auc')
tune_glm_1 |> show_best(metric='roc_auc')

best_params_xg_1 <- tune_xg_1 |> select_by_one_std_err(metric='roc_auc', tree_depth)

mod_xg_1 <- finalize_workflow(flow_xg_1, parameters = best_params_xg_1)
mod_xg_1

fitted_xg_1 <- fit(mod_xg_1, data=train)
fitted_xg_1 |> 
    vip::vip()

# Compare Final Models to Each Other ####

# {tune}

eval_glm_1 <- last_fit(mod_glm_1, split = credit_split, metrics = loss_fn)
eval_xg_1 <- last_fit(mod_xg_1, split = credit_split, metrics = loss_fn)
eval_glm_1
eval_xg_1

dplyr::bind_rows(
    eval_glm_1 |> collect_metrics() |> dplyr::mutate(mod='glm_1'),
    eval_xg_1 |> collect_metrics() |> dplyr::mutate(mod='xg_1')
) |> 
    dplyr::filter(.metric == 'roc_auc')

# Use Winning Model ####

champion <- fit(mod_xg_1, data=credit)
champion

