library(dplyr)
library(caret)
library(janitor)
library(xgboost)
library(glmnet)
library(h2o)
library(keras)
library(tfruns)
library(vip)

# carica i dati 
data <- readxl::read_xlsx("datiquestionario2.xlsx")
data <- clean_names(data)
data <- data[, -c(1, 6)]
data <- data %>% mutate(across(where(is.character), factor)) 

# one hot encoding
dmy <- dummyVars(" ~ .", data = data)
data_oh <- predict(dmy, newdata = data)

set.seed(123)
# train-test splitting
train_ind <- sample(1:nrow(data), 0.8 * nrow(data), replace = FALSE)

X <- data_oh[, -dim(data_oh)[2]] 
y <- data_oh[, dim(data_oh)[2]] 
X_train <- X[train_ind, ] |> as.matrix()
X_test <- X[-train_ind, ] |> as.matrix()
y_train <- y[train_ind] 
y_test <- y[-train_ind] 

#### xgboost ####
# grid search
hyper_grid <- expand.grid(eta = c(0.01, 0.02, 0.05), max_depth = c(3, 10),
                          min_child_weight = 3, subsample = 0.5, colsample_bytree = 0.5,
                          gamma = c(0, 0.5 , 10),
                          lambda = c(0, 1e-2, 0.1, 1, 10),
                          alpha = c(0, 1e-2, 0.1, 1, 10),
                          rmse = 0, # a place to dump rmse results
                          trees = 0)# a place to dump required number of trees

# takes a while
for(i in seq_len(nrow(hyper_grid))) {
  cat("Iterazione:", i, "/",nrow(hyper_grid), "\n")
  grades_gb_cv_grid <- xgb.cv(data = X_train, label = y_train, nrounds = 1000, 
                              objective = "reg:squarederror", 
                              early_stopping_rounds = 50,
                              nfold = 5, params = list(
                                eta = hyper_grid$eta[i],
                                max_depth = hyper_grid$max_depth[i],
                                min_child_weight = hyper_grid$min_child_weight[i],
                                subsample = hyper_grid$subsample[i],
                                colsample_bytree = hyper_grid$colsample_bytree[i],
                                gamma = hyper_grid$gamma[i], lambda = hyper_grid$lambda[i],
                                alpha = hyper_grid$alpha[i]),verbose = 0)
  hyper_grid$rmse[i] <- min(grades_gb_cv_grid$evaluation_log$test_rmse_mean)
  hyper_grid$trees[i] <- grades_gb_cv_grid$best_iteration
}

# results order for the lowest mrse
hyper_grid %>%
  filter(rmse > 0) %>%
  arrange(rmse) %>%
  glimpse()
# load("app//MLmodel.RData")

# best model
xgb_cv <- xgb.cv(data = X_train, label = y_train, 
              eta = 0.02, max_depth = 3, min_child_weight = 3,
              subsample = 0.5, colsample_bytree = 0.5, gamma = 0,
              lambda = 0, alpha = 0.10, nrounds = 1000, nfold = 5, 
               objective = "reg:squarederror")
which.min(xgb_cv$evaluation_log$test_rmse_mean)

# plot for cross validation results
dati.cv <- data.frame(Train = xgb_cv$evaluation_log$train_rmse_mean,
                      Test = xgb_cv$evaluation_log$test_rmse_mean,
                      Iter = 1:length(xgb_cv$evaluation_log$iter))
figura5 <- dati.cv %>% 
  ggplot() + 
  geom_line(aes(x = Iter, y = Train, col = "Train"), linewidth = 1) + 
  geom_line(aes(x = Iter, y = Test, col = "Test"), linewidth = 1) +
  ylab("RMSE") + 
  scale_color_manual(values = c("Train" = "black", "Test" = "red"), 
                     name = " ") +
  theme_bw() + 
  theme(legend.position = "right")

set.seed(123)
xgb_best <- xgboost(data = X_train, label = y_train, 
              eta = 0.02, max_depth = 3, min_child_weight = 3,
              subsample = 0.5, colsample_bytree = 0.5, gamma = 0,
              lambda = 0, alpha = 0.10, nrounds = 882, 
               objective = "reg:squarederror")

pred <- predict(xgb_best, X_test)
sum((y_test - pred)^2) / length(pred) #mse sulle osservazioni rimaste

# variable importance plot
importanceDT <- xgb.importance(feature_names = colnames(X), model = xgb_best, data = X, label = Y)
figura8 <- data.frame(importanceDT) %>% 
  mutate(Feature = forcats::fct_reorder(Feature, Gain)) %>%
  filter(Gain >= 0.02) %>% 
  ggplot(aes(x=Feature, y=Gain)) +
  geom_segment(aes(xend=Feature, yend=0)) + 
  geom_point(size=3, color="orange") + 
  coord_flip() + 
  theme_bw() +
  theme(axis.text = element_text(family = "mono", color = "darkblue",size = "8" ),
        axis.line.y = element_line(linetype = 1,color="black",linewidth = 1))


#### glmnet ####
# grid search for regularized regression
cv_glmnet <- train(
  x = X_train,
  y = y_train,
  method = "glmnet",
  preProc = c("zv", "center", "scale"),
  trControl = trainControl(method = "cv", number = 5),
  tuneLength = 10,
  standardize = FALSE
)
cv_glmnet$bestTune

glmnet_fit <- glmnet(X_train, y_train, alpha = 0.1, lambda = 3.079553)
assess.glmnet(glmnet_fit, newx = X_test, newy = y_test)
vip(glmnet_fit)

# con cv di glmnet
cv.fit <- cv.glmnet(X_train, y_train, alpha = 0.1)
cv.fit$lambda.min
plot(cv.fit)

# grafico cross validation
cv.fit_dati <- data.frame(lmb = cv.fit$lambda,
                          low = cv.fit$cvlo, mid = cv.fit$cvm, up = cv.fit$cvup)
figura6 <- cv.fit_dati %>% ggplot(aes(x = log(lmb), y = mid)) + 
  geom_point(shape = 23, col = "red", fill = "red") + 
  geom_errorbar(aes(x = log(lmb), ymin = low, ymax = up), lwd = 0.5) +
  labs(y = "MSE", x = expression(log(lambda))) + 
  scale_y_continuous(breaks = c(150, 200, 250, 300, 350)) +
  theme_bw() 

#### h2o Neural Network ####
h2o.init()
h2o_data <- as.h2o(data_oh)
h2o_train <- data_oh[train_ind, ] |> as.h2o()
h2o_test <- data_oh[-train_ind, ] |> as.h2o()

hyper_grid <- list(hidden = list(c(20, 15, 5), c(5, 15, 20), c(100, 50, 25, 5), 
                                  c(5, 25, 50, 100), c(50, 25, 10, 25, 50),
                                  c(10, 25, 50, 25, 10), c(100, 50, 25, 10)),
                   l1 = c(0,1e-3,1e-5),
                   l2 = c(0,1e-3,1e-5))

ae_grid <- h2o.grid(x = c(1:183),
                    y = "livello",
                    distribution = "gamma",
                    epochs = 100,
                    train_samples_per_iteration = -1,
                    activation = "Tanh",
                    seed = 23123,
                    stopping_rounds = 0,
                    training_frame = h2o_train,
                    algorithm = "deeplearning",
                    grid_id = "grid",
                    hyper_params = hyper_grid,
                    nfolds = 5)

#get best model
d_grid <- h2o.getGrid("grid", sort_by = "mse")
best_dl_model <- h2o.getModel(d_grid@model_ids[[1]])
h2o.performance(best_dl_model, xval = T)

best_dnn <- h2o.deeplearning(x = c(1:183),
                    y = "livello",
                    distribution = "gamma",
                    epochs = 100,
                    train_samples_per_iteration = -1,
                    activation = "Tanh",
                    seed = 23123,
                    stopping_rounds = 0,
                    training_frame = h2o_train,
                    nfolds = 5,
                    hidden = c(100, 50, 25, 5),
                    l1 = 0.00001, l2 = 0.00100)
pred_dnn <- h2o.predict(best_dnn, h2o_test)
sum((y_test - pred_dnn)^2) / nrow(data)

#### DNN with Keras ####
# set up a training run
training_run("trainingRun.R")

# grid search
runs <- tuning_run("trainingRun.R",
  flags = list(
    nodes1 = c(64, 128, 256, 512),
    nodes2 = c(64, 128, 256),
    nodes3 = c(64, 128, 256),
    dropout1 = c(0.2, 0.4),
    dropout2 = c(0.2, 0.4),
    dropout3 = c(0.2, 0.4),
    optimizer = "adam")
  # ,sample = 0.05
)

runs[order(runs$metric_mse, decreasing = FALSE), ]
view_run(runs[order(runs$metric_mse, decreasing = FALSE), ][1,1])

# fit best model
model <- keras_model_sequential() %>%
  layer_dense(units = 128, activation = "linear", input_shape = ncol(X_train)) %>%
  layer_dropout(rate = 0.2) %>%
  layer_dense(units = 256, activation = "linear") %>%
  layer_dropout(rate = 0.2) %>%
  layer_dense(units = 256, activation = "linear") %>%
  layer_dropout(rate = 0.2) %>%
  layer_dense(units = 10, activation = "linear")

# compile the model (backpropagation)
model %>% compile(
  loss = 'mse',
  optimizer = optimizer_adam(),
  metrics = c('mse')
)

# fit the model
best_dnn <- model %>% fit(
  X_train, y_train,
  batch_size = 32,
  epochs = 50,
  verbose = 0,
  validation_split = 0.2
)

figura7 <- plot(best_dnn, metrics = "mse") + theme_bw()

# prediction on the test set
score <- model %>% evaluate(
  X_test, y_test,
  verbose = 0
)

cat('Test loss:', score[1], '\n')
cat('Test mse:', score[2], '\n')
