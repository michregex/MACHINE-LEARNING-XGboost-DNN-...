library(dplyr)
library(keras)

data <- readxl::read_xlsx("datiquestionario.xlsx")
data <- clean_names(data)
data <- data[, -c(1, 6)]
data <- data %>% mutate(across(where(is.character), factor)) 

dmy <- dummyVars(" ~ .", data = data)
data_oh <- predict(dmy, newdata = data)

set.seed(123)
train_ind <- sample(1:nrow(data), 0.8 * nrow(data), replace = FALSE)

X <- data_oh[, -dim(data_oh)[2]] 
y <- data_oh[, dim(data_oh)[2]] 
X_train <- X[train_ind, ] |> as.matrix()
X_test <- X[-train_ind, ] |> as.matrix()
y_train <- y[train_ind] 
y_test <- y[-train_ind] 

# Hyperparameter flags ---------------------------------------------------

FLAGS <- flags(
  flag_integer("nodes1", 64),
  flag_integer("nodes2", 128),
  flag_integer("nodes3", 256),
  flag_numeric("dropout1", 0.4),
  flag_numeric("dropout2", 0.3),
  flag_numeric("dropout3", 0.2),
  flag_string("optimizer", "adam")
  )

# Define Model --------------------------------------------------------------

model <- keras_model_sequential() %>%
  layer_dense(units = FLAGS$nodes1, activation = "linear", input_shape = ncol(X_train)) %>%
  layer_dropout(rate = FLAGS$dropout1) %>%
  layer_dense(units = FLAGS$nodes2, activation = "linear") %>%
  layer_dropout(rate = FLAGS$dropout2) %>%
  layer_dense(units = FLAGS$nodes3, activation = "linear") %>%
  layer_dropout(rate = FLAGS$dropout3) %>%
  layer_dense(units = 10, activation = "linear") %>%
  compile(
    loss = "mse",
    metrics = c("mse"),
    optimizer = FLAGS$optimizer
  )

model %>% compile(
  loss = 'mse',
  optimizer = optimizer_adam(),
  metrics = c('mse')
)

# Training & Evaluation ----------------------------------------------------

history <- model %>% fit(
  X_train, y_train,
  batch_size = 32,
  epochs = 50,
  verbose = 0,
  validation_split = 0.2
)

plot(history)

score <- model %>% evaluate(
  X_test, y_test,
  verbose = 0
)

cat('Test loss:', score[1], '\n')
cat('Test mse:', score[2], '\n')