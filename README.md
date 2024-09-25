# Predictive Machine Learning Models for Questionnaire Data

This project focuses on the analysis of a dataset collected from a questionnaire, using advanced machine learning techniques to predict an outcome variable. The analysis includes several modeling approaches, including **Gradient Boosting**, **Regularized Regression (GLMNet)**, and **Deep Neural Networks (DNN)**, aiming to optimize model performance and interpretability.

## Project Overview

The project involves the following key steps:

1. **Data Preprocessing**:
    - Cleaned and transformed the dataset using `dplyr` and `janitor` to prepare it for modeling.
    - Applied **One-Hot Encoding** using `dummyVars` from the `caret` package to handle categorical variables.
    - Split the dataset into training and test sets (80/20 ratio) to ensure reliable evaluation of the models.

2. **Modeling Techniques**:
    - **Gradient Boosting with XGBoost**: 
        - A grid search was performed to tune the model's hyperparameters (e.g., learning rate `eta`, `max_depth`, `lambda`, `alpha`, etc.).
        - Cross-validation was used to identify the optimal number of trees.
        - The model's performance was evaluated based on the RMSE, and results were visualized using `ggplot2`.

    - **Regularized Regression (GLMNet)**:
        - This model utilized `glmnet` with elastic net regularization to handle multicollinearity and perform feature selection.
        - Hyperparameters such as `lambda` and `alpha` were tuned using cross-validation, and the results were compared to the other models.

    - **Deep Neural Networks (DNN)**:
        - A DNN was implemented using the `Keras` library, with a custom architecture and grid search for the number of nodes and dropout rates.
        - Regularization techniques (dropout) were applied to avoid overfitting.
        - The performance was evaluated based on MSE, and training history was plotted.

3. **H2O Neural Networks**:
    - A deep learning model was also built using the `H2O` framework with a hyperparameter grid search. This model was compared to the `Keras`-based neural network to evaluate performance differences.

4. **Model Evaluation**:
    - All models were evaluated using **Mean Squared Error (MSE)** on the test set.
    - A comparison of the models in terms of MSE, computational time, and tuning complexity was performed, with **XGBoost** showing the best performance.

5. **Feature Importance**:
    - For the **XGBoost** model, a feature importance plot was generated to identify the most significant predictors, providing insights into the factors driving the predictions.

## Results Summary

| Model                 | MSE   | Training Time | Complexity   |
|-----------------------|-------|---------------|--------------|
| **XGBoost**           | 74.60 | 3 hours       | **Medium**   |
| **GLMNet (Elastic Net)**| 77.80 | 1 minute      | **Easy**     |
| **DNN (Keras)**        | 111.51| 1-2 hours     | **Hard**     |

The best-performing model was **XGBoost**, which balanced complexity, accuracy, and computational efficiency. Feature importance analysis revealed that certain lifestyle factors, such as sports activity frequency and health indicators, were key drivers of the predictions.

## Setup Instructions

1. Clone the repository.
2. Install the required R packages using the command below:
    ```R
    install.packages(c('dplyr', 'caret', 'xgboost', 'glmnet', 'keras', 'h2o', 'vip', 'ggplot2'))
    ```
3. Run the analysis script to train the models and evaluate performance.
