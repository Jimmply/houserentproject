# houserentproject

Advanced Modeling & Hyperparameter Tuning
House Rent Price Prediction
1. Project Objective

This notebook extends the work completed in the baseline modeling notebook and demonstrates a more advanced, industry-level machine learning workflow.

Many real-world ML tasks require more than a simple train/test split and a single model. Recruiters and hiring managers often look for:

Rigorous cross-validation for reliable model assessment

Use of advanced algorithms such as XGBoost

Hyperparameter tuning to maximize predictive power

Clear model comparison and justification of choices

Thoughtful interpretation of results

This notebook delivers exactly that.

Specifically, we will:

Reuse the preprocessing pipeline defined previously

Evaluate a baseline Random Forest using 5-fold cross-validation

Train an advanced XGBoost regression model

Apply RandomizedSearchCV to tune key hyperparameters

Compare the final models on test data

Inspect and interpret XGBoost’s most important features

By the end of this notebook, we will have a robust and production-ready modeling workflow.

2. Data & Preprocessing Summary

The dataset used originates from the House Rent Prediction dataset on Kaggle and contains features describing rental properties across multiple Indian cities.

Key feature categories include:

Numerical: Size (sqft), BHK, Bathroom count

Categorical: City, Area Type, Furnishing Status, Tenant Preferred

These features are processed using a ColumnTransformer:

Numerical features → passed through unchanged

Categorical features → encoded using OneHotEncoder

This ensures both model consistency and compatibility across all algorithms used.

3. Baseline Model: Random Forest with Cross-Validation

Before introducing more complex models, it is important to establish a strong baseline.

Why cross-validation?

A single train/test split may give a misleading estimate of performance due to randomness.
K-fold cross-validation (here, k=5):

Trains the model on different subsets

Tests on unique folds

Produces a more stable and trustworthy performance estimate

Reduces overfitting to a particular split

Why Random Forest?

Random Forest is a robust baseline:

Naturally handles non-linear relationships

Resistant to noise

Performs well across many tabular datasets

Requires minimal tuning for decent performance

We report metrics using:

MAE (Mean Absolute Error)

RMSE (Root Mean Squared Error)

R² (Coefficient of Determination)

These metrics provide a comprehensive view of prediction accuracy.

4. Advanced Model: XGBoost

XGBoost is one of the strongest machine learning algorithms for structured/tabular data.

Why XGBoost?

Excellent performance due to gradient boosting

Built-in support for missing values

Fast training via histogram optimization

Flexible and highly tunable

Widely used in industry and Kaggle competitions

We begin with a base model, then significantly improve it through hyperparameter tuning.

5. Hyperparameter Tuning with RandomizedSearchCV

Rather than performing an exhaustive grid search (which can be very slow), we use RandomizedSearchCV, which:

Samples parameter combinations

Finds strong configurations faster

Uses 5-fold CV on each sampled configuration

Key parameters tuned:

n_estimators — number of boosting rounds

max_depth — model complexity

learning_rate — step size

subsample — row sampling ratio

colsample_bytree — feature sampling ratio

The tuning process identifies a high-performing model without excessive computational cost.

6. Final Model Comparison

After training both models:

Random Forest (baseline)

XGBoost (tuned)

we evaluate them on the held-out test set.

Metrics compared:

Metric	Random Forest	XGBoost (tuned)
MAE	lower is better	lower is better
RMSE	lower is better	lower is better
R²	higher is better	higher is better

The tuned XGBoost model is expected to outperform the baseline across all metrics thanks to:

Gradient boosting

Hyperparameter optimization

Better ability to capture complex relationships

A bar chart summarizing the RMSE comparison is also included for clarity.

7. Feature Importance Analysis

Understanding why the model makes predictions is essential, especially in real-world ML applications.

For XGBoost:

Feature importance scores highlight which variables contribute most to the model

Helps validate assumptions seen during EDA

Provides actionable insights (e.g., which property characteristics impact rent most)

Typical top features include:

Size — larger properties tend to rent for more

City_ features* — location strongly influences pricing

Bathroom / BHK — more amenities correlate with higher rent

Including this analysis demonstrates your ability to interpret ML models beyond just training them.

8. Final Conclusion & Professional Summary

In this notebook, we implemented an advanced machine learning workflow suitable for real hiring scenarios:

Conducted cross-validation for robust model evaluation

Built and tuned a competitive XGBoost model

Compared multiple models using test metrics

Identified key drivers of rent pricing through feature importance

Key achievements:

✔ Built reproducible ML pipelines
✔ Applied industry-standard evaluation techniques
✔ Performed hyperparameter tuning professionally
✔ Communicated results clearly and effectively

Potential next steps:

Explore LightGBM or CatBoost as alternatives

Perform full GridSearchCV for deeper tuning

Add SHAP explainability for richer interpretation

Deploy the model using FastAPI or Streamlit

Build a real-time rent prediction dashboard
