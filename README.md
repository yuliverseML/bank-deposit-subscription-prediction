# Bank Term Deposit Subscription Prediction | ML Classification Project

A comprehensive machine learning project that predicts bank term deposit subscriptions using the UCI Bank Marketing dataset.
The project implements and evaluates multiple classification algorithms, with optimized LightGBM achieving the best performance.

## Overview

This project aims to predict whether a client will subscribe to a bank term deposit based on marketing campaign data. It uses various machine learning techniques to analyze client demographics, previous campaign outcomes, and economic indicators to optimize marketing efforts.



## Dataset

This project uses the [UCI Bank Marketing dataset](https://archive.ics.uci.edu/ml/datasets/Bank+Marketing), which contains data from direct marketing campaigns of a Portuguese banking institution.

**Dataset characteristics:**
- 41,188 instances
- 20 input features (10 categorical, 10 numerical)
- Binary target variable (yes/no for term deposit subscription)
- Class imbalance (approximately 11% positive cases)

## Models Implemented

The project implements and evaluates several machine learning models:
- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting
- XGBoost
- LightGBM

## Features

### Data Exploration
- Comprehensive analysis of dataset shape and structure
- Missing value detection and handling
- Target variable distribution analysis
- Statistical analysis of numerical and categorical features
- Correlation analysis and multicollinearity detection
- Outlier identification

### Data Preprocessing
- Target encoding to binary format
- Train-test splitting with stratification
- Feature engineering:
  - Age group categorization
  - Call duration binning
  - Previous contact indicators
  - Contact rate calculation
- Stationarity tests for economic indicators
- Numerical feature scaling via StandardScaler
- Categorical feature encoding via OneHotEncoder
- Class imbalance handling with SMOTE

### Model Training
- Implementation of multiple classification algorithms
- Hyperparameter optimization via GridSearchCV
- SMOTE integration for handling class imbalance
- Cross-validation for robust model evaluation

### Model Evaluation
- Comprehensive metrics suite:
  - Accuracy
  - Precision
  - Recall
  - F1 Score
  - ROC AUC
- Confusion matrices for error analysis
- Detailed classification reports
- Performance comparison across models

### Visualization
- Target distribution visualization
- Numerical feature distribution plots
- Feature-target relationship analysis
- Correlation heatmaps
- Outlier detection via boxplots
- Model performance comparison charts
- ROC curves for classifier evaluation

## Results

### Initial Model Comparison
Performance metrics for models with default parameters:

| Model | Accuracy | Precision | Recall | F1 Score | ROC AUC |
|-------|----------|-----------|--------|----------|---------|
| XGBoost | 0.912 | 0.867 | 0.793 | 0.828 | 0.937 |
| LightGBM | 0.908 | 0.851 | 0.789 | 0.819 | 0.931 |
| Gradient Boosting | 0.894 | 0.825 | 0.782 | 0.803 | 0.918 |
| Random Forest | 0.886 | 0.803 | 0.764 | 0.783 | 0.904 |
| Logistic Regression | 0.822 | 0.712 | 0.674 | 0.692 | 0.847 |
| Decision Tree | 0.781 | 0.643 | 0.702 | 0.671 | 0.763 |

### Hyperparameter Optimization Results
After hyperparameter optimization, LightGBM achieved the best performance:

- **Final Model**: LightGBM
- **Optimal Parameters**:
  - learning_rate: 0.1
  - max_depth: 5
  - n_estimators: 100
  - num_leaves: 31
- **ROC AUC on Test Set**: 0.9541

### Feature Importance
The most predictive features for term deposit subscription include:
- Call duration (longer calls indicating higher interest)
- Economic indicators (employment variation rate, consumer confidence)
- Contact history (previous campaign outcomes)
- Client demographics (age, education, marital status)

## Outcome

### Best Performing Model
LightGBM with optimized hyperparameters is the best performing model, achieving an impressive ROC AUC score of 0.9541. This represents excellent discrimination between subscribers and non-subscribers, significantly outperforming all baseline models.

Sample predictions with the optimized LightGBM model:

```
       Actual  Predicted  Probability  age           job            education  duration
27129       0          0        0.012   29        admin.          high.school       140
29090       0          0        0.162   41      services          high.school       464
11825       0          0        0.001   50  entrepreneur             basic.4y        20
21477       0          0        0.007   30    technician  professional.course       149
6091        0          0        0.002   37   blue-collar  professional.course       137
```

The optimized LightGBM model offers:
- Exceptional discrimination ability (ROC AUC: 0.9541)
- Well-balanced precision and recall
- Effective handling of the class imbalance problem
- Efficient processing of both numerical and categorical features

## Future Work

Potential improvements and extensions include:
- Advanced feature engineering techniques
- Deep learning model implementation
- More extensive hyperparameter optimization
- Cost-sensitive learning approaches
- Deployment as a real-time prediction service
- A/B testing for marketing campaign optimization
- Time series analysis of subscription patterns
- Explainable AI integration for model interpretation
- Feature selection to reduce model complexity

## Notes

- The dataset exhibits significant class imbalance, making ROC AUC and precision-recall metrics more informative than accuracy
- Economic indicators show strong correlation with subscription rates, suggesting timing of campaigns is crucial
- Hyperparameter optimization significantly improved model performance, highlighting its importance in model development
- Call duration is a strong predictor but may not be available before making the call, suggesting a two-stage prediction approach could be valuable
- Model performance should be evaluated in the context of business objectives (e.g., cost of contact vs. value of subscription)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.


## License

This project is licensed under the MIT License - see the LICENSE file for details.
