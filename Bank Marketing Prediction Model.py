# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import io
import zipfile
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report, roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import xgboost as xgb
from xgboost import XGBClassifier
try:
    from lightgbm import LGBMClassifier
except ImportError:
    print("LightGBM not available, will skip this model")
try:
    from imblearn.over_sampling import SMOTE
except ImportError:
    print("imbalanced-learn not available, will skip SMOTE")
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

# 1. Data Loading
def load_bank_data():
    """Download and load the UCI Bank Marketing dataset"""
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank-additional.zip"
    response = requests.get(url)
    zip_file = zipfile.ZipFile(io.BytesIO(response.content))

    # Print available files in the zip
    print("Files in the zip archive:")
    for file in zip_file.namelist():
        if file.endswith('.csv'):
            print(file)

    # Extract the full dataset file
    data_file = 'bank-additional/bank-additional-full.csv'  # Using the full dataset
    with zip_file.open(data_file) as file:
        data = pd.read_csv(file, sep=';')

    return data

# Load the data
data = load_bank_data()

# 2. Data Exploration
print(f"\nDataset shape: {data.shape}")
print("\nFirst 5 rows:")
print(data.head())

# Check for missing values
print("\nMissing values per column:")
print(data.isnull().sum())

# Examine target variable distribution
print("\nTarget variable distribution:")
print(data['y'].value_counts())
print(data['y'].value_counts(normalize=True).round(4))

# Visualize target distribution
plt.figure(figsize=(8, 6))
sns.countplot(x='y', data=data)
plt.title('Target Variable Distribution')
plt.tight_layout()
plt.savefig('target_distribution.png')
plt.close()

# 3. Analyze numerical features
plt.figure(figsize=(15, 10))
numerical_cols = data.select_dtypes(include=['int64', 'float64']).columns
for i, col in enumerate(numerical_cols):
    plt.subplot(3, 4, i+1)
    sns.histplot(data[col], kde=True)
    plt.title(f'Distribution of {col}')
plt.tight_layout()
plt.savefig('numerical_distributions.png')
plt.close()

# Analyze relationship between numerical features and target
plt.figure(figsize=(15, 10))
for i, col in enumerate(numerical_cols):
    plt.subplot(3, 4, i+1)
    sns.boxplot(x='y', y=col, data=data)
    plt.title(f'{col} vs Target')
plt.tight_layout()
plt.savefig('numerical_vs_target.png')
plt.close()

# Analyze categorical features - FIXED
categorical_cols = data.select_dtypes(include='object').columns
num_cat_cols = len(categorical_cols)

# Calculate appropriate grid dimensions
n_rows, n_cols = 3, 4  # Set to handle up to 12 categorical variables

plt.figure(figsize=(n_cols*4, n_rows*3))
print(f"Number of categorical columns: {num_cat_cols}")
print(f"Using subplot grid: {n_rows}x{n_cols}")

for i, col in enumerate(categorical_cols):
    if i < n_rows * n_cols:  # Only plot if we have space
        plt.subplot(n_rows, n_cols, i+1)

        # Fixed approach - calculate percentages manually
        cross_tab = pd.crosstab(data[col], data['y'])
        yes_pct = cross_tab['yes'] / cross_tab.sum(axis=1)

        sns.barplot(x=yes_pct.index, y=yes_pct.values)
        plt.title(f'% Yes for {col}')
        plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('categorical_vs_target.png')
plt.close()

# Correlation analysis for numerical features
plt.figure(figsize=(12, 10))
correlation_matrix = data.select_dtypes(include=['int64', 'float64']).corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix of Numerical Features')
plt.tight_layout()
plt.savefig('correlation_matrix.png')
plt.close()

# Detect outliers in numerical features
plt.figure(figsize=(15, 10))
for i, col in enumerate(numerical_cols):
    plt.subplot(3, 4, i+1)
    sns.boxplot(data[col])
    plt.title(f'Boxplot for {col}')
plt.tight_layout()
plt.savefig('outliers_boxplots.png')
plt.close()

# 4. Data Preprocessing and Feature Engineering
# Convert target to binary format
data['y_binary'] = data['y'].map({'yes': 1, 'no': 0})

# Split features and target
X = data.drop(['y', 'y_binary'], axis=1)
y = data['y_binary']

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Training set shape: {X_train.shape}")
print(f"Testing set shape: {X_test.shape}")

# Identify numerical and categorical columns
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_cols = X.select_dtypes(include='object').columns.tolist()

print(f"Numerical columns: {numerical_cols}")
print(f"Categorical columns: {categorical_cols}")

# Create new features
# Age groups
X_train['age_group'] = pd.cut(X_train['age'], bins=[0, 30, 40, 50, 60, 100],
                             labels=['<30', '30-40', '40-50', '50-60', '>60'])
X_test['age_group'] = pd.cut(X_test['age'], bins=[0, 30, 40, 50, 60, 100],
                            labels=['<30', '30-40', '40-50', '50-60', '>60'])

# Duration bins (call duration in seconds)
X_train['duration_bin'] = pd.cut(X_train['duration'], bins=[0, 100, 300, 600, 1000, 5000],
                                labels=['very_short', 'short', 'medium', 'long', 'very_long'])
X_test['duration_bin'] = pd.cut(X_test['duration'], bins=[0, 100, 300, 600, 1000, 5000],
                               labels=['very_short', 'short', 'medium', 'long', 'very_long'])

# Has previous contact
X_train['has_previous_contact'] = (X_train['previous'] > 0).astype(int)
X_test['has_previous_contact'] = (X_test['previous'] > 0).astype(int)

# Contact rate (safely handling division by zero)
X_train['contact_rate'] = X_train['campaign'] / X_train['previous'].replace(0, 1)
X_test['contact_rate'] = X_test['campaign'] / X_test['previous'].replace(0, 1)

# Update the categorical and numerical columns lists
new_categorical_cols = categorical_cols + ['age_group', 'duration_bin']
new_numerical_cols = numerical_cols + ['has_previous_contact', 'contact_rate']

# Test for white noise and stationarity in economic indicators
print("\nStationarity Tests for Economic Indicators:")
for col in ['emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed']:
    if col in X_train.columns:  # Check column exists
        # Safely run adfuller test
        try:
            result = sm.tsa.stattools.adfuller(X_train[col].dropna())
            print(f"{col}: p-value = {result[1]:.4f} {'(stationary)' if result[1] < 0.05 else '(non-stationary)'}")
        except Exception as e:
            print(f"Error testing {col}: {e}")

# Create preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]), new_numerical_cols),
        ('cat', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ]), new_categorical_cols)
    ])

# Apply preprocessing
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)
print(f"Processed training set shape: {X_train_processed.shape}")
print(f"Processed testing set shape: {X_test_processed.shape}")

# Apply SMOTE for handling class imbalance if available
try:
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train_processed, y_train)
    print(f"SMOTE training set shape: {X_train_smote.shape}")
    print(f"Original class distribution: {pd.Series(y_train).value_counts(normalize=True)}")
    print(f"SMOTE class distribution: {pd.Series(y_train_smote).value_counts(normalize=True)}")
    use_smote = True
except:
    print("Skipping SMOTE - continuing with original imbalanced data")
    X_train_smote, y_train_smote = X_train_processed, y_train
    use_smote = False

# 5. Model Training and Evaluation
# Function to evaluate models
def evaluate_model(model, X_train, X_test, y_train, y_test, model_name=None):
    """Train and evaluate a model, returning performance metrics"""
    # Train the model
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    conf_matrix = confusion_matrix(y_test, y_pred)

    # Print detailed results
    print(f"\n{model_name if model_name else model.__class__.__name__} Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")
    print("\nConfusion Matrix:")
    print(conf_matrix)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    return {
        'model': model_name if model_name else model.__class__.__name__,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'confusion_matrix': conf_matrix,
        'y_pred_proba': y_pred_proba,
        'model_object': model
    }

# Define models to evaluate
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42),
    'XGBoost': XGBClassifier(random_state=42)
}

# Add LightGBM if available
try:
    models['LightGBM'] = LGBMClassifier(random_state=42)
except:
    pass

# Train and evaluate each model
results = []
for name, model in models.items():
    print(f"\nTraining {name}...")
    # Use SMOTE-resampled data if available
    result = evaluate_model(model,
                           X_train_smote if use_smote else X_train_processed,
                           X_test_processed,
                           y_train_smote if use_smote else y_train,
                           y_test, name)
    results.append(result)

# Create a results dataframe
results_df = pd.DataFrame([
    {
        'Model': r['model'],
        'Accuracy': r['accuracy'],
        'Precision': r['precision'],
        'Recall': r['recall'],
        'F1 Score': r['f1_score'],
        'ROC AUC': r['roc_auc']
    } for r in results
])

# Sort by ROC AUC
results_df = results_df.sort_values('ROC AUC', ascending=False)
print("\nModel Performance Comparison (sorted by ROC AUC):")
print(results_df)

# Visualize model comparison
plt.figure(figsize=(12, 8))
sns.barplot(x='Model', y='ROC AUC', data=results_df)
plt.title('Model Performance Comparison (ROC AUC)')
plt.xlabel('Model')
plt.ylabel('ROC AUC Score')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('model_comparison.png')
plt.close()

# Plot ROC curves for all models
plt.figure(figsize=(10, 8))
for result in results:
    fpr, tpr, _ = roc_curve(y_test, result['y_pred_proba'])
    plt.plot(fpr, tpr, label=f"{result['model']} (AUC = {result['roc_auc']:.3f})")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves for All Models')
plt.legend()
plt.grid(True)
plt.savefig('roc_curves.png')
plt.close()

# 6. Hyperparameter Tuning for Best Model
# Get the best model based on ROC AUC
best_model_name = results_df.iloc[0]['Model']
best_model_index = next(i for i, r in enumerate(results) if r['model'] == best_model_name)
best_model_class = results[best_model_index]['model_object'].__class__

print(f"\nTuning hyperparameters for best model: {best_model_name}")

# Define hyperparameter grid based on the best model
if best_model_name == 'XGBoost':
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [3, 5],
        'learning_rate': [0.01, 0.1],
        'subsample': [0.8, 1.0]
    }
    grid_model = XGBClassifier(random_state=42)
elif best_model_name == 'Random Forest':
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 10],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }
    grid_model = RandomForestClassifier(random_state=42)
elif best_model_name == 'Gradient Boosting':
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [3, 5],
        'learning_rate': [0.01, 0.1],
        'subsample': [0.8, 1.0]
    }
    grid_model = GradientBoostingClassifier(random_state=42)
elif best_model_name == 'LightGBM':
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [3, 5],
        'learning_rate': [0.01, 0.1],
        'num_leaves': [31, 50]
    }
    grid_model = LGBMClassifier(random_state=42)
else:  # Default to logistic regression
    param_grid = {
        'C': [0.1, 1, 10],
        'penalty': ['l2'],
        'solver': ['liblinear', 'saga']
    }
    grid_model = LogisticRegression(max_iter=1000, random_state=42)

# Perform grid search with cross-validation
grid_search = GridSearchCV(
    grid_model, param_grid, cv=3, scoring='roc_auc', n_jobs=-1, verbose=1
)
grid_search.fit(X_train_processed, y_train)  # Use original data for tuning

# Get best parameters and model
best_params = grid_search.best_params_
print(f"Best parameters: {best_params}")

# Evaluate the optimized model
optimized_model = grid_search.best_estimator_
optimized_result = evaluate_model(
    optimized_model, X_train_processed, X_test_processed,
    y_train, y_test, f"Optimized {best_model_name}"
)

# Compare with SMOTE-balanced data
if use_smote:
    optimized_result_smote = evaluate_model(
        optimized_model, X_train_smote, X_test_processed,
        y_train_smote, y_test, f"Optimized {best_model_name} with SMOTE"
    )

# 7. Feature importance for the optimized model
if hasattr(optimized_model, 'feature_importances_'):
    # Get feature names after preprocessing (a bit tricky)
    feature_names = []

    # Add numerical feature names directly
    feature_names.extend(new_numerical_cols)

    # For categorical features, we need to consider one-hot encoding
    # This is an approximation since exact feature names after one-hot encoding are not easily accessible
    for col in new_categorical_cols:
        # If the column is one of the original categorical columns
        if col in categorical_cols:
            unique_vals = X_train[col].nunique()
            feature_names.extend([f"{col}_{i}" for i in range(unique_vals)])
        # If it's one of our created categorical columns
        else:
            if col == 'age_group':
                feature_names.extend([f"{col}_{age}" for age in ['<30', '30-40', '40-50', '50-60', '>60']])
            elif col == 'duration_bin':
                feature_names.extend([f"{col}_{dur}" for dur in ['very_short', 'short', 'medium', 'long', 'very_long']])

    # Get feature importances
    importances = optimized_model.feature_importances_

    # Match feature names with importances (take only what we need)
    n_features = len(importances)
    if len(feature_names) > n_features:
        feature_names = feature_names[:n_features]
    elif len(feature_names) < n_features:
        feature_names.extend([f"Feature_{i}" for i in range(len(feature_names), n_features)])

    # Create DataFrame of importances
    importance_df = pd.DataFrame({
        'Feature': feature_names[:n_features],
        'Importance': importances
    }).sort_values('Importance', ascending=False).head(20)

    # Plot feature importance
    plt.figure(figsize=(12, 10))
    sns.barplot(x='Importance', y='Feature', data=importance_df)
    plt.title(f'Top 20 Feature Importances ({best_model_name})')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.close()

# 8. Final Model Implementation and Prediction Function
def predict_subscription(new_data, model=optimized_model, preprocessor=preprocessor):
    """
    Predict whether a client will subscribe to a term deposit.

    Parameters:
    -----------
    new_data : pandas DataFrame
        Data for new clients with the same format as the training data
    model : trained model
        The final trained model
    preprocessor : ColumnTransformer
        The preprocessor used to transform the data

    Returns:
    --------
    predictions : numpy array
        Binary predictions (0/1)
    probabilities : numpy array
        Probability of subscription
    """
    # Create a copy to avoid modifying the original data
    data_copy = new_data.copy()

    # Apply the same feature engineering steps as in training

    # Age groups
    if 'age' in data_copy.columns:
        data_copy['age_group'] = pd.cut(data_copy['age'], bins=[0, 30, 40, 50, 60, 100],
                                       labels=['<30', '30-40', '40-50', '50-60', '>60'])

    # Duration bins
    if 'duration' in data_copy.columns:
        data_copy['duration_bin'] = pd.cut(data_copy['duration'], bins=[0, 100, 300, 600, 1000, 5000],
                                         labels=['very_short', 'short', 'medium', 'long', 'very_long'])

    # Has previous contact
    if 'previous' in data_copy.columns:
        data_copy['has_previous_contact'] = (data_copy['previous'] > 0).astype(int)

    # Contact rate
    if 'campaign' in data_copy.columns and 'previous' in data_copy.columns:
        data_copy['contact_rate'] = data_copy['campaign'] / data_copy['previous'].replace(0, 1)

    # Preprocess the new data
    X_new_processed = preprocessor.transform(data_copy)

    # Make predictions
    predictions = model.predict(X_new_processed)
    probabilities = model.predict_proba(X_new_processed)[:, 1]

    return predictions, probabilities

# Sample usage with a few test cases
print("\nSample prediction with a few test samples:")
sample_data = X_test.sample(5, random_state=42)
sample_predictions, sample_probabilities = predict_subscription(sample_data)

# Create a DataFrame to display results
sample_results = pd.DataFrame({
    'Actual': y_test.loc[sample_data.index],
    'Predicted': sample_predictions,
    'Probability': sample_probabilities.round(3)
})

# Add a few key features for context
key_features = ['age', 'job', 'education', 'duration']
for col in key_features:
    if col in sample_data.columns:
        sample_results[col] = sample_data[col].values

print(sample_results)

# Print final model summary
print(f"\nFinal Model: {best_model_name}")
print(f"Optimal Parameters: {best_params}")
print(f"ROC AUC on Test Set: {optimized_result['roc_auc']:.4f}")
print("\nModel Training and Evaluation Complete!")
