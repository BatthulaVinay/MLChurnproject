# %%
# Basic Import
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns

# %%
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.metrics import accuracy_score,recall_score,precision_score
from sklearn.metrics import f1_score,roc_auc_score, roc_curve, auc
from sklearn.feature_selection import mutual_info_classif
from imblearn.over_sampling import SMOTE
import warnings

warnings.filterwarnings("ignore")

# %% [markdown]
# **Import the CSV Data as Pandas DataFrame**

# %%
df = pd.read_csv('data/Telecom_churn.csv')

# %% [markdown]
# **Show Top 5 Records**

# %%
df.head()

# %% [markdown]
# ## Final Feature Decisions
# 
# Target:
# - Churn (binary, imbalanced)
# 
# Dropped Features:
# - State (high cardinality, overfitting risk)
# - All charge columns (linear transforms of minutes → redundancy)
# 
# Numerical Features:
# - Account length
# - Total day minutes
# - Total eve minutes
# - Total night minutes
# - Total intl minutes
# - Customer service calls
# - Number vmail messages
# 
# Binary Features:
# - International plan
# - Voice mail plan
# 
# Categorical Features:
# - Area code
# 
# Preprocessing:
# - Scale numerical features (StandardScaler)
# - One-hot encode categorical features
# - Binary features passed as-is
# 
# Model:
# - Logistic Regression
# - class_weight = "balanced"
# 
# Metrics:
# - ROC-AUC
# - F1-score
# 

# %% [markdown]
# # **Data Preprocessing**

# %% [markdown]
# Defined `Churn` as the target variable and removed high-cardinality and redundant charge features to prevent multicollinearity and improve model stability.

# %%
target_col = "Churn"

drop_cols = [
    "State",
    "Total day charge",
    "Total eve charge",
    "Total night charge",
    "Total intl charge"
]

df_model = df.drop(columns=drop_cols)

X = df_model.drop(columns=target_col)
y = df_model[target_col]


# %% [markdown]
# Separated features into numeric, binary, and categorical groups to enable appropriate preprocessing and encoding during the modeling pipeline.

# %%
numeric_features = [
    "Account length",
    "Total day minutes",
    "Total eve minutes",
    "Total night minutes",
    "Total intl minutes",
    "Customer service calls",
    "Number vmail messages"
]

binary_features = [
    "International plan",
    "Voice mail plan"
]

categorical_features = [
    "Area code"
]

# %% [markdown]
# Split the data into train and test sets using stratified sampling to preserve the churn class distribution and ensure reliable evaluation.

# %%
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# %% [markdown]
# Created a preprocessing pipeline that scales numeric features, one-hot encodes categorical features, and passes binary features through unchanged for consistent model input.

# %%
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

numeric_transformer = Pipeline(steps=[
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
        ("bin", "passthrough", binary_features)
    ]
)

# %% [markdown]
# Built a logistic regression pipeline with class-weight balancing, binary feature encoding, and preprocessing to handle numeric scaling and categorical one-hot encoding, then trained it on the training set.

# %%
from sklearn.linear_model import LogisticRegression

# Encode binary features
X_train_encoded = X_train.copy()
X_train_encoded['International plan'] = (X_train_encoded['International plan'] == 'Yes').astype(int)
X_train_encoded['Voice mail plan'] = (X_train_encoded['Voice mail plan'] == 'Yes').astype(int)

X_test_encoded = X_test.copy()
X_test_encoded['International plan'] = (X_test_encoded['International plan'] == 'Yes').astype(int)
X_test_encoded['Voice mail plan'] = (X_test_encoded['Voice mail plan'] == 'Yes').astype(int)

model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    random_state=42
)

clf = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", model)
])

clf.fit(X_train_encoded, y_train)

# %% [markdown]
# Evaluated the trained model on the test set using ROC-AUC to measure probability ranking performance and F1-score to balance precision and recall for the imbalanced churn dataset.

# %%
from sklearn.metrics import (
    classification_report,
    recall_score,
    precision_score,
    roc_auc_score,
    f1_score
)

# probabilities
y_proba = clf.predict_proba(X_test_encoded)[:, 1]

# labels
y_pred = (y_proba >= 0.3).astype(int)

print(classification_report(y_test, y_pred))

recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_proba)
f1 = f1_score(y_test, y_pred)

print("Recall:", recall)
print("Precision:", precision)
print("ROC-AUC:", roc_auc)
print("F1-score:", f1)


# %% [markdown]
# ## Logistic Regression with SMOTE
# 
# Training SMOTE-enhanced logistic regression model to improve minority class recall through synthetic sample generation during training.
# 

# %%
from imblearn.pipeline import Pipeline as ImbPipeline

# Define SMOTE
smote = SMOTE(random_state=42)

# Logistic Regression model
model_smote = LogisticRegression(max_iter=1000, random_state=42)

# Build the pipeline: preprocessing + SMOTE + model
clf_smote = ImbPipeline(steps=[
    ("preprocessor", preprocessor),
    ("smote", smote),
    ("model", model_smote)
])

# Fit on training data
clf_smote.fit(X_train_encoded, y_train)

# Evaluate
y_pred_smote = clf_smote.predict(X_test_encoded)
y_proba_smote = clf_smote.predict_proba(X_test_encoded)[:, 1]


y_pred_smote = (y_proba_smote >= 0.3).astype(int)

print(classification_report(y_test, y_pred_smote))

recall_smote = recall_score(y_test, y_pred_smote)
precision_smote = precision_score(y_test, y_pred_smote)
roc_auc_smote = roc_auc_score(y_test, y_proba_smote)
f1_smote = f1_score(y_test, y_pred_smote)

print("SMOTE Model Performance:")
print(f"Recall: {recall_smote}")
print(f"Precision: {precision_smote}")
print(f"ROC-AUC: {roc_auc_smote}")
print(f"F1-score: {f1_smote}")


# %% [markdown]
# ## Model Comparison: Standard vs SMOTE
# 

# %%
import pandas as pd

comparison = pd.DataFrame({
    'Model': ['Standard LR (class_weight)', 'LR + SMOTE'],
    'Recall': [recall, recall_smote],
    'Precision': [precision, precision_smote],
    'ROC-AUC': [roc_auc, roc_auc_smote],
    'F1-Score': [f1, f1_smote],
})

print("\n" + "="*60)
print("MODEL PERFORMANCE COMPARISON")
print("="*60)
print(comparison.to_string(index=False))
print("="*60)

# Show improvements
print(f"\nRecall Improvement: {((recall_smote - recall) / recall * 100):.2f}%")
print(f"Precision Improvement: {((precision_smote - precision) / precision * 100):.2f}%")
print(f"\nF1-Score Improvement: {((f1_smote - f1) / f1 * 100):.2f}%")
print(f"ROC-AUC Change: {((roc_auc_smote - roc_auc) / roc_auc * 100):.2f}%")


# %% [markdown]
# # Additional Models: Random Forest, Decision Tree, XGBoost
# 
# Training ensemble and gradient boosting models to benchmark against Logistic Regression.
# 

# %%
## Random Forest Classifier

# Build Random Forest pipeline
rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, class_weight="balanced")

clf_rf = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", rf_model)
])

clf_rf.fit(X_train_encoded, y_train)

y_pred_rf = clf_rf.predict(X_test_encoded)
y_proba_rf = clf_rf.predict_proba(X_test_encoded)[:, 1]

classification_report(y_test, y_pred_rf)

y_pred_rf = (y_proba_rf >= 0.3).astype(int)

recall_rf = recall_score(y_test, y_pred_rf)
precision_rf = precision_score(y_test, y_pred_rf)
roc_auc_rf = roc_auc_score(y_test, y_proba_rf)
f1_rf = f1_score(y_test, y_pred_rf)

print("Random Forest Performance:")
print(f"Recall: {recall_rf}")
print(f"Precision: {precision_rf}")
print(f"ROC-AUC: {roc_auc_rf}")
print(f"F1-score: {f1_rf}")


# %%
## Decision Tree Classifier

dt_model = DecisionTreeClassifier(max_depth=10, random_state=42, class_weight="balanced")

clf_dt = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", dt_model)
])

clf_dt.fit(X_train_encoded, y_train)

y_pred_dt = clf_dt.predict(X_test_encoded)
y_proba_dt = clf_dt.predict_proba(X_test_encoded)[:, 1]

y_pred_dt = (y_proba_dt >= 0.3).astype(int)
classification_report(y_test, y_pred_dt)

recall_dt = recall_score(y_test, y_pred_dt)
precision_dt = precision_score(y_test, y_pred_dt)
roc_auc_dt = roc_auc_score(y_test, y_proba_dt)
f1_dt = f1_score(y_test, y_pred_dt)

print("Decision Tree Performance:")
print(f"Recall: {recall_dt}")
print(f"Precision: {precision_dt}")
print(f"ROC-AUC: {roc_auc_dt}")
print(f"F1-score: {f1_dt}")


# %%
## XGBoost Classifier

xgb_model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=7,
    learning_rate=0.1,
    random_state=42,
    scale_pos_weight=len(y_train[y_train == False]) / len(y_train[y_train == True]),
    eval_metric='logloss'
)

clf_xgb = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", xgb_model)
])

clf_xgb.fit(X_train_encoded, y_train)

y_pred_xgb = clf_xgb.predict(X_test_encoded)
y_proba_xgb = clf_xgb.predict_proba(X_test_encoded)[:, 1]

y_pred_xgb = (y_proba_xgb >= 0.3).astype(int)
classification_report(y_test, y_pred_xgb)

recall_xgb = recall_score(y_test, y_pred_xgb)
precision_xgb = precision_score(y_test, y_pred_xgb)
roc_auc_xgb = roc_auc_score(y_test, y_proba_xgb)
f1_xgb = f1_score(y_test, y_pred_xgb)

print("XGBoost Performance:")
print(f"Recall: {recall_xgb}")
print(f"Precision: {precision_xgb}")
print(f"ROC-AUC: {roc_auc_xgb}")
print(f"F1-score: {f1_xgb}")


# %% [markdown]
# ## All Models Comparison
# 

# %%
all_models_comparison = pd.DataFrame({
    'Model': [
        'Standard LR',
        'LR + SMOTE',
        'Random Forest',
        'Decision Tree',
        'XGBoost'
    ],
    'precision': [precision, precision_smote, precision_rf, precision_dt, precision_xgb],
    'Recall': [recall, recall_smote, recall_rf, recall_dt, recall_xgb],
    'ROC-AUC': [roc_auc, roc_auc_smote, roc_auc_rf, roc_auc_dt, roc_auc_xgb],
    'F1-Score': [f1, f1_smote, f1_rf, f1_dt, f1_xgb],
})

# Sort by Recall descending
all_models_comparison = all_models_comparison.sort_values('Recall', ascending=False).reset_index(drop=True)

print("\n" + "="*70)
print("ALL MODELS PERFORMANCE COMPARISON")
print("="*70)
print(all_models_comparison.to_string(index=False))
print("="*70)

# Highlight best model
best_recall = all_models_comparison.loc[0, 'Model']
best_f1 = all_models_comparison.loc[all_models_comparison['F1-Score'].idxmax(), 'Model']

print(f"\n✓ Best Recall: {best_recall}")
print(f"✓ Best F1-Score: {best_f1}")


# %%
## Best Model Detailed Analysis

# Get best model by F1-Score
best_model_name = all_models_comparison.loc[all_models_comparison['F1-Score'].idxmax(), 'Model']

if best_model_name == 'Standard LR':
    best_model = clf
    y_pred_best = y_pred
elif best_model_name == 'LR + SMOTE':
    best_model = clf_smote
    y_pred_best = y_pred_smote
elif best_model_name == 'Random Forest':
    best_model = clf_rf
    y_pred_best = y_pred_rf
elif best_model_name == 'Decision Tree':
    best_model = clf_dt
    y_pred_best = y_pred_dt
else:  # XGBoost
    best_model = clf_xgb
    y_pred_best = y_pred_xgb

print("\n" + "="*70)
print(f"BEST MODEL ANALYSIS: {best_model_name}")
print("="*70)

cm_best = confusion_matrix(y_test, y_pred_best)
print("\nConfusion Matrix:")
print(cm_best)
print("\nClassification Report:")
print(classification_report(y_test, y_pred_best))

# Show business metrics
tn, fp, fn, tp = cm_best.ravel()
print(f"\nBusiness Metrics:")
print(f"  True Negatives (Correct Non-Churn): {tn}")
print(f"  False Positives (Incorrect Churn): {fp}")
print(f"  False Negatives (Missed Churners): {fn}")
print(f"  True Positives (Correct Churn): {tp}")
print(f"\n  Retention Rate: {(tp / (tp + fn) * 100):.2f}% (Catching churners)")



