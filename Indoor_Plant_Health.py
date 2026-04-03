# Indoor_Plant_Health_and_Growth_Factors.py
# ------------------------------------------

# Install dependencies before running:
# pip install catboost scikit-learn pandas matplotlib seaborn

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder

from catboost import CatBoostClassifier, Pool
import matplotlib.pyplot as plt
import seaborn as sns

# Global Settings
RANDOM_STATE = 42
N_FOLDS = 5
USE_GPU = False  # Set to True only if you have GPU-compatible CatBoost

np.random.seed(RANDOM_STATE)
plt.rcParams["figure.figsize"] = (10, 6)
sns.set_style("whitegrid")

# =========================================
# Load dataset (EDIT THIS PATH)
# =========================================
DATA_PATH = "your_file.csv"   # <--- Replace with actual CSV filename

df = pd.read_csv(DATA_PATH)

print("Dataset Shape:", df.shape)
print(df.head())
print(df.info())

TARGET_COL = "health_status"   # <-- Change if your target column is different

if TARGET_COL not in df.columns:
    raise ValueError(f"Target column '{TARGET_COL}' not found. Please update TARGET_COL.")

# Remove missing values
df = df.dropna().reset_index(drop=True)

X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL]

# Encode labels if target is categorical
if y.dtype == "O" or str(y.dtype).startswith("category"):
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    print("Encoded Classes:", le.classes_)
else:
    y_encoded = y.values

class_labels = np.unique(y_encoded)
n_classes = len(class_labels)

# Detect categorical features by dtype
cat_features = [
    i for i, col in enumerate(X.columns)
    if str(X[col].dtype) == "object" or str(X[col].dtype).startswith("category")
]

print("Categorical feature indices:", cat_features)

# CatBoost parameters
params = {
    "loss_function": "MultiClass",
    "eval_metric": "TotalF1",
    "learning_rate": 0.05,
    "iterations": 1000,
    "depth": 6,
    "random_seed": RANDOM_STATE,
    "verbose": 100,
    "task_type": "GPU" if USE_GPU else "CPU"
}

# Stratified K-Fold training
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

oof_preds = np.zeros((len(X), n_classes))
oof_labels = np.zeros(len(X), dtype=int)
fold_scores = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y_encoded), 1):
    print(f"\n===== Fold {fold}/{N_FOLDS} =====")

    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y_encoded[train_idx], y_encoded[val_idx]

    train_pool = Pool(data=X_train, label=y_train, cat_features=cat_features)
    val_pool = Pool(data=X_val, label=y_val, cat_features=cat_features)

    model = CatBoostClassifier(**params)
    model.fit(train_pool, eval_set=val_pool, use_best_model=True)

    val_probs = model.predict_proba(X_val)
    val_pred_idx = np.argmax(val_probs, axis=1)

    oof_preds[val_idx] = val_probs
    oof_labels[val_idx] = val_pred_idx

    acc = accuracy_score(y_val, val_pred_idx)
    f1 = f1_score(y_val, val_pred_idx, average="macro")
    fold_scores.append((acc, f1))

    print(f"Fold {fold} Accuracy: {acc:.4f} | Macro-F1: {f1:.4f}")

# Overall Metrics
overall_acc = accuracy_score(y_encoded, oof_labels)
overall_f1 = f1_score(y_encoded, oof_labels, average="macro")

print("\n===== Overall Performance =====")
print(f"Accuracy: {overall_acc:.4f}")
print(f"Macro-F1: {overall_f1:.4f}")
print("\nClassification Report:")
print(classification_report(y_encoded, oof_labels))

# Confusion Matrix
cm = confusion_matrix(y_encoded, oof_labels, labels=class_labels)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d",
            xticklabels=class_labels,
            yticklabels=class_labels)
plt.title("Confusion Matrix (OOF)")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.show()

# Feature Importance (from final fold)
importances = model.get_feature_importance(type="FeatureImportance")
feature_importance_df = pd.DataFrame({
    "feature": X.columns,
    "importance": importances
}).sort_values(by="importance", ascending=False)

print("\nTop Features:")
print(feature_importance_df.head())

plt.figure(figsize=(10, 6))
sns.barplot(data=feature_importance_df.head(20), x="importance", y="feature")
plt.title("Top 20 Feature Importances")
plt.tight_layout()
plt.show()
