# ============================================
# LUNG CANCER PREDICTION MODULE
# Dataset: jillanisofttech/lung-cancer-detection
# ============================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, roc_auc_score

# ============================================
# LOAD DATA
# ============================================

DATA_PATH = "data/raw/survey lung cancer.csv"

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError("Dataset not found. Please place lung_cancer.csv inside data/raw/")

df = pd.read_csv(DATA_PATH)

print("\n================ DATASET OVERVIEW ================\n")
print("Shape:", df.shape)
print("\nFirst 5 Rows:\n")
print(df.head())

# ============================================
# DATASET DESCRIPTION
# ============================================

print("\n================ DATASET DESCRIPTION ================\n")
print(df.describe(include="all"))

print("\nClass Distribution:\n")
print(df.iloc[:, -1].value_counts())

# ============================================
# PREPROCESSING
# ============================================

# Encode categorical features
le = LabelEncoder()

for col in df.columns:
    if df[col].dtype == "object":
        df[col] = le.fit_transform(df[col])

# Assume target is last column
target_col = df.columns[-1]
print("\nTarget Column:", target_col)

X = df.drop(target_col, axis=1)
y = df[target_col]

# ============================================
# DEFINE MODELS
# ============================================

SEED = 42
np.random.seed(SEED)

models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(random_state=SEED))
    ]),
    "Support Vector Machine": Pipeline([
        ("scaler", StandardScaler()),
        ("model", SVC(probability=True, random_state=SEED))
    ]),
    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=SEED
    ),
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        random_state=SEED
    )
}

# ============================================
# 5-FOLD CROSS VALIDATION
# ============================================

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
scoring = ["accuracy", "precision", "recall", "f1"]

print("\n================ MODEL RESULTS (5-FOLD CV) ================\n")

results = {}

csv_results = []

for name, model in models.items():
    scores = cross_validate(model, X, y, cv=cv, scoring=scoring)

    results[name] = {metric: np.mean(scores[f"test_{metric}"]) for metric in scoring}

    row_data = {"Model Name": name}
    for metric in scoring:
        row_data[metric.capitalize()] = results[name][metric]
    csv_results.append(row_data)

    print(name)
    for metric in scoring:
        print(f"  {metric}: {results[name][metric]:.4f}")
    print("")

results_df = pd.DataFrame(csv_results)
results_df = results_df.sort_values(by="Accuracy", ascending=False)
results_df.to_csv("lung_cancer_model_results.csv", index=False)
print("Model results saved successfully to lung_cancer_model_results.csv")

# ============================================
# TRAIN BEST MODEL (Random Forest Example)
# ============================================

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)

best_model = RandomForestClassifier(n_estimators=100, random_state=SEED)
best_model.fit(X_train, y_train)

y_pred = best_model.predict(X_test)
y_prob = best_model.predict_proba(X_test)[:, 1]

print("\n================ CONFUSION MATRIX ================\n")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# ============================================
# ROC CURVE
# ============================================

fpr, tpr, _ = roc_curve(y_test, y_prob)
auc_score = roc_auc_score(y_test, y_prob)

plt.figure()
plt.plot(fpr, tpr)
plt.plot([0, 1], [0, 1])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Lung Cancer Dataset")
plt.savefig("lung_cancer_roc_curve.png")

print("\nROC-AUC Score:", round(auc_score, 4))
