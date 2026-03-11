# ============================================
# LUNG CANCER PREDICTION MODULE
# Dataset: jillanisofttech/lung-cancer-detection
# ============================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, roc_auc_score

# ============================================
# 1️⃣ LOAD DATA
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
# 2️⃣ DATASET DESCRIPTION
# ============================================

print("\n================ DATASET DESCRIPTION ================\n")
print(df.describe(include="all"))

print("\nClass Distribution:\n")
print(df.iloc[:, -1].value_counts())

# ============================================
# 3️⃣ PREPROCESSING
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
# 4️⃣ DEFINE MODELS
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
# 5️⃣ 5-FOLD CROSS VALIDATION
# ============================================

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
scoring = ["accuracy", "precision", "recall", "f1"]

print("\n================ MODEL RESULTS (5-FOLD CV) ================\n")

results = {}

for name, model in models.items():
    scores = cross_validate(model, X, y, cv=cv, scoring=scoring)

    results[name] = {metric: np.mean(scores[f"test_{metric}"]) for metric in scoring}

    print(name)
    for metric in scoring:
        print(f"  {metric}: {results[name][metric]:.4f}")
    print("")

# ============================================
# 6️⃣ TRAIN BEST MODEL (Random Forest Example)
# ============================================

best_model = RandomForestClassifier(n_estimators=100, random_state=SEED)
best_model.fit(X, y)

y_pred = best_model.predict(X)
y_prob = best_model.predict_proba(X)[:, 1]

print("\n================ CONFUSION MATRIX ================\n")
print(confusion_matrix(y, y_pred))

print("\nClassification Report:\n")
print(classification_report(y, y_pred))

# ============================================
# 7️⃣ ROC CURVE
# ============================================

fpr, tpr, _ = roc_curve(y, y_prob)
auc_score = roc_auc_score(y, y_prob)

plt.figure()
plt.plot(fpr, tpr)
plt.plot([0, 1], [0, 1])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Lung Cancer Dataset")
plt.show()

print("\nROC-AUC Score:", round(auc_score, 4))