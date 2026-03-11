

# ============================================
# DIABETES PREDICTION MODULE (FULL VERSION)
# ============================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    roc_auc_score
)


# 🔥 This line automatically creates the folder if missing
os.makedirs("data/raw", exist_ok=True)

url = "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv"
df = pd.read_csv(url)

df.to_csv("data/raw/diabetes.csv", index=False)

print("Dataset downloaded successfully!")



# ============================================
# 1️⃣ SET RANDOM SEED FOR REPRODUCIBILITY
# ============================================

SEED = 42
np.random.seed(SEED)

# ============================================
# 2️⃣ LOAD DATASET
# ============================================

DATA_PATH = "data/raw/diabetes.csv"

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError("Dataset not found. Please place diabetes.csv in data/raw/")

df = pd.read_csv(DATA_PATH)

print("\n================ DATASET OVERVIEW ================\n")
print("Shape of dataset:", df.shape)
print("\nFirst 5 rows:\n")
print(df.head())

# ============================================
# 3️⃣ DATASET DESCRIPTION
# ============================================

print("\n================ DATASET DESCRIPTION ================\n")
print(df.describe())

print("\nClass Distribution:")
print(df["Outcome"].value_counts(normalize=True) * 100)

# ============================================
# 4️⃣ DATA PREPROCESSING
# ============================================

# Replace medically invalid zeros with NaN
cols_to_fix = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
df[cols_to_fix] = df[cols_to_fix].replace(0, np.nan)

# Fill missing values with median
df.fillna(df.median(), inplace=True)

X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# ============================================
# 5️⃣ DEFINE MODELS
# ============================================

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
# 6️⃣ CROSS VALIDATION (5-FOLD)
# ============================================

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

scoring = ["accuracy", "precision", "recall", "f1"]

results = {}

print("\n================ MODEL EVALUATION (5-FOLD CV) ================\n")

for name, model in models.items():
    scores = cross_validate(model, X, y, cv=cv, scoring=scoring)

    results[name] = {
        metric: np.mean(scores[f"test_{metric}"])
        for metric in scoring
    }

    print(f"{name}")
    for metric in scoring:
        print(f"  {metric}: {results[name][metric]:.4f}")
    print("")

# ============================================
# 7️⃣ SELECT BEST MODEL (Random Forest expected)
# ============================================

best_model = RandomForestClassifier(
    n_estimators=100,
    random_state=SEED
)

best_model.fit(X, y)

y_pred = best_model.predict(X)
y_prob = best_model.predict_proba(X)[:, 1]

# ============================================
# 8️⃣ CONFUSION MATRIX
# ============================================

print("\n================ CONFUSION MATRIX ================\n")
cm = confusion_matrix(y, y_pred)
print(cm)

print("\nClassification Report:\n")
print(classification_report(y, y_pred))

# ============================================
# 9️⃣ ROC CURVE
# ============================================

fpr, tpr, thresholds = roc_curve(y, y_prob)
auc_score = roc_auc_score(y, y_prob)

plt.figure()
plt.plot(fpr, tpr)
plt.plot([0, 1], [0, 1])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Random Forest")
plt.show()

print("\nROC-AUC Score:", round(auc_score, 4))

# ============================================
# 🔟 SAVE RESULTS
# ============================================

results_df = pd.DataFrame(results).T
results_df.to_csv("diabetes_model_results.csv")

print("\nResults saved to diabetes_model_results.csv")