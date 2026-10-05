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
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, roc_auc_score, brier_score_loss
from sklearn.calibration import calibration_curve

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

# Target encoding (must be global for sklearn target)
target_col = df.columns[-1]
print("\nTarget Column:", target_col)

le = LabelEncoder()
df[target_col] = le.fit_transform(df[target_col])

X = df.drop(target_col, axis=1)
y = df[target_col]

# Identify categorical features for pipeline
cat_cols = X.select_dtypes(include=["object"]).columns.tolist()

# ============================================
# DEFINE MODELS
# ============================================

SEED = 42
np.random.seed(SEED)

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder

preprocessor = ColumnTransformer(
    transformers=[("cat", OrdinalEncoder(), cat_cols)],
    remainder="passthrough"
)

models = {
    "Logistic Regression": Pipeline([
        ("preprocessor", preprocessor),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(random_state=SEED))
    ]),
    "Support Vector Machine": Pipeline([
        ("preprocessor", preprocessor),
        ("scaler", StandardScaler()),
        ("model", SVC(probability=True, random_state=SEED))
    ]),
    "Random Forest": Pipeline([
        ("preprocessor", preprocessor),
        ("model", RandomForestClassifier(n_estimators=100, random_state=SEED))
    ]),
    "Gradient Boosting": Pipeline([
        ("preprocessor", preprocessor),
        ("model", GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=SEED))
    ])
}

# ============================================
# INDEPENDENT TRAIN / TEST SPLIT
# ============================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=SEED,
    stratify=y
)

# ============================================
# 5-FOLD CROSS-VALIDATION ON TRAINING SET ONLY
# ============================================

from sklearn.metrics import make_scorer, matthews_corrcoef, average_precision_score, accuracy_score, precision_score, recall_score, f1_score, balanced_accuracy_score

def calc_specificity(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return tn / (tn + fp) if (tn + fp) > 0 else 0.0

def calc_npv(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return tn / (tn + fn) if (tn + fn) > 0 else 0.0

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

scoring = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "specificity": make_scorer(calc_specificity),
    "f1": "f1",
    "balanced_accuracy": "balanced_accuracy",
    "npv": make_scorer(calc_npv),
    "mcc": make_scorer(matthews_corrcoef),
    "pr_auc": make_scorer(average_precision_score, response_method="predict_proba")
}

print("\n================ MODEL RESULTS (5-FOLD CV) ================\n")

csv_results = []

metric_display_names = {
    "accuracy": "Accuracy",
    "precision": "Precision",
    "recall": "Recall",
    "specificity": "Specificity",
    "f1": "F1",
    "balanced_accuracy": "Balanced Accuracy",
    "npv": "NPV",
    "mcc": "MCC",
    "pr_auc": "PR-AUC"
}

for name, model in models.items():
    scores = cross_validate(
        model,
        X_train,
        y_train,
        cv=cv,
        scoring=scoring
    )
    
    row_data = {"Model Name": name}
    print(name)
    
    for metric_key, display_name in metric_display_names.items():
        mean_val = np.mean(scores[f"test_{metric_key}"])
        std_val = np.std(scores[f"test_{metric_key}"])
        
        row_data[f"{display_name} Mean"] = mean_val
        row_data[f"{display_name} SD"] = std_val
        
        print(f"  {display_name}: {mean_val:.4f} ± {std_val:.4f}")
    
    csv_results.append(row_data)
    print("")

results_df = pd.DataFrame(csv_results)
cols = ["Model Name"]
for display_name in metric_display_names.values():
    cols.extend([f"{display_name} Mean", f"{display_name} SD"])
results_df = results_df[cols]
results_df = results_df.sort_values(by="Accuracy Mean", ascending=False)
results_df.to_csv("lung_cancer_model_results.csv", index=False)
print("Model results saved successfully to lung_cancer_model_results.csv")

# Model selection based ONLY on training-set CV accuracy
best_model_name = results_df.loc[
    results_df["Accuracy Mean"].idxmax(),
    "Model Name"
]

print("\nSelected model:", best_model_name)

# ============================================
# FIT SELECTED MODEL ON COMPLETE TRAINING SET
# ============================================

selected_model = models[best_model_name]
selected_model.fit(X_train, y_train)

# ============================================
# FINAL INDEPENDENT TEST EVALUATION
# ============================================

y_pred = selected_model.predict(X_test)
y_prob = selected_model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
specificity = calc_specificity(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
npv = calc_npv(y_test, y_pred)
mcc = matthews_corrcoef(y_test, y_pred)
pr_auc = average_precision_score(y_test, y_prob)
roc_auc = roc_auc_score(y_test, y_prob)

print("\n================ FINAL TEST EVALUATION ================")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"Specificity: {specificity:.4f}")
print(f"F1: {f1:.4f}")
print(f"Balanced Accuracy: {balanced_accuracy:.4f}")
print(f"NPV: {npv:.4f}")
print(f"MCC: {mcc:.4f}")
print(f"PR-AUC: {pr_auc:.4f}")
print(f"ROC-AUC: {roc_auc:.4f}")

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

# ============================================
# CALIBRATION ANALYSIS
# ============================================
brier_score = brier_score_loss(y_test, y_prob)
print(f"\nBrier Score: {brier_score:.4f}")

prob_true, prob_pred = calibration_curve(y_test, y_prob, n_bins=10, strategy="uniform")

plt.figure()
plt.plot(prob_pred, prob_true, marker='o', label="Logistic Regression")
plt.plot([0, 1], [0, 1], linestyle="--", color="black", label="Perfectly Calibrated")
plt.xlabel("Mean Predicted Probability")
plt.ylabel("Observed Proportion")
plt.title("Calibration Curve - Lung Cancer Logistic Regression")
plt.legend()
plt.savefig("lung_cancer_calibration_curve.png", dpi=300)
print("Calibration curve generated: lung_cancer_calibration_curve.png")

# ============================================
# SUBGROUP ANALYSIS
# ============================================
print("\n================ SUBGROUP ANALYSIS ================")

def calculate_subgroup_metrics_lc(y_true, y_pred, y_prob):
    n = len(y_true)
    if n == 0 or len(np.unique(y_true)) < 2:
        return {"N": n, "Accuracy": "Unavailable", "Precision": "Unavailable", "Recall": "Unavailable", "Specificity": "Unavailable", "F1": "Unavailable", "Balanced Accuracy": "Unavailable"}
    
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1 = f1_score(y_true, y_pred, zero_division=0)
    bacc = balanced_accuracy_score(y_true, y_pred)
    
    return {"N": n, "Accuracy": f"{acc:.4f}", "Precision": f"{prec:.4f}", "Recall": f"{rec:.4f}", "Specificity": f"{spec:.4f}", "F1": f"{f1:.4f}", "Balanced Accuracy": f"{bacc:.4f}"}

median_age_lc = df["AGE"].median()
print(f"Age median split threshold: {median_age_lc}")

age_group1_mask_lc = X_test["AGE"] <= median_age_lc
age_group2_mask_lc = X_test["AGE"] > median_age_lc

sg_results_lc = []
res1_lc = calculate_subgroup_metrics_lc(y_test[age_group1_mask_lc], y_pred[age_group1_mask_lc], y_prob[age_group1_mask_lc])
sg_results_lc.append({"Dataset": "Lung Cancer", "Subgroup Type": "Age", "Subgroup": f"<= {median_age_lc}", **res1_lc})

res2_lc = calculate_subgroup_metrics_lc(y_test[age_group2_mask_lc], y_pred[age_group2_mask_lc], y_prob[age_group2_mask_lc])
sg_results_lc.append({"Dataset": "Lung Cancer", "Subgroup Type": "Age", "Subgroup": f"> {median_age_lc}", **res2_lc})

# Gender
gender_m_mask = X_test["GENDER"] == "M"
gender_f_mask = X_test["GENDER"] == "F"

res_m = calculate_subgroup_metrics_lc(y_test[gender_m_mask], y_pred[gender_m_mask], y_prob[gender_m_mask])
sg_results_lc.append({"Dataset": "Lung Cancer", "Subgroup Type": "Gender", "Subgroup": "M", **res_m})

res_f = calculate_subgroup_metrics_lc(y_test[gender_f_mask], y_pred[gender_f_mask], y_prob[gender_f_mask])
sg_results_lc.append({"Dataset": "Lung Cancer", "Subgroup Type": "Gender", "Subgroup": "F", **res_f})

sg_df_lc = pd.DataFrame(sg_results_lc)
sg_df_lc.to_csv("lung_cancer_subgroup_results.csv", index=False)
print("Subgroup analysis saved to lung_cancer_subgroup_results.csv")
print(sg_df_lc.to_string(index=False))
