# lung_cancer_model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load and preprocess the dataset
def load_and_preprocess(file_path):
    df = pd.read_csv("D:\LangchainProjects\Q_AChatbot\PBL_Project\pages\lung_cancer.csv")

    # Strip column names and make uppercase
    df.columns = df.columns.str.strip().str.upper()

    # Clean and encode target
    target_column = 'LUNG_CANCER'
    if target_column not in df.columns:
        raise ValueError(f"{target_column} column not found in dataset")

    label_encoders = {}
    for col in df.columns:
        if df[col].dtype == object:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le

    X = df.drop(target_column, axis=1)
    y = df[target_column]

    return X, y, label_encoders

# Train model
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    accuracy = accuracy_score(y_test, clf.predict(X_test))
    return clf, accuracy
