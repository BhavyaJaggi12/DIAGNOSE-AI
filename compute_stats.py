import pandas as pd
import numpy as np

# 1. Diabetes Dataset
df_diabetes = pd.read_csv("data/raw/diabetes.csv")
stats_diabetes = df_diabetes.describe().T[['mean', 'std', 'min', 'max']]
stats_diabetes.columns = ['Mean', 'Std. Dev.', 'Min', 'Max']
stats_diabetes.index.name = 'Feature'

print("Diabetes Stats:")
print(stats_diabetes.round(2))
print()

# 2. Lung Cancer Dataset
df_lung = pd.read_csv("data/raw/survey lung cancer.csv")

# Encode Gender: M=1, F=0 (Let's check the mean if M=1)
# if M=1, mean is ~0.51? Let's verify:
df_lung['Gender (Encoded)'] = df_lung['GENDER'].map({'M': 1, 'F': 0})

# Encode target: YES=1, NO=0
df_lung['Lung Cancer (Target)'] = df_lung['LUNG_CANCER'].map({'YES': 1, 'NO': 0})

# Other columns are 1 and 2. Convert to 0 and 1 by subtracting 1
cols_to_shift = ['SMOKING', 'YELLOW_FINGERS', 'ANXIETY', 'PEER_PRESSURE', 'CHRONIC DISEASE', 
                 'FATIGUE ', 'ALLERGY ', 'WHEEZING', 'ALCOHOL CONSUMING', 'COUGHING', 
                 'SHORTNESS OF BREATH', 'SWALLOWING DIFFICULTY', 'CHEST PAIN']

for col in cols_to_shift:
    df_lung[col] = df_lung[col] - 1

# Rename columns to match Table II
rename_map = {
    'AGE': 'Age (years)',
    'SMOKING': 'Smoking',
    'YELLOW_FINGERS': 'Yellow Fingers',
    'ANXIETY': 'Anxiety',
    'PEER_PRESSURE': 'Peer Pressure',
    'CHRONIC DISEASE': 'Chronic Disease',
    'FATIGUE ': 'Fatigue',
    'ALLERGY ': 'Allergy',
    'WHEEZING': 'Wheezing',
    'ALCOHOL CONSUMING': 'Alcohol Consumption',
    'COUGHING': 'Coughing',
    'SHORTNESS OF BREATH': 'Shortness of Breath',
    'SWALLOWING DIFFICULTY': 'Swallowing Difficulty',
    'CHEST PAIN': 'Chest Pain'
}
df_lung = df_lung.rename(columns=rename_map)

features_lung = ['Age (years)', 'Gender (Encoded)', 'Smoking', 'Yellow Fingers', 'Anxiety', 
                 'Peer Pressure', 'Chronic Disease', 'Fatigue', 'Allergy', 'Wheezing', 
                 'Alcohol Consumption', 'Coughing', 'Shortness of Breath', 'Swallowing Difficulty', 
                 'Chest Pain', 'Lung Cancer (Target)']

stats_lung = df_lung[features_lung].describe().T[['mean', 'std', 'min', 'max']]
stats_lung.columns = ['Mean', 'Std. Dev.', 'Min', 'Max']
stats_lung.index.name = 'Feature'

print("Lung Cancer Stats:")
print(stats_lung.round(2))

# Save both to CSV
stats_diabetes['Dataset'] = 'PIMA Indians Diabetes'
stats_lung['Dataset'] = 'Survey Lung Cancer'

combined_stats = pd.concat([stats_diabetes, stats_lung])
combined_stats.to_csv("descriptive_statistics.csv")
print("\nSaved to descriptive_statistics.csv")
