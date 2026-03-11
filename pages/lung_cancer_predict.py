# lung_cancer_predict.py

import streamlit as st
import pandas as pd
from pages.lung_cancer_model import load_and_preprocess, train_model

# Load and train model
file_path = "D:/LangchainProjects/Q_AChatbot/PBL_Project/pages/lung_cancer.csv"
X, y, label_encoders = load_and_preprocess(file_path)
model, accuracy = train_model(X, y)

# User Input
def get_user_input():
    gender = st.selectbox("Gender", ["M", "F"])
    age = st.slider("Age", 30, 100, 60)
    smoking = st.selectbox("Smoking (1=Low, 2=High)", [1, 2])
    yellow_fingers = st.selectbox("Yellow Fingers", [1, 2])
    anxiety = st.selectbox("Anxiety", [1, 2])
    peer_pressure = st.selectbox("Peer Pressure", [1, 2])
    chronic_disease = st.selectbox("Chronic Disease", [1, 2])
    fatigue = st.selectbox("Fatigue", [1, 2])
    allergy = st.selectbox("Allergy", [1, 2])
    wheezing = st.selectbox("Wheezing", [1, 2])
    alcohol = st.selectbox("Alcohol Consuming", [1, 2])
    coughing = st.selectbox("Coughing", [1, 2])
    sob = st.selectbox("Shortness of Breath", [1, 2])
    swallowing = st.selectbox("Swallowing Difficulty", [1, 2])
    chest_pain = st.selectbox("Chest Pain", [1, 2])

    input_data = {
        "GENDER": label_encoders["GENDER"].transform([gender])[0],
        "AGE": age,
        "SMOKING": smoking,
        "YELLOW_FINGERS": yellow_fingers,
        "ANXIETY": anxiety,
        "PEER_PRESSURE": peer_pressure,
        "CHRONIC DISEASE": chronic_disease,
        "FATIGUE": fatigue,
        "ALLERGY": allergy,
        "WHEEZING": wheezing,
        "ALCOHOL CONSUMING": alcohol,
        "COUGHING": coughing,
        "SHORTNESS OF BREATH": sob,
        "SWALLOWING DIFFICULTY": swallowing,
        "CHEST PAIN": chest_pain
    }

    return pd.DataFrame([input_data])

# Streamlit Interface
st.title("🚭 Lung Cancer Prediction")
st.write("Provide health indicators to predict lung cancer risk:")

user_input = get_user_input()

if st.button("Predict"):
    prediction = model.predict(user_input)[0]
    result = label_encoders['LUNG_CANCER'].inverse_transform([prediction])[0]
    st.success(f"Prediction: **{result}**")

st.write(f"🔍 Model Accuracy: **{accuracy:.2%}**")
