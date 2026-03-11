# pages/predict_diabetes.py

# diabetes_predict.py

import streamlit as st
import numpy as np
from pages import diabetes_model

st.set_page_config(page_title="Diabetes Predictor", layout="centered")

st.title("🩺 Diagnose AI - Diabetes Prediction")
st.write("Enter the following medical metrics to predict diabetes progression:")

# Input sliders for the 10 features in the diabetes dataset
age = st.slider("Age (standardized)", -0.1, 0.1, 0.0, step=0.001)
sex = st.slider("Sex (standardized)", -0.1, 0.1, 0.0, step=0.001)
bmi = st.slider("BMI", 0.0, 0.2, 0.1, step=0.001)
bp = st.slider("Blood Pressure", 0.0, 0.2, 0.1, step=0.001)
s1 = st.slider("S1: Total Serum Cholesterol", -0.1, 0.1, 0.0, step=0.001)
s2 = st.slider("S2: Low-Density Lipoproteins", -0.1, 0.1, 0.0, step=0.001)
s3 = st.slider("S3: High-Density Lipoproteins", -0.1, 0.1, 0.0, step=0.001)
s4 = st.slider("S4: Total Cholesterol / HDL", 0.0, 0.2, 0.1, step=0.001)
s5 = st.slider("S5: Log of serum triglycerides", 0.0, 0.2, 0.1, step=0.001)
s6 = st.slider("S6: Blood sugar level", -0.1, 0.1, 0.0, step=0.001)

input_data = [age, sex, bmi, bp, s1, s2, s3, s4, s5, s6]

if st.button("🔍 Predict Diabetes Progression"):
    with st.spinner("Training model and making prediction..."):
        # Load and split data
        X, y = diabetes_model.load_data()
        X_train, X_test, y_train, y_test = diabetes_model.split_data(X, y)

        # Train models and get results
        results = diabetes_model.train_models(X_train, X_test, y_train, y_test)

        # Select model with best R2 score
        best_model_name = max(results, key=lambda name: results[name]["R2"])
        best_model = results[best_model_name]["model"]

        # Make prediction
        prediction = diabetes_model.predict_input(best_model, input_data)

        st.success("✅ Prediction complete!")
        st.markdown(f"### 🔮 Predicted Diabetes Progression Score: `{prediction:.2f}`")

        from sklearn.datasets import load_diabetes
        data = load_diabetes()
        y = data.target
        min_score = np.min(y)
        max_score = np.max(y)

        st.markdown("---")
        st.subheader("📘 Understanding the Score")
        st.info(f"""
        - The diabetes progression score ranges from **{min_score:.1f} to {max_score:.1f}**.
        - **Higher scores** indicate **more severe expected disease progression** over the next year.
        - Your predicted score of **{prediction:.2f}** is on this scale.

        🩺 If your score is on the higher end, it's recommended to **consult a healthcare professional** for a thorough diagnosis and further testing.
        """)


        # Show model performance
        # st.markdown("---")
        # st.subheader("📊 Best Model Performance")
        # best_model_name = type(best_model).__name__
        # best_metrics = results[best_model_name]
        # st.write(f"**Model:** {best_model_name}")
        # st.write(f"**R² Score:** {best_metrics['R2']:.4f}")
        # st.write(f"**MAE:** {best_metrics['MAE']:.2f}")
        # st.write(f"**RMSE:** {best_metrics['RMSE']:.2f}")
