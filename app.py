import streamlit as st

# ✅ Configure main entry page (only here)
st.set_page_config(
    page_title="DIAGNOSE AI - Medical Assistant",
    page_icon="🏠",
    layout="wide"
)

# ----------- HEADER -----------
st.title("🏠 DIAGNOSE AI: Your Personal Medical Assistant")
st.write(
    "Your intelligent AI-powered health assistant — analyze, summarize, "
    "chat, and predict medical conditions."
)

st.markdown("---")

# ----------- NAVIGATION LINKS -----------
st.subheader("🔗 Available Services")

col1, col2 = st.columns(2)

with col1:
    st.page_link("pages/chatbot.py", label="🤖 Chat with DIAGNOSE AI")
    st.page_link("pages/summarize.py", label="📄 Summarize Medical Reports")
    st.page_link("pages/report.py", label="📚 Chat with PDF Reports")  # ✅ NEW PAGE

with col2:
    st.page_link("pages/diabetes_predict.py", label="🩺 Diabetes Predictor")
    st.page_link("pages/lung_cancer_predict.py", label="🫁 Lung Cancer Risk Predictor")

st.markdown("---")

# ----------- ABOUT SECTION -----------
with st.expander("ℹ️ About Diagnose AI"):
    st.markdown("""
**Diagnose AI** is a smart multi-feature medical assistant designed to help users:

✅ Predict disease risks (Diabetes, Lung Cancer)  
✅ Summarize medical reports into simple language  
✅ Chat with AI regarding symptoms & health queries  
✅ Upload & interact with medical PDFs using AI (PDF RAG)

> ⚠️ **Disclaimer:** This tool is for informational & assistance purposes only.  
> It does *not* replace professional medical consultation.
    """)
