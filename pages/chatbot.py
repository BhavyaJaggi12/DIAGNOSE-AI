import streamlit as st
import google.generativeai as genai
import os
from dotenv import load_dotenv

# ===============================
# Load environment variables
# ===============================
load_dotenv()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# ===============================
# Initialize Gemini model
# ===============================
model = genai.GenerativeModel("gemini-2.5-flash")

# ===============================
# Initialize chat history
# ===============================
if "messages" not in st.session_state:
    st.session_state.messages = []

if "chat_session" not in st.session_state:
    st.session_state.chat_session = model.start_chat(history=[])

# ===============================
# Streamlit UI
# ===============================
st.title("🤖 DIAGNOSE AI Chatbot")
st.subheader("🌍 Multilingual Medical Assistant")

# Language selector
language = st.selectbox(
    "Choose Language:",
    [
        "English",
        "Hindi",
        "Tamil",
        "Telugu",
        "Bengali",
        "Marathi",
        "Gujarati",
        "Kannada",
        "Malayalam"
    ]
)

# Language instruction
language_prompt = f"""
You are a medical assistant AI.

Respond only in {language}.
Explain medical terms in simple {language}.
Provide helpful healthcare information but remind users that this is not medical advice.
"""

user_input = st.text_input("Enter your health question")

# ===============================
# Generate response
# ===============================
if st.button("Generate Response"):

    if user_input.strip():

        st.session_state.messages.append(
            {"role": "user", "content": user_input}
        )

        final_prompt = f"{language_prompt}\nUser question: {user_input}"

        response = st.session_state.chat_session.send_message(final_prompt)

        reply = response.text

        st.session_state.messages.append(
            {"role": "assistant", "content": reply}
        )

    else:
        st.warning("Please enter a prompt.")


# ===============================
# Display chat history
# ===============================
if st.session_state.messages:

    st.subheader("💬 Chat History")

    for msg in st.session_state.messages:

        if msg["role"] == "user":
            st.markdown(f"🧑 **You:** {msg['content']}")

        else:
            st.markdown(f"🤖 **Diagnose AI ({language}):** {msg['content']}")