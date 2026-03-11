import streamlit as st
from modules.chatwithpdf import run

st.set_page_config(page_title="Medical Report Assistant", page_icon="📄")
run()
