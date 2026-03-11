import validators, streamlit as st
from dotenv import load_dotenv
import os

from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain.schema import Document
from youtube_transcript_api import YouTubeTranscriptApi

# ------------------------------#
# ✅ Streamlit UI Config
# ------------------------------#
st.set_page_config(page_title="Summarize YouTube Video / Website", page_icon="🧠")
st.title("📄 Summarize YouTube Video or Website Content")

# ------------------------------#
# ✅ Load Environment Variables
# ------------------------------#
load_dotenv()

with st.sidebar:
    groq_api_key = st.text_input("🔑 Enter Groq API Key", type="password")
    st.caption("Get API key from: https://console.groq.com/keys")

# ------------------------------#
# ✅ Input Field
# ------------------------------#
generic_url = st.text_input("Paste YouTube / Website URL here 👇")

# ------------------------------#
# ✅ Initialize LLM
# ------------------------------#
# Latest recommended Groq model (fast + accurate)
llm = ChatGroq(
    model="llama-3.3-70b-versatile",   # ✅ updated model
    api_key=groq_api_key               # ✅ correct parameter
)

prompt_template = """
Summarize the following content clearly and concisely in **200–300 words**.

Content:
{text}
"""

prompt = PromptTemplate(template=prompt_template, input_variables=["text"])

# ------------------------------#
# ✅ Main Execution
# ------------------------------#
if st.button("🚀 Generate Summary"):
    if not groq_api_key.strip() or not generic_url.strip():
        st.error("⚠️ Please enter both Groq API Key and a URL")
    elif not validators.url(generic_url):
        st.error("❌ Invalid URL. Enter a valid YouTube or website link.")
    else:
        try:
            with st.spinner("🔍 Fetching content and generating summary..."):
                docs = []

                # ✅ Detect YouTube URL
                if "youtube.com" in generic_url or "youtu.be" in generic_url:
                    try:
                        video_id = (
                            generic_url.split("v=")[-1].split("&")[0]
                            if "youtube" in generic_url
                            else generic_url.split("/")[-1]
                        )

                        transcript = YouTubeTranscriptApi.get_transcript(video_id)
                        text = " ".join([t["text"] for t in transcript])

                        docs = [Document(page_content=text)]

                    except Exception as yt_error:
                        st.error(f"❌ Could not fetch YouTube transcript: {yt_error}")

                # ✅ Webpage summarization
                else:
                    try:
                        loader = UnstructuredURLLoader(
                            urls=[generic_url],
                            ssl_verify=False,
                            headers={"User-Agent": "Mozilla/5.0"}
                        )
                        docs = loader.load()

                    except Exception as web_error:
                        st.error(f"❌ Error loading webpage: {web_error}")

                if docs:
                    chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt)
                    summary = chain.run(docs)

                    st.success("✅ Summary Generated Successfully!")
                    st.write(summary)

                else:
                    st.error("⚠️ Nothing to summarize. Please try another URL.")

        except Exception as e:
            st.exception(f"🚨 Error occurred: {e}")
