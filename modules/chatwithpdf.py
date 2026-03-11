import streamlit as st
import os
import tempfile
from dotenv import load_dotenv

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains.question_answering import load_qa_chain

# RAGAS
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    context_precision,
    context_recall,
    faithfulness,
    answer_relevancy
)

load_dotenv()

FAISS_FOLDER = "faiss_index"
INDEX_NAME = "index"


# ===============================
# VECTOR DATABASE CREATION
# ===============================

def save_vector_db(docs):

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )

    documents = text_splitter.split_documents(docs)

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )

    vector_db = FAISS.from_documents(documents, embeddings)

    os.makedirs(FAISS_FOLDER, exist_ok=True)

    vector_db.save_local(
        folder_path=FAISS_FOLDER,
        index_name=INDEX_NAME
    )

    return vector_db


def load_vector_db():

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )

    vector_db = FAISS.load_local(
        folder_path=FAISS_FOLDER,
        index_name=INDEX_NAME,
        embeddings=embeddings,
        allow_dangerous_deserialization=True
    )

    return vector_db


# ===============================
# RAG PIPELINE
# ===============================

def rag_pipeline(prompt):

    if not os.path.exists(f"{FAISS_FOLDER}/{INDEX_NAME}.faiss"):
        st.error("⚠️ Please upload a PDF first.")
        return None

    vector_db = load_vector_db()

    docs = vector_db.similarity_search(prompt, k=8)

    model = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0.3
    )

    chain = load_qa_chain(model, chain_type="stuff")

    response = chain(
        {
            "input_documents": docs,
            "question": prompt
        },
        return_only_outputs=True
    )

    answer = response["output_text"]

    st.write("### ✅ Answer")
    st.write(answer)

    retrieved_contexts = [doc.page_content for doc in docs]

    return {
        "question": prompt,
        "answer": answer,
        "contexts": retrieved_contexts
    }


# ===============================
# RAGAS EVALUATION FUNCTION
# ===============================

def run_ragas(evaluation_data):

    dataset = Dataset.from_list(evaluation_data)

    eval_llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0
    )

    result = evaluate(
        dataset,
        metrics=[
            context_precision,
            context_recall,
            faithfulness,
            answer_relevancy
        ],
        llm=eval_llm
    )

    return result


# ===============================
# STREAMLIT UI
# ===============================

def run():

    st.title("📚 Chat with Medical PDF")
    st.write("Upload a medical report and ask questions about it.")

    if "evaluation_data" not in st.session_state:
        st.session_state.evaluation_data = []

    uploaded_file = st.file_uploader("Upload Medical PDF", type=["pdf"])

    if uploaded_file:

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        loader = PyPDFLoader(tmp_path)
        docs = loader.load()

        save_vector_db(docs)

        st.success("✅ PDF processed successfully!")

    user_question = st.text_input("💬 Ask a question based on the PDF")

    if user_question:

        result = rag_pipeline(user_question)

        if result:

            ground_truth = st.text_area(
                "✍ Enter Ground Truth (for evaluation)"
            )

            if ground_truth:

                st.session_state.evaluation_data.append({
                    "question": result["question"],
                    "answer": result["answer"],
                    "contexts": result["contexts"],
                    "ground_truth": ground_truth
                })

                st.success("✔ Added to evaluation dataset")

    st.divider()

    st.subheader("📊 RAGAS Evaluation")

    if st.button("Run RAGAS Evaluation"):

        if len(st.session_state.evaluation_data) == 0:
            st.warning("No evaluation data available")

        else:

            results = run_ragas(st.session_state.evaluation_data)

            st.write("### 📈 RAGAS Results")
            st.write(results)


# Run Streamlit page
run()