# DiagnoseAI: A Unified AI System Combining Disease Prediction and Retrieval-Augmented Medical Consultation

DiagnoseAI is an application-level framework that integrates supervised disease-risk prediction with a Retrieval-Augmented Generation (RAG) system for medical report question answering. 

This repository contains the full implementation, evaluation scripts, and experimental results as described in the research paper.

## 🏗️ System Architecture

The end-to-end architecture of DiagnoseAI consists of two complementary computational pathways exposed through a unified Streamlit interface:

1. **Disease Prediction Module (Structured Data)**
   - Processes structured clinical and demographic variables provided by the user.
   - **Diabetes Prediction**: Evaluated using Logistic Regression, Random Forest, XGBoost, KNN, SVM, and Gradient Boosting (using PIMA Indians Diabetes Database).
   - **Lung Cancer Risk Prediction**: Evaluated using Logistic Regression, SVM, Random Forest, and Gradient Boosting (using Survey Lung Cancer Dataset).
   - Features undergo rigorous preprocessing (median imputation, standard scaling) to avoid data leakage and are evaluated using 5-fold stratified cross-validation before final independent test-set assessment.

2. **Medical Consultation Module (Unstructured Text)**
   - Powered by a **Retrieval-Augmented Generation (RAG)** pipeline.
   - Extracts text from user-uploaded medical reports (PDF).
   - Segments text using `RecursiveCharacterTextSplitter` and embeds chunks using Google Gemini embeddings (`models/gemini-embedding-001`).
   - Stores dense vector embeddings in a local **FAISS Vector Database** for similarity-based retrieval.
   - Generates context-grounded responses using the `Gemini API` LLM, specifically constrained by prompts to avoid hallucinations and inference when sufficient evidence is missing.
   - Evaluated using a fixed 12-question benchmark using **RAGAS** metrics (Context Precision, Context Recall, Faithfulness, and Answer Relevance).

## 📂 Project Structure

- `app.py`: Main Streamlit application entry point connecting the modules.
- `pages/`: Individual Streamlit UI pages for the Chatbot, Report Summarization, and Risk Predictors.
- `modules/`: Core backend functions handling the RAG pipeline and disease models.
- `scripts/`: Source code for reproducing the paper's experiments (`diabetes.py`, `lung_cancer.py`, `rag_ablation.py`, `run_ragas_evaluation.py`).
- `results/`: Contains all generated evaluation outputs (CSVs, ROC curves, calibration curves, subgroup analyses, and RAGAS metrics).
- `tests/`: Environment and model validation scripts.
- `data/`: Raw medical datasets and sample PDF reports used in the evaluation.
- `faiss_index/`: Local vector database storage directory.
- `rag_evaluation_questions.json`: The fixed 12-question benchmark dataset used for the RAG ablation study.

## 🚀 Getting Started

1. **Install dependencies**: `pip install -r requirements.txt`
2. **Environment Variables**: Add your `GOOGLE_API_KEY` to the `.env` file.
3. **Launch the Application**: `streamlit run app.py`

## 📊 Reproducing Experiments

To regenerate the paper's results, run the scripts located in the `scripts/` directory. All outputs will be routed to the `results/` folder.
```bash
python scripts/diabetes.py
python scripts/lung_cancer.py
python scripts/run_ragas_evaluation.py
python scripts/rag_ablation.py
```
