import os
import json
import pandas as pd
import sys
import time
import numpy as np

import langchain
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from datasets import Dataset

try:
    import ragas
    from ragas import evaluate
    from ragas.metrics import context_precision, context_recall, faithfulness, answer_relevancy
    RAGAS_AVAILABLE = True
    RAGAS_VERSION = ragas.__version__
except ImportError:
    RAGAS_AVAILABLE = False
    RAGAS_VERSION = "Not installed"

from dotenv import load_dotenv

# Ensure API Key is available
load_dotenv()

def run_experiment():
    print(f"RAGAS installation/version status: {RAGAS_VERSION}")
    print(f"Python version: {sys.version.split(' ')[0]}")
    print(f"LangChain version: {langchain.__version__}")
    
    if not RAGAS_AVAILABLE:
        print("RAGAS fails completely. Stopping.")
        return

    generation_model = "gpt-4o-mini"
    evaluation_model = "gpt-4o-mini"
    
    print(f"Generation model: {generation_model}")
    print(f"Evaluation model: {evaluation_model}")

    # 1. Load questions
    try:
        with open("rag_evaluation_questions.json", "r") as f:
            questions = json.load(f)
    except Exception as e:
        print(f"Failed to load questions: {e}")
        return
        
    print(f"Number of questions: {len(questions)}")

    # 2. Load PDF
    pdf_path = "data/raw/UMNwriteup.pdf"
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    # Models
    from langchain_community.embeddings import HuggingFaceEmbeddings
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    llm = ChatOpenAI(model=generation_model, temperature=0.3)
    eval_llm = ChatOpenAI(model=evaluation_model, temperature=0.0)
    
    # Custom RAG Prompt
    rag_prompt_template = """You are a medical AI assistant.
Use the following pieces of retrieved clinical document context to answer the question.

If the provided context does not contain sufficient evidence to answer the question, do not infer, assume, or fabricate information. State clearly that the available clinical document does not provide sufficient information to answer the question.

If the retrieved context contains contradictory information, explicitly identify the contradiction rather than selecting one statement without evidence.

Answer using only information supported by the provided clinical document context.

Context: {context}

Question: {input}

Answer:"""
    rag_prompt = PromptTemplate.from_template(rag_prompt_template)
    
    configurations = [
        {"name": "baseline", "chunk_size": 1000, "chunk_overlap": 150, "top_k": 8},
        {"name": "chunk_size_500", "chunk_size": 500, "chunk_overlap": 150, "top_k": 8},
        {"name": "chunk_size_1500", "chunk_size": 1500, "chunk_overlap": 150, "top_k": 8},
        {"name": "overlap_100", "chunk_size": 1000, "chunk_overlap": 100, "top_k": 8},
        {"name": "overlap_200", "chunk_size": 1000, "chunk_overlap": 200, "top_k": 8},
        {"name": "top_k_3", "chunk_size": 1000, "chunk_overlap": 150, "top_k": 3},
        {"name": "top_k_5", "chunk_size": 1000, "chunk_overlap": 150, "top_k": 5}
    ]

    per_question_results = []
    aggregated_results = []
    
    successful_configs = 0
    failed_configs = []

    for config in configurations:
        print(f"Running config: {config['name']}")
        
        # Split text
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config["chunk_size"], 
            chunk_overlap=config["chunk_overlap"]
        )
        splits = text_splitter.split_documents(docs)
        
        # Vector Store
        vectorstore = FAISS.from_documents(splits, embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={"k": config["top_k"]})
        
        data_for_ragas = {
            "question": [],
            "answer": [],
            "contexts": [],
            "ground_truth": [],
            "question_id": [],
            "category": []
        }
        
        for q in questions:
            q_text = q["question"]
            try:
                retrieved_docs = retriever.invoke(q_text)
                contexts = [doc.page_content for doc in retrieved_docs]
                context_str = "\n\n".join(contexts)
                prompt_text = rag_prompt.format(context=context_str, input=q_text)
                response = llm.invoke(prompt_text)
                answer = response.content
                if isinstance(answer, list):
                    answer = "".join([m.get("text", "") if isinstance(m, dict) else str(m) for m in answer])
            except Exception as e:
                print(f"Error querying RAG: {e}")
                answer = "Error generating response"
                contexts = []
                
            data_for_ragas["question"].append(q_text)
            data_for_ragas["answer"].append(answer)
            data_for_ragas["contexts"].append(contexts)
            data_for_ragas["ground_truth"].append(q["ground_truth"])
            data_for_ragas["question_id"].append(q["question_id"])
            data_for_ragas["category"].append(q["category"])

        # Create evaluation dataset (excluding question_id and category for RAGAS evaluate function)
        eval_dataset_dict = {
            "question": data_for_ragas["question"],
            "answer": data_for_ragas["answer"],
            "contexts": data_for_ragas["contexts"],
            "ground_truth": data_for_ragas["ground_truth"]
        }
        dataset = Dataset.from_dict(eval_dataset_dict)
        metrics = [context_precision, context_recall, faithfulness, answer_relevancy]
        
        try:
            result = evaluate(
                dataset, 
                metrics=metrics, 
                llm=eval_llm,
                embeddings=embeddings
            )
            
            result_df = result.to_pandas()
            
            # Record per-question results
            cp_vals, cr_vals, f_vals, ar_vals = [], [], [], []
            for i, q in enumerate(questions):
                row = result_df.iloc[i]
                
                cp = row.get("context_precision", np.nan)
                cr = row.get("context_recall", np.nan)
                f_val = row.get("faithfulness", np.nan)
                ar = row.get("answer_relevancy", np.nan)
                
                cp_vals.append(cp)
                cr_vals.append(cr)
                f_vals.append(f_val)
                ar_vals.append(ar)
                
                per_question_results.append({
                    "question_id": data_for_ragas["question_id"][i],
                    "category": data_for_ragas["category"][i],
                    "configuration": config["name"],
                    "chunk_size": config["chunk_size"],
                    "chunk_overlap": config["chunk_overlap"],
                    "top_k": config["top_k"],
                    "question": data_for_ragas["question"][i],
                    "answer": data_for_ragas["answer"][i],
                    "ground_truth": data_for_ragas["ground_truth"][i],
                    "contexts": data_for_ragas["contexts"][i],
                    "context_precision": cp if pd.notna(cp) else "Missing",
                    "context_recall": cr if pd.notna(cr) else "Missing",
                    "faithfulness": f_val if pd.notna(f_val) else "Missing",
                    "answer_relevance": ar if pd.notna(ar) else "Missing",
                })
            
            # Record aggregated results
            cp_series = pd.to_numeric(pd.Series(cp_vals), errors='coerce')
            cr_series = pd.to_numeric(pd.Series(cr_vals), errors='coerce')
            f_series = pd.to_numeric(pd.Series(f_vals), errors='coerce')
            ar_series = pd.to_numeric(pd.Series(ar_vals), errors='coerce')
            
            aggregated_results.append({
                "configuration": config["name"],
                "chunk_size": config["chunk_size"],
                "chunk_overlap": config["chunk_overlap"],
                "top_k": config["top_k"],
                "embedding_model": "all-MiniLM-L6-v2",
                "generation_model": generation_model,
                "generation_temperature": 0.3,
                "evaluation_model": evaluation_model,
                "evaluation_temperature": 0.0,
                "context_precision_mean": cp_series.mean(),
                "context_precision_sd": cp_series.std(),
                "context_recall_mean": cr_series.mean(),
                "context_recall_sd": cr_series.std(),
                "faithfulness_mean": f_series.mean(),
                "faithfulness_sd": f_series.std(),
                "answer_relevance_mean": ar_series.mean(),
                "answer_relevance_sd": ar_series.std(),
                "number_of_questions": len(questions)
            })
            
            successful_configs += 1
            print(f"[{config['name']}] CP: {cp_series.mean():.4f}±{cp_series.std():.4f}, CR: {cr_series.mean():.4f}±{cr_series.std():.4f}, F: {f_series.mean():.4f}±{f_series.std():.4f}, AR: {ar_series.mean():.4f}±{ar_series.std():.4f}")

        except Exception as e:
            err_msg = str(e)
            print(f"RAGAS evaluation failed for {config['name']}: {err_msg}")
            failed_configs.append({"configuration": config["name"], "error": err_msg})
            
            for i, q in enumerate(questions):
                per_question_results.append({
                    "question_id": data_for_ragas["question_id"][i],
                    "category": data_for_ragas["category"][i],
                    "configuration": config["name"],
                    "chunk_size": config["chunk_size"],
                    "chunk_overlap": config["chunk_overlap"],
                    "top_k": config["top_k"],
                    "question": data_for_ragas["question"][i],
                    "answer": data_for_ragas["answer"][i],
                    "ground_truth": data_for_ragas["ground_truth"][i],
                    "contexts": data_for_ragas["contexts"][i],
                    "context_precision": "Missing: Evaluation Failed",
                    "context_recall": "Missing: Evaluation Failed",
                    "faithfulness": "Missing: Evaluation Failed",
                    "answer_relevance": "Missing: Evaluation Failed",
                })

    # Save results
    if per_question_results:
        pd.DataFrame(per_question_results).to_csv("ragas_per_question_results.csv", index=False)
    if aggregated_results:
        pd.DataFrame(aggregated_results).to_csv("ragas_ablation_results.csv", index=False)
        
    summary = {
        "source_document": "UMNwriteup.pdf",
        "number_of_documents": 1,
        "number_of_questions": len(questions),
        "question_categories": list(set([q["category"] for q in questions])),
        "baseline_configuration": configurations[0],
        "tested_configurations": configurations,
        "environment": {
            "python_version": sys.version.split(' ')[0],
            "langchain_version": langchain.__version__,
            "ragas_version": RAGAS_VERSION,
            "embedding_model": "all-MiniLM-L6-v2",
            "generation_model": generation_model,
            "generation_temperature": 0.3,
            "evaluation_model": evaluation_model,
            "evaluation_temperature": 0.0
        },
        "execution_status": {
            "successful_configs": successful_configs,
            "failed_configs": failed_configs
        }
    }
    
    with open("rag_evaluation_summary.json", "w") as f:
        json.dump(summary, f, indent=4)
        
    print(f"\nNumber of configurations successfully evaluated: {successful_configs}")
    if failed_configs:
        print("Failed configurations:")
        for fc in failed_configs:
            print(f"  - {fc['configuration']}: {fc['error']}")
            
    print("\nGenerated files:")
    print("  - ragas_per_question_results.csv")
    print("  - ragas_ablation_results.csv")
    print("  - rag_evaluation_summary.json")

if __name__ == "__main__":
    run_experiment()
