import os
import json
import pandas as pd
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from datasets import Dataset

try:
    from ragas import evaluate
    from ragas.metrics import context_precision, context_recall, faithfulness, answer_relevancy
    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False
    print("RAGAS not available. Will record error.")

from dotenv import load_dotenv

# Ensure API Key is available
load_dotenv()

def run_experiment():
    # 1. Load questions
    with open("rag_evaluation_questions.json", "r") as f:
        questions = json.load(f)

    # 2. Load PDF
    pdf_path = "data/raw/UMNwriteup.pdf"
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    # Models
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite", temperature=0.3)
    eval_llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite", temperature=0.0)
    
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
    
    no_rag_prompt_template = """You are a medical AI assistant.
Answer the following question based on your general knowledge.

Question: {input}

Answer:"""
    no_rag_prompt = PromptTemplate.from_template(no_rag_prompt_template)

    configurations = [
        {"name": "baseline", "chunk_size": 1000, "chunk_overlap": 150, "top_k": 8},
        {"name": "chunk_size_500", "chunk_size": 500, "chunk_overlap": 150, "top_k": 8},
        {"name": "chunk_size_1500", "chunk_size": 1500, "chunk_overlap": 150, "top_k": 8},
        {"name": "overlap_100", "chunk_size": 1000, "chunk_overlap": 100, "top_k": 8},
        {"name": "overlap_200", "chunk_size": 1000, "chunk_overlap": 200, "top_k": 8},
        {"name": "top_k_3", "chunk_size": 1000, "chunk_overlap": 150, "top_k": 3},
        {"name": "top_k_5", "chunk_size": 1000, "chunk_overlap": 150, "top_k": 5}
    ]

    ablation_results = []
    rag_vs_norag = []
    
    # Store RAG answers for the baseline to use in rag_vs_norag
    baseline_rag_answers = {}
    
    for config in configurations:
        print(f"Running config: {config['name']}")
        
        # Split text
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config["chunk_size"], 
            chunk_overlap=config["chunk_overlap"]
        )
        splits = text_splitter.split_documents(docs)
        
        # Vector Store
        import time
        time.sleep(15) # Prevent rate limiting on embed_content
        vectorstore = FAISS.from_documents(splits, embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={"k": config["top_k"]})
        
        data_for_ragas = {
            "question": [],
            "answer": [],
            "contexts": [],
            "ground_truth": []
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
            except Exception as e:
                print(f"Error querying RAG: {e}")
                answer = "Error generating response"
                contexts = []
            time.sleep(5) # Prevent Gemini Free Tier 15 RPM rate limit exhaustion
                
            data_for_ragas["question"].append(q_text)
            data_for_ragas["answer"].append(answer)
            data_for_ragas["contexts"].append(contexts)
            data_for_ragas["ground_truth"].append(q["ground_truth"])
            
            if config["name"] == "baseline":
                baseline_rag_answers[q["question_id"]] = answer
                
        # Run RAGAS
        if RAGAS_AVAILABLE:
            try:
                dataset = Dataset.from_dict(data_for_ragas)
                metrics = [context_precision, context_recall, faithfulness, answer_relevancy]
                result = evaluate(
                    dataset, 
                    metrics=metrics, 
                    llm=eval_llm,
                    embeddings=embeddings
                )
                
                ablation_results.append({
                    "configuration": config["name"],
                    "chunk_size": config["chunk_size"],
                    "chunk_overlap": config["chunk_overlap"],
                    "top_k": config["top_k"],
                    "embedding_model": "models/gemini-embedding-001",
                    "context_precision": result.get("context_precision", "Failed"),
                    "context_recall": result.get("context_recall", "Failed"),
                    "faithfulness": result.get("faithfulness", "Failed"),
                    "answer_relevance": result.get("answer_relevancy", "Failed")
                })
            except Exception as e:
                print(f"RAGAS evaluation failed for {config['name']}: {e}")
                ablation_results.append({
                    "configuration": config["name"],
                    "chunk_size": config["chunk_size"],
                    "chunk_overlap": config["chunk_overlap"],
                    "top_k": config["top_k"],
                    "embedding_model": "models/gemini-embedding-001",
                    "context_precision": "Error",
                    "context_recall": "Error",
                    "faithfulness": "Error",
                    "answer_relevance": "Error"
                })
        else:
            ablation_results.append({
                    "configuration": config["name"],
                    "chunk_size": config["chunk_size"],
                    "chunk_overlap": config["chunk_overlap"],
                    "top_k": config["top_k"],
                    "embedding_model": "models/gemini-embedding-001",
                    "context_precision": "RAGAS not installed",
                    "context_recall": "RAGAS not installed",
                    "faithfulness": "RAGAS not installed",
                    "answer_relevance": "RAGAS not installed"
                })

    # Save ablation results
    pd.DataFrame(ablation_results).to_csv("rag_ablation_results.csv", index=False)
    
    # Run NO-RAG
    print("Running No-RAG baseline...")
    for q in questions:
        q_text = q["question"]
        try:
            no_rag_resp = llm.invoke(no_rag_prompt.format(input=q_text))
            no_rag_answer = no_rag_resp.content
        except Exception as e:
            print(f"Error querying No-RAG: {e}")
            no_rag_answer = "Error generating response"
        time.sleep(5)
            
        rag_vs_norag.append({
            "question_id": q["question_id"],
            "category": q["category"],
            "rag_answer": baseline_rag_answers.get(q["question_id"], "N/A"),
            "no_rag_answer": no_rag_answer,
            "ground_truth": q["ground_truth"],
            "rag_assessment": "REQUIRES HUMAN REVIEW",
            "no_rag_assessment": "REQUIRES HUMAN REVIEW"
        })
        
    pd.DataFrame(rag_vs_norag).to_csv("rag_vs_norag_results.csv", index=False)
    
    # Write summary
    import sys
    import langchain
    try:
        import ragas
        ragas_version = ragas.__version__
    except:
        ragas_version = "Not installed"
        
    summary = {
        "source_document": "UMNwriteup.pdf",
        "number_of_documents": 1,
        "number_of_questions": len(questions),
        "question_categories": list(set([q["category"] for q in questions])),
        "baseline_configuration": configurations[0],
        "tested_configurations": configurations,
        "environment": {
            "python_version": sys.version,
            "langchain_version": langchain.__version__,
            "ragas_version": ragas_version,
            "generation_model": "gemini-2.5-flash",
            "generation_temperature": 0.3,
            "embedding_model": "models/gemini-embedding-001",
            "evaluation_model": "gemini-2.5-flash",
            "evaluation_temperature": 0.0
        }
    }
    
    with open("rag_evaluation_summary.json", "w") as f:
        json.dump(summary, f, indent=4)
        
    print("Evaluation complete.")

if __name__ == "__main__":
    run_experiment()
