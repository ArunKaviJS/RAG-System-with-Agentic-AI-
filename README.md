# RAG-System-with-Agentic-AI-
Design and implement a Retrieval-Augmented Generation (RAG) system that integrates agentic behavior using Python libraries and either a cloud-based LLM (e.g., OpenAI, Azure)  or a local model 
🔍 README Instructions – RAG System with Agentic AI
📘 Project Overview
This project implements a Retrieval-Augmented Generation (RAG) pipeline enhanced with Agentic AI. It enables users to ask intelligent questions from a document corpus (e.g., Indian History) and get verified, context-rich answers using large language models (LLMs). It was developed in Google Colab due to local system limitations and later adapted to Streamlit using a lightweight model.

⚙️ How It Works
Document Embedding
Text documents are embedded into vector format using HuggingFace embeddings.

Vector Store
Embedded vectors are stored and queried using FAISS, allowing efficient similarity search.

Agents in Action

Retriever Agent: Finds relevant text chunks for the query.

QA Agent: Generates an initial answer using the context and user query.

Verifier Agent: Checks if the answer is aligned with the input and refines it if needed.

Streamlit Frontend
A user-friendly UI lets users enter questions and view the intelligent response.

🚧 Issues Faced During Development
Falcon Model (tiiuae/falcon-rw-1b):
Caused CUDA out-of-memory issues and was too large for local or basic Colab use.

GPT-2:
Lightweight and easy to run, but generated poor or inaccurate answers.

Flan-T5-Small:
Final choice for Streamlit deployment. Balanced in accuracy and memory usage. Performed well on Colab and suitable for real-world use cases.

Local Environment Challenges:
Local development faced frequent crashes due to computation load. Transitioned fully to Google Colab for processing, and used lightweight model for frontend.

🌐 Deployment Strategy
Google Colab was used for intensive model operations and testing.

A streamlined version with flan-t5-small was integrated into a Streamlit app for public access.

The final Streamlit app includes stylized titles and a polished UI using HTML/CSS.

📈 Benefits of the Project
Low-cost and lightweight solution for intelligent document querying.

Easily adaptable to any document corpus (e.g., legal, medical, educational).

Ideal for companies seeking affordable, client-ready QA systems.

Enhances explainability and trust by showing the retrieved context and verified answers.

✅ Final Conclusion
This RAG system demonstrates that high-quality QA systems can be built using:

Lightweight models like Flan-T5

Open-source tools like LangChain, FAISS, and Streamlit

Simple integration of agentic verification

This model is a cost-efficient, client-friendly, and scalable solution for any organization looking to deploy document intelligence features with minimal resources.