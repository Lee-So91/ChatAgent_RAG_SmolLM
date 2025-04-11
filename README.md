Welcome to ChatAgent, your lightweight yet powerful conversational AI assistant built using RAG (Retrieval-Augmented Generation) techniques. This repo uses LangChain, ChromaDB, pandas, and the blazing-fast SmolLM from Ollama, perfect for resource-friendly environments and fast experimentation.

🚀 Features
🔍 RAG-based architecture for context-aware conversations

🧠 SmolLM – the smallest, fastest open-source LLM via Ollama

📚 LangChain for modular, composable pipelines

🗂️ ChromaDB for vector storage and similarity search

📊 pandas for structured data processing

⚡ Lightweight and efficient – runs on most local machines

🧱 Architecture Overview
Data Loading: Using pandas to ingest and manipulate structured/unstructured data.

Embedding & Indexing: Convert text to embeddings and store them using langchain-chroma.

Retrieval: On user query, retrieve relevant chunks using similarity search.

LLM Response: Feed context + query into SmolLM (running via Ollama).

Answer Generation: LangChain orchestrates the RAG pipeline and returns concise answers.
