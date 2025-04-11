
# 🤖 ChatAgent: RAG-Powered Conversational Agent with SmolLM

Welcome to **ChatAgent**, your lightweight yet powerful conversational AI assistant built using **RAG (Retrieval-Augmented Generation)** techniques. This repo uses **LangChain**, **ChromaDB**, **pandas**, and the blazing-fast **SmolLM** from [Ollama](https://ollama.com/library/smollm), perfect for resource-friendly environments and fast experimentation.

---

## 🚀 Features

- 🔍 **RAG-based architecture** for context-aware conversations  
- 🧠 **SmolLM** – the smallest, fastest open-source LLM via Ollama  
- 📚 **LangChain** for modular, composable pipelines  
- 🗂️ **ChromaDB** for vector storage and similarity search  
- 📊 **pandas** for structured data processing  
- ⚡ Lightweight and efficient – runs on most local machines

---

## 🧱 Architecture Overview

1. **Data Loading**: Using `pandas` to ingest and manipulate structured/unstructured data.
2. **Embedding & Indexing**: Convert text to embeddings and store them using `langchain-chroma`.
3. **Retrieval**: On user query, retrieve relevant chunks using similarity search.
4. **LLM Response**: Feed context + query into **SmolLM** (running via Ollama).
5. **Answer Generation**: LangChain orchestrates the RAG pipeline and returns concise answers.

---

## 🧰 Tech Stack

| Tool             | Purpose                                |
|------------------|----------------------------------------|
| LangChain        | RAG pipeline orchestration              |
| ChromaDB         | Vector database for document retrieval |
| Ollama + SmolLM  | Lightweight LLM backend                |
| pandas           | Data preprocessing                     |
| Python           | Programming language                   |

---

## 📦 Installation

### 1. Clone the repo

```bash
git clone https://github.com/yourusername/chatagent-rag-smollm.git
cd chatagent-rag-smollm
```

### 2. Install dependencies

> Create a virtual environment (recommended)

```bash
pip install -r requirements.txt
```

Sample `requirements.txt`:

```txt
langchain
langchain-community
langchain-chroma
pandas
ollama
```

### 3. Run Ollama & Pull SmolLM

Ensure Ollama is installed: [https://ollama.com/download](https://ollama.com/download)

```bash
ollama run smollm
```

---

## 🧪 Usage

```python
from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama
from langchain.vectorstores import Chroma
from langchain.embeddings import OllamaEmbeddings
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.prompts import PromptTemplate

# Load and split docs
loader = TextLoader("data/your_docs.txt")
docs = loader.load()
splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
split_docs = splitter.split_documents(docs)

# Embedding and vector store
embedding = OllamaEmbeddings(model="smollm")
vectorstore = Chroma.from_documents(split_docs, embedding=embedding)

# Set up LLM and RAG pipeline
llm = Ollama(model="smollm")
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever())

# Ask a question
response = qa_chain.run("What does this document say about XYZ?")
print(response)
```

---

## 🧠 Why SmolLM?

SmolLM is optimized for:

- **Speed** – lightning fast generation
- **Resource efficiency** – runs on laptops and minimal setups
- **Offline usage** – no cloud API dependency


