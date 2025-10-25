# 🩺 Medical RAG ChatBot (Streamlit + LangChain + FAISS + Hugging Face)

WebApp link - https://medical-rag-chatbot-2khwtnyvzmtd3fgoqfztfr.streamlit.app/

---

**A production-leaning Retrieval-Augmented Generation (RAG) chatbot for medical PDFs.**
Built with Streamlit for a clean chat UX, LangChain for orchestration, FAISS for fast local vector search, and Hugging Face Inference for chat LLMs (Mistral-7B-Instruct).


---

## ✨ Features

* **RAG over local PDFs** (`/data`) with FAISS index at `vectorstore/db_faiss`
* **Source citations** rendered inline after each answer
* **Configurable** temperature and top-k retrieved docs (sidebar)
* **Hugging Face Inference** endpoint for the chat model

  * Default model: `mistralai/Mistral-7B-Instruct-v0.3`
* **Lightweight embeddings** for speed: `sentence-transformers/all-MiniLM-L6-v2`
* **Clean Streamlit UI** with custom sticky chat input and an architecture expander
* **CLI mode** to query the RAG pipeline from a terminal

---

## 🧱 Tech Stack

* **Python**, **Streamlit**
* **LangChain** (`langchain`, `langchain-community`, `langchain-huggingface`)
* **FAISS** (local vector store)
* **sentence-transformers** for embeddings
* **Hugging Face Inference API** for chat LLM

---



## 🧠 How it Works (RAG Pipeline)

1. **Ingestion & Chunking** (`create_memory.py`)

   * Loads PDFs from `/data`
   * Splits into ~500-char chunks with 50 overlap (`RecursiveCharacterTextSplitter`)

2. **Embeddings & Index**

   * Embeds chunks with `sentence-transformers/all-MiniLM-L6-v2`
   * Stores vectors in a local **FAISS** index (`vectorstore/db_faiss`)

3. **Chat Model**

   * Uses `HuggingFaceEndpoint` with `task="conversational"`
   * Wrapped as a chat model via `ChatHuggingFace` (default: Mistral-7B-Instruct v0.3)

4. **Retrieval-QA Chain** (`RetrievalQA` with `chain_type="stuff"`)

   * Retrieves top-k relevant chunks
   * Builds a strict prompt that **forbids fabrication** and **uses only context**:

     > “Use the pieces of information provided in the context… If you don't know, say you don't know.”

5. **UI & Citations** (`bot.py`)

   * Streamlit chat, sticky input styling, adjustable settings, and a **sources** panel
   * Optional architecture image (`Architecture.png`) via expander

---


**Questions or ideas?** Open an issue or reach out.
