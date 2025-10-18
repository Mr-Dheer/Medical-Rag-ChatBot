# bot.py
import os
import streamlit as st
from dotenv import load_dotenv, find_dotenv

# Load .env so HF token is available
load_dotenv(find_dotenv(), override=True)

from langchain_huggingface import (
    HuggingFaceEmbeddings,
    HuggingFaceEndpoint,
    ChatHuggingFace,
)
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate




# -------------------------
# Config (paths & models)
# -------------------------
import os
import streamlit as st
from dotenv import load_dotenv, find_dotenv

# --- Load .env ---
load_dotenv(find_dotenv(), override=True)

from langchain_huggingface import (
    HuggingFaceEmbeddings,
    HuggingFaceEndpoint,
    ChatHuggingFace,
)
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

# -------------------------
# Config
# -------------------------
DB_FAISS_PATH = "vectorstore/db_faiss"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"

# -------------------------
# Helper Functions
# -------------------------
def get_hf_token():
    return (
        os.getenv("HUGGINGFACEHUB_API_TOKEN")
        or os.getenv("HF_TOKEN")
        or os.getenv("HUGGINGFACE_HUB_TOKEN")
    )

@st.cache_resource
def get_vectorstore():
    emb = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    try:
        return FAISS.load_local(DB_FAISS_PATH, emb, allow_dangerous_deserialization=True)
    except Exception as e:
        st.error(f"Failed to load FAISS at '{DB_FAISS_PATH}': {e}")
        return None

@st.cache_resource
def get_mistral_chat_llm(temperature=0.5):
    token = get_hf_token()
    if not token:
        st.warning("No HF token found in environment or .env file.")
        st.stop()

    endpoint = HuggingFaceEndpoint(
        repo_id=HUGGINGFACE_REPO_ID,
        task="conversational",
        huggingfacehub_api_token=token,
        temperature=temperature,
        max_new_tokens=512,
    )
    return ChatHuggingFace(llm=endpoint)

def set_custom_prompt(template: str) -> PromptTemplate:
    return PromptTemplate(template=template, input_variables=["context", "question"])

def render_sources(sources):
    if not sources:
        return
    st.markdown("### 📚 Source Documents")
    for i, d in enumerate(sources, 1):
        meta = d.metadata or {}
        src = meta.get("source", "unknown")
        page = meta.get("page_label", meta.get("page", "?"))
        snippet = (d.page_content or "").strip().replace("\n", " ")
        if len(snippet) > 200:
            snippet = snippet[:200] + "..."
        st.markdown(f"**[{i}]** {src} — p.{page}\n> {snippet}")

# -------------------------
# Streamlit UI
# -------------------------
def main():
    st.set_page_config(page_title="Medical RAG ChatBot", page_icon="🩺", layout="wide")
    st.title("🩺 Medical RAG ChatBot")
    st.caption("Ask questions grounded in your medical knowledge base.")

    # --- Sidebar ---
    with st.sidebar:
        st.header("⚙️ Settings")
        temperature = st.slider("Creativity (temperature)", 0.0, 1.2, 0.5, 0.1)
        k_docs = st.slider("Top-k retrieved docs", 1, 8, 3, 1)
        st.divider()

        token_ok = bool(get_hf_token())
        st.markdown(f"**HF Token:** {'✅ Found' if token_ok else '❌ Missing'}")
        vs_exists = os.path.exists(DB_FAISS_PATH)
        st.markdown(f"**Vectorstore:** {'✅ Found' if vs_exists else '❌ Missing'}")

    # --- Setup ---
    vectorstore = get_vectorstore()
    if vectorstore is None:
        st.info("⚠️ Build your FAISS index first using: `python create_memory_for_llm.py`")

    # --- Chat history ---
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for m in st.session_state.messages:
        st.chat_message(m["role"], avatar=("👤" if m["role"]=="user" else "🩺")).markdown(m["content"])

    # --- Chat input ---
    user_prompt = st.chat_input("Ask your medical question here...")
    if user_prompt:
        st.chat_message("user", avatar="👤").markdown(user_prompt)
        st.session_state.messages.append({"role": "user", "content": user_prompt})

        # Build RAG prompt
        CUSTOM_PROMPT_TEMPLATE = """
        Use the pieces of information provided in the context to answer the user's question.
        If you don't know the answer, say you don't know—do not make one up.
        Only use the given context.

        Context:
        {context}

        Question:
        {question}

        Start the answer directly. No small talk.
        """.strip()

        if vectorstore is not None:
            llm = get_mistral_chat_llm(temperature=temperature)

            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=vectorstore.as_retriever(search_kwargs={"k": k_docs}),
                return_source_documents=True,
                chain_type_kwargs={"prompt": set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)},
            )

            try:
                response = qa_chain.invoke({"query": user_prompt})
                result = (response.get("result") or "").strip()
                sources = response.get("source_documents") or []
                st.chat_message("assistant", avatar="🩺").markdown(result)
                render_sources(sources)
                st.session_state.messages.append({"role": "assistant", "content": result})
            except Exception as e:
                st.error(f"Error: {e}")

    # --- Click-to-view Architecture ---
    ARCH_PATH = os.path.join(os.path.dirname(__file__), "Architecture.png")
    st.markdown("---")
    with st.expander("📊 Show Project Architecture"):
        if os.path.exists(ARCH_PATH):
            st.image(ARCH_PATH, caption="RAG pipeline architecture", width=1400)
            # st.markdown(f"[🔗 Open full size]({ARCH_PATH})")
        else:
            st.info(f"Architecture image not found at: `{ARCH_PATH}`")


if __name__ == "__main__":
    main()
