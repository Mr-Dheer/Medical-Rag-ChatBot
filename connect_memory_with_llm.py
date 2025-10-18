import os
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace, HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS

HF_TOKEN = (
    os.getenv("HUGGINGFACEHUB_API_TOKEN")
    or os.getenv("HF_TOKEN")
    or os.getenv("HUGGINGFACE_HUB_TOKEN")
)

HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"

def load_chat_llm(repo_id: str):
    # Under the hood this will use HF Inference "chat" endpoint, not text_generation
    llm_endpoint = HuggingFaceEndpoint(
        repo_id=repo_id,
        task="conversational",                # <- crucial
        huggingfacehub_api_token=HF_TOKEN,
        temperature=0.5,
        max_new_tokens=512,
        model_kwargs={
            # optional extras:
            # "top_p": 0.9,
            # "repetition_penalty": 1.05,
        },
    )
    return ChatHuggingFace(llm=llm_endpoint)  # <- wrap as chat model

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

def set_custom_prompt(template_str: str):
    # Chat model expects a chat prompt; we'll feed the whole template as a single human message
    return ChatPromptTemplate.from_messages([("human", template_str)])

# --- Vector store (must already exist) ---
DB_FAISS_PATH = "vectorstore/db_faiss"
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)

# --- Build Retrieval-QA chain with a CHAT LLM and a CHAT prompt ---
qa_chain = RetrievalQA.from_chain_type(
    llm=load_chat_llm(HUGGINGFACE_REPO_ID),
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={"k": 3}),
    return_source_documents=True,
    chain_type_kwargs={"prompt": set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)},
)

# --- Run ---
user_query = input("Write Query Here: ")
response = qa_chain.invoke({"query": user_query})
print("RESULT:", response["result"])
print("SOURCE DOCUMENTS:", response["source_documents"])
