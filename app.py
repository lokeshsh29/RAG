import streamlit as st
import os
from dotenv import load_dotenv
from src.components.EmbeddingManager import EmbeddingManager
from src.components.RagRetriever import RAGRetriever
from src.components.VectorStore import VectorStore
from langchain_groq import ChatGroq

# Load environment variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# Initialize once in session_state
if "llm" not in st.session_state:
    st.session_state.llm = ChatGroq(
        groq_api_key=groq_api_key,
        model_name="gemma2-9b-it",
        temperature=0.1,
        max_tokens=1024
    )

if "embedding_manager" not in st.session_state:
    st.session_state.embedding_manager = EmbeddingManager()

if "vectorstore" not in st.session_state:
    # Use a safe path inside current directory for Streamlit Cloud
    st.session_state.vectorstore = VectorStore(
    collection_name="pdf_documents",
    persist_directory="../data/vector_store"
    )

if "retriever" not in st.session_state:
    st.session_state.retriever = RAGRetriever(
        st.session_state.vectorstore,
        st.session_state.embedding_manager
    )

# RAG helper
def rag_simple(query, retriever, llm, top_k=3):
    results = retriever.retrieve(query, top_k=top_k)
    context = "\n\n".join([doc['content'] for doc in results]) if results else ""
    
    if not context:
        return "No relevant context found to answer the question."
    
    prompt = f"""Use the following context to answer the question concisely.
    Context:
    {context}

    Question: {query}

    Answer:"""
    
    response = llm.invoke([prompt])
    return response.content

# ---------------- UI -----------------
st.title("RAG Application 🔍")
st.markdown(
    """
    ### 📊 Welcome to the Annual Reports Analysis App
    
    This application provides insights from **Apple's 2023 & 2024 Annual Reports**  
    and **Nvidia's 2024 Annual Report**.  

    💡 **How to use:**  
    Ask a question, and the app will intelligently analyze the reports to give you a detailed response.
    """
)

st.sidebar.image(
    "https://developer-blogs.nvidia.com/wp-content/uploads/2024/05/genai-multi-modal-rag-featured-960x540.jpg", 
    width=150
)

st.sidebar.markdown(
    """
    ### 🚀 Project
    **GitHub Repository:** [RAG Application](https://github.com/lokeshsh29/RAG)  

    ### 👤 Author
    **Lokesh Shekhar**
    ---
    📝 **Note:**  
    For more extensive usage, additional details, and instructions, please visit the **GitHub repository**.
    """
)

query = st.text_input("Enter your query:")

if st.button("Search") and query.strip() != "":
    with st.spinner("Retrieving relevant documents..."):
        results = rag_simple(
            query=query,
            retriever=st.session_state.retriever,
            llm=st.session_state.llm
        )
    if results:
        st.markdown("### Results:")
        st.write(results)
    else:
        st.write("No relevant documents found for your query.")
