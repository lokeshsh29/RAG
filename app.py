import streamlit as st
from src.components.EmbeddingManager import EmbeddingManager
from src.components.RagRetriever import RAGRetriever
from langchain_groq import ChatGroq
from src.components.VectorStore import VectorStore
import os
from dotenv import load_dotenv

load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

llm=ChatGroq(groq_api_key=groq_api_key,model_name="gemma2-9b-it",temperature=0.1,max_tokens=1024)

def main():
    print("Hello from rag!")

def rag_simple(query,retriever,llm,top_k=3):
    ## retriever the context
    results=retriever.retrieve(query,top_k=top_k)
    context="\n\n".join([doc['content'] for doc in results]) if results else ""
    
    if not context:
        return "No relevant context found to answer the question."
    
    ## generate the answwer using GROQ LLM
    prompt=f"""Use the following context to answer the question concisely.
        Context:
        {context}

        Question: {query}

        Answer:"""
    
    response=llm.invoke([prompt.format(context=context,query=query)])
    return response.content

embedding_manager=EmbeddingManager()
vectorstore = VectorStore(
                                collection_name="pdf_documents",
                                persist_directory="../data/vector_store"  # same path you used when creating
                            )

rag_retriever=RAGRetriever(vectorstore,embedding_manager)


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

if st.button("Search") and query.strip()!="":
    with st.spinner("Retrieving relevant documents:..."):
        results=rag_simple(query=query,retriever=rag_retriever,llm=llm)
    if results:
        st.markdown("### Results:")
        st.write(results)
    else:
        st.write("No relevant documents found for your query.")