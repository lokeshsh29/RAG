import os
from dotenv import load_dotenv
from src.components.EmbeddingManager import EmbeddingManager
from src.components.RagRetriever import RAGRetriever
from langchain_groq import ChatGroq
from src.components.VectorStore import VectorStore

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

if __name__ == "__main__":
    embedding_manager=EmbeddingManager()
    vectorstore = VectorStore(
                                    collection_name="pdf_documents",
                                    persist_directory="../data/vector_store"  # same path you used when creating
                                )
    
    rag_retriever=RAGRetriever(vectorstore,embedding_manager)

    

    answer=rag_simple("What is Nvidia's technological advancement in 2024?",rag_retriever,llm)
    print(answer)