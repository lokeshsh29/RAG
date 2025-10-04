import os
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import PyMuPDFLoader,PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pathlib import Path
from src.components.EmbeddingManager import EmbeddingManager
from src.components.VectorStore import VectorStore

class DataIngestion:
    
### Read all the pdf's inside the directory
    def process_all_pdfs(self,pdf_directory):
        """Process all PDF files in a directory"""
        all_documents = []
        pdf_dir = Path(pdf_directory)
        
        # Find all PDF files recursively
        pdf_files = list(pdf_dir.glob("**/*.pdf"))
        
        print(f"Found {len(pdf_files)} PDF files to process")
        
        for pdf_file in pdf_files:
            print(f"\nProcessing: {pdf_file.name}")
            try:
                loader = PyPDFLoader(str(pdf_file))
                documents = loader.load()
                
                # Add source information to metadata
                for doc in documents:
                    doc.metadata['source_file'] = pdf_file.name
                    doc.metadata['file_type'] = 'pdf'
                
                all_documents.extend(documents)
                print(f"  ✓ Loaded {len(documents)} pages")
                
            except Exception as e:
                print(f"  ✗ Error: {e}")
        
        print(f"\nTotal documents loaded: {len(all_documents)}")
        return all_documents
    ### Text splitting get into chunks

    def split_documents(self,documents,chunk_size=1000,chunk_overlap=200):
        """Split documents into smaller chunks for better RAG performance"""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        split_docs = text_splitter.split_documents(documents)
        print(f"Split {len(documents)} documents into {len(split_docs)} chunks")
        
        # Show example of a chunk
        if split_docs:
            print(f"\nExample chunk:")
            print(f"Content: {split_docs[0].page_content[:200]}...")
            print(f"Metadata: {split_docs[0].metadata}")
        
        return split_docs

if __name__ == "__main__":
    obj = DataIngestion()
    path = Path(__file__).resolve().parent.parent.parent / "data"
    #print(f"Looking inside: {path.resolve()}")
    #print(f"Exists? {path.exists()}")
    #print(f"Contents: {list(path.glob('*'))}")
    
    all_pdf_documents = obj.process_all_pdfs(path)
    chunks=obj.split_documents(all_pdf_documents)

    embedding_manager=EmbeddingManager()
    vectorstore=VectorStore()

    ### Convert the text to embeddings
    texts=[doc.page_content for doc in chunks]

    ## Generate the Embeddings

    embeddings=embedding_manager.generate_embeddings(texts)

    ##store int he vector dtaabase
    vectorstore.add_documents(chunks,embeddings)
