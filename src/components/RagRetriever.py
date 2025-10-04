from typing import List,Dict,Any

class RAGRetriever:
    """Handles query-based retrieval from the vector store"""
    
    def __init__(self, vector_store, embedding_manager):
        """
        Initialize the retriever
        
        Args:
            vector_store: Vector store containing document embeddings
            embedding_manager: Manager for generating query embeddings
        """
        self.vector_store = vector_store
        self.embedding_manager = embedding_manager

    def retrieve(self, query: str, top_k: int = 5, score_threshold: float = 0.0) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents for a query
        
        Args:
            query: The search query
            top_k: Number of top results to return
            score_threshold: Minimum similarity score threshold
            
        Returns:
            List of dictionaries containing retrieved documents and metadata
        """
        print(f"Retrieving documents for query: '{query}'")
        print(f"Top K: {top_k}, Score threshold: {score_threshold}")
        
        # Generate query embedding
        query_embedding = self.embedding_manager.generate_embeddings([query])[0]
        
        try:
            results = self.vector_store.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=top_k
            )
            
            # Process results
            retrieved_docs = []
            
            if results['documents'] and results['documents'][0]:
                documents = results['documents'][0]
                metadatas = results['metadatas'][0]
                distances = results['distances'][0]
                ids = results['ids'][0]
                
                for i, (doc_id, document, metadata, distance) in enumerate(zip(ids, documents, metadatas, distances)):
                    # Convert distance to similarity score (ChromaDB uses cosine distance)
                    similarity_score = 1 - distance
                    
                    if similarity_score >= score_threshold:
                        retrieved_docs.append({
                            'id': doc_id,
                            'content': document,
                            'metadata': metadata,
                            'similarity_score': similarity_score,
                            'distance': distance,
                            'rank': i + 1
                        })
                
                print(f"Retrieved {len(retrieved_docs)} documents (after filtering)")
            else:
                print("No documents found")
            
            return retrieved_docs
        
        
        except Exception as e:
            print(f"Error during retrieval: {e}")
            return []
        
        '''
        # Search in vector store
        try:
            # Use LangChain's similarity_search_with_score
            docs_with_scores = self.vector_store.similarity_search_with_score(query, k=top_k)

            retrieved_docs = []

            for i, (doc, score) in enumerate(docs_with_scores):
                # LangChain scores are similarity scores (or distances depending on version)
                similarity_score = score  # if score is distance, use: similarity_score = 1 - score

                if similarity_score >= score_threshold:
                    retrieved_docs.append({
                        'id': f"doc_{i+1}",
                        'content': doc.page_content,
                        'metadata': doc.metadata,
                        'similarity_score': similarity_score,
                        'rank': i + 1
                    })

            print(f"Retrieved {len(retrieved_docs)} documents (after filtering)")
            return retrieved_docs
            '''