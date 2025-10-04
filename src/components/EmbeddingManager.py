import numpy as np
from sentence_transformers import SentenceTransformer
import chromadb
from typing import List

class EmbeddingManager:
    """Handles document embedding generation using Sentence Transformer"""

    def __init__(self,model_name:str="sentence-transformers/all-MiniLM-L6-v2"):
        """ 
        Initialize the embedding manager
        Args:
            model_name: HuggingFace model name for sentence embeddings
            """
        self.model_name=model_name
        self.model=None
        self._load_model()

    def _load_model(self):
        """Load the SentenceTransformer model"""

        try:
            print(f"Loading embedding model: {self.model_name}")
            self.model=SentenceTransformer(self.model_name,device="cpu")
            print(f"Model loaded successfully. Embedding dimension: {self.model.get_sentence_embedding_dimension()}")
        except Exception as e:
            print(f"Error loading model {self.model_name}: {e}")
            raise


    def generate_embeddings(self,texts:List[str]) ->np.ndarray:
        """ 
        Generate embeddings for a list of texts

        Args:
            texts: List of text strings to embed

        Returns:
            numpy arrary of embeddings with shape (len(texts),embedding_dim)

        """

        if not self.model:
            raise ValueError("Model not loaded.")
        
        print(f"Generate embeddings for {len(texts)} texts...")
        embeddings=self.model.encode(texts,show_progress_bar=True)
        print(f"Generated embeddings with shape: {embeddings.shape}")
        return embeddings
