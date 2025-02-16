import json
from pathlib import Path
from typing import Dict, List

import chromadb
from chromadb.config import Settings
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_community.vectorstores import Chroma

class MedicalDatabase:
    def __init__(self, db_dir: str = "./data/chroma"):
        self.db_dir = Path(db_dir)
        self.db_dir.mkdir(parents=True, exist_ok=True)
        self.embeddings = FastEmbedEmbeddings()
        self.vector_store = None

    def load_medical_data(self, data_path: str) -> List[Dict]:
        """Load medical data from JSON file."""
        with open(data_path, 'r') as f:
            return json.load(f)

    def initialize_vector_store(self, medical_data: List[Dict]):
        """Initialize the vector store with medical data."""
        texts = [str(record) for record in medical_data]
        metadata = medical_data

        self.vector_store = Chroma.from_texts(
            texts=texts,
            embedding=self.embeddings,
            persist_directory=str(self.db_dir),
            metadatas=metadata
        )
        self.vector_store.persist()

    def search_similar_cases(self, query: str, k: int = 5) -> List[Dict]:
        """Search for similar medical cases."""
        if not self.vector_store:
            raise ValueError("Vector store not initialized. Call initialize_vector_store first.")
        
        results = self.vector_store.similarity_search(query, k=k)
        return [doc.metadata for doc in results] 