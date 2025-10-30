import chromadb
from chromadb.utils import embedding_functions

class EmbeddingsManager:
    def __init__(self):
        # Use ChromaDB's default embedding function (no sentence-transformers needed!)
        self.client = chromadb.Client()
        self.collection = self.client.create_collection(name="business_data")
    
    def add_documents(self, documents, metadatas=None):
        """Add documents to vector database"""
        if metadatas is None:
            metadatas = [{} for _ in documents]
        
        # Create IDs for documents
        ids = [f"doc_{i}" for i in range(len(documents))]
        
        # Add to collection
        self.collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
    
    def similarity_search(self, query, k=3):
        """Search for similar documents"""
        results = self.collection.query(
            query_texts=[query],
            n_results=k
        )
        return results['documents'][0] if results['documents'] else []