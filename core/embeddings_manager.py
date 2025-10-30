import chromadb
from chromadb.utils import embedding_functions

class EmbeddingsManager:
    def __init__(self):
        self.client = chromadb.Client()
        
        # FIX: Check if collection exists before creating
        try:
            # Try to get existing collection
            self.collection = self.client.get_collection(name="business_data")
        except:
            # If it doesn't exist, create it
            self.collection = self.client.create_collection(name="business_data")
    
    def add_documents(self, documents, metadatas=None):
        """Add documents to vector database"""
        # Create IDs for documents
        ids = [f"doc_{i}" for i in range(len(documents))]
        
        # FIX: Clear existing data before adding new data
        self.collection.delete(ids=ids)  # Delete any existing docs with same IDs
        
        # Add to collection
        if metadatas:
            self.collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
        else:
            self.collection.add(
                documents=documents,
                ids=ids
            )
    
    def similarity_search(self, query, k=3):
        """Search for similar documents"""
        results = self.collection.query(
            query_texts=[query],
            n_results=k
        )
        return results['documents'][0] if results['documents'] else []