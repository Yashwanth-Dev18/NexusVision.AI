import chromadb
from chromadb.utils import embedding_functions

class EmbeddingsManager:
    def __init__(self):
        # Use ChromaDB's default embedding function (no sentence-transformers needed!)
        self.client = chromadb.Client()
        self.collection = self.client.create_collection(name="business_data")
    
    def add_documents(self, documents, metadatas=None):
        # Adding documents to vector database
        # Create IDs for documents
        ids = [f"doc_{i}" for i in range(len(documents))]

        # Add to collection - ONLY include metadatas if provided
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
                # ← No metadatas parameter at all
            )
    
    def similarity_search(self, query, k=3):
        """Search for similar documents"""
        results = self.collection.query(
            query_texts=[query],
            n_results=k
        )
        return results['documents'][0] if results['documents'] else []