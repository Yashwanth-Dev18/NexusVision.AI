import os

class Config:
    # Embeddings
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    
    # Text Processing
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    
    # RAG
    SIMILARITY_TOP_K = 3
    
    # LLM
    MAX_RESPONSE_LENGTH = 512