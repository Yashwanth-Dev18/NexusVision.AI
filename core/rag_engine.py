from models.llm_client import FlanT5Client
from models.prompt_templates import FlanT5Prompts

class RAGEngine:
    def __init__(self, embeddings_manager):
        self.embeddings_manager = embeddings_manager
        self.llm = FlanT5Client()
        self.prompts = FlanT5Prompts()
    
    def build_knowledge_base(self, documents):
        """Build the vector knowledge base"""
        self.embeddings_manager.add_documents(documents)
    
    def query(self, question, k=3):
        """Query the RAG system"""
        # Find relevant chunks
        relevant_chunks = self.embeddings_manager.similarity_search(question, k=k)
        
        if not relevant_chunks:
            return "I couldn't find relevant information in the data to answer this question."
        
        # Combine chunks into context
        context = "\n\n".join(relevant_chunks)
        
        # Format prompt
        prompt = self.prompts.qa_prompt().format(
            question=question, 
            context=context
        )
        
        # Get response from LLM
        response = self.llm.generate_response(prompt)
        return response