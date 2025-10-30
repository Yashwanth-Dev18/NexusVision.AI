class FlanT5Prompts:
    @staticmethod
    def qa_prompt():
        return """Based on the following business data, answer the question clearly and concisely.

Relevant Data:
{context}

Question: {question}

Provide a direct answer focusing on the data provided:"""
    
    @staticmethod
    def insight_prompt():
        return """Analyze this business data and provide key insights:

{context}

Focus on trends, patterns, and business implications:"""