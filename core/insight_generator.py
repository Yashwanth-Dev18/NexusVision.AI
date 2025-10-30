import time

class InsightGenerator:
    def __init__(self, rag_engine):
        self.rag_engine = rag_engine
    
    def generate_automatic_insights(self):
        """Generate automatic business insights"""
        insight_questions = [
            "What are the main trends or patterns in this data?",
            "What are the key metrics or important numbers?",
            "Are there any unusual patterns or outliers?",
            "What business recommendations would you suggest based on this data?",
            "What are the potential opportunities or risks?"
        ]
        
        insights = "## 🔍 Automated Business Insights\n\n"
        
        for i, question in enumerate(insight_questions, 1):
            insights += f"**{i}. {question}**\n"
            answer = self.rag_engine.query(question, k=5)
            insights += f"{answer}\n\n"
            time.sleep(0.5)  # Small delay to avoid overwhelming
        
        return insights