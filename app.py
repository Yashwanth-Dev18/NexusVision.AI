import streamlit as st
import pandas as pd
from core.data_processor import DataProcessor
from core.embeddings_manager import EmbeddingsManager
from core.rag_engine import RAGEngine
from core.insight_generator import InsightGenerator
import time

def main():

    st.set_page_config(page_title="NexusVision.AI", layout="wide")

    st.markdown(
        """
        <h1 style='text-align: center; color: white;'>NexusVision.AI</h1>
        <h4 style='text-align: center;'>AI-Powered Business Intelligence Assistant integrated with RAG & LLM</h4>
        """,
        unsafe_allow_html=True
    )
    
    # Initialize session state
    if 'rag_engine' not in st.session_state:
        st.session_state.rag_engine = None
    if 'data_processed' not in st.session_state:
        st.session_state.data_processed = False
    
    # File upload section
    st.header("📁 Upload Your Data")
    uploaded_file = st.file_uploader("Upload CSV File", type="csv")
    
    if uploaded_file is not None:
        try:
            # Process the uploaded file
            with st.spinner("Processing your data..."):
                processor = DataProcessor()
                df = processor.load_csv(uploaded_file)
                chunks = processor.process_dataframe(df)
                
                # Initialize RAG system
                embeddings_mgr = EmbeddingsManager()
                st.session_state.rag_engine = RAGEngine(embeddings_mgr)
                st.session_state.rag_engine.build_knowledge_base(chunks)
                st.session_state.data_processed = True
                
            st.success("✅ Data processed successfully!")
            
            # Show data preview
            st.subheader("📊 Data Preview")
            st.dataframe(df.head())
            
            # Generate automatic insights
            st.header("🤖 Automated Insights")
            if st.button("Generate Insights"):
                with st.spinner("Analyzing your data..."):
                    insight_gen = InsightGenerator(st.session_state.rag_engine)
                    insights = insight_gen.generate_automatic_insights()
                    st.markdown(insights)
            
            # Q&A Section
            st.header("❓ Ask Questions About Your Data")
            question = st.text_input("Enter your question:")
            
            if question and st.button("Get Answer"):
                with st.spinner("Finding answers..."):
                    answer = st.session_state.rag_engine.query(question)
                    st.info(f"**Answer:** {answer}")
                    
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
    
    else:
        st.info("👆 Please upload a CSV file to get started")
    
    #centered footer
    st.markdown("")
    st.markdown("")
    st.markdown("")
    st.markdown("<div style='text-align: center;'>   Developed by:    Yashwanth Krishna Devanaboina    </div>", unsafe_allow_html=True)
    st.markdown("")
    st.markdown("<div style='text-align: center;'>AI/ML Engineer | Software Developer | CS student at Lnu | AWS Certified Cloud Practitioner | Cisco Certified Data Analyst</div>", unsafe_allow_html=True)
    st.markdown("""<div style='text-align: center;'><a href="https://github.com/Yashwanth-Dev18?tab=repositories" target="_blank">GitHub</a> | <a href="https://www.linkedin.com/in/yashwanth-krishna-devanaboina-66ab83212" target="_blank">LinkedIn</a></div>""", unsafe_allow_html=True)


if __name__ == "__main__":
    main()