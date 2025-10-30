import pandas as pd
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter

class DataProcessor:
    def __init__(self, chunk_size=1000, chunk_overlap=200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
    
    def load_csv(self, uploaded_file):
        """Load and validate CSV file"""
        df = pd.read_csv(uploaded_file)
        return df
    
    def clean_dataframe(self, df):
        """Basic data cleaning"""
        # Fill NaN values
        df = df.fillna('Unknown')
        
        # Convert all columns to string for text processing
        for col in df.columns:
            df[col] = df[col].astype(str)
            
        return df
    
    def dataframe_to_text_chunks(self, df):
        """Convert DataFrame rows to descriptive text chunks"""
        chunks = []
        
        for _, row in df.iterrows():
            # Create a descriptive text for each row
            row_description = "Business Data Record: "
            for col_name, value in row.items():
                row_description += f"{col_name}: {value}, "
            
            chunks.append(row_description.strip(', '))
        
        return chunks
    
    def process_dataframe(self, df):
        """Main processing function"""
        # Clean data
        df_clean = self.clean_dataframe(df)
        
        # Convert to text chunks
        text_chunks = self.dataframe_to_text_chunks(df_clean)
        
        # Split into smaller chunks if needed
        documents = []
        for chunk in text_chunks:
            split_chunks = self.text_splitter.split_text(chunk)
            documents.extend(split_chunks)
        
        return documents