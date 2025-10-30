import pandas as pd

def validate_csv(file):
    """Basic CSV validation"""
    try:
        df = pd.read_csv(file)
        if df.empty:
            return False, "CSV file is empty"
        return True, df
    except Exception as e:
        return False, f"Invalid CSV file: {str(e)}"