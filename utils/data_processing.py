import pandas as pd
import re

def clean_dataframe(df):
    """Clean and standardize dataframe"""
    # Remove extra whitespace
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype(str).str.strip()
    
    # Standardize N/A values
    df = df.replace(['', 'nan', 'None'], 'N/A')
    
    return df

def add_abstract_metrics(df):
    """Add abstract-related metrics to dataframe"""
    if 'abstract' in df.columns:
        df = df.copy()
        df['has_abstract'] = df['abstract'] != 'N/A'
        df['abstract_length'] = df['abstract'].str.len()
        df['abstract_length'].fillna(0, inplace=True)
    
    return df

def extract_keywords_from_abstracts(df, n_keywords=20):
    """Extract most common keywords from abstracts"""
    if 'abstract' not in df.columns:
        return []
    
    # Combine all abstracts
    all_abstracts = ' '.join(df[df['abstract'] != 'N/A']['abstract'].tolist())
    
    # Simple keyword extraction (you could use more sophisticated NLP)
    words = re.findall(r'\b[a-zA-Z]{4,}\b', all_abstracts.lower())
    
    # Common stop words to filter out
    stop_words = {
        'study', 'research', 'results', 'methods', 'analysis', 'data',
        'patients', 'clinical', 'medical', 'health', 'disease', 'treatment'
    }
    
    # Count words
    word_counts = {}
    for word in words:
        if word not in stop_words:
            word_counts[word] = word_counts.get(word, 0) + 1
    
    # Return top keywords
    return sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:n_keywords]