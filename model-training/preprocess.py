"""
Mental Health Crisis Detection - Preprocessing Module

This module handles the loading and preprocessing of the Reddit Mental Health dataset.
It includes functions for text cleaning, normalization, and data preparation.
"""
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
import numpy as np
import concurrent.futures
import multiprocessing
from tqdm import tqdm
import time

# Number of CPU cores for parallel processing
NUM_CORES = max(1, multiprocessing.cpu_count() - 1)
print(f"Using {NUM_CORES} CPU cores for parallel processing")

# Download NLTK stopwords
nltk.download('stopwords', quiet=True)
STOP_WORDS = set(stopwords.words('english'))

def remove_urls(text):
    """Remove URLs using regex pattern matching"""
    return re.sub(r'http\S+', '', text)

def clean_special_chars(text):
    """Retain only alphanumeric characters and whitespace"""
    return re.sub(r'[^A-Za-z0-9\s]+', '', text)

def clean_data(df):
    """Sanitize raw dataframe
    
    Args:
        df: Raw dataframe with 'text' and 'class' columns
        
    Returns:
        Cleaned dataframe ready for preprocessing
    """
    # If DataFrame has index as first column, drop it
    if df.columns[0] != 'text' and df.columns[0] != 'class':
        df.drop(columns=df.columns[0], inplace=True)  
    
    # Handle missing values
    df.dropna(inplace=True)  
    
    # Ensure consistent class labels - convert to string
    df['class'] = df['class'].astype(str)
    
    print("Cleaning text (URLs and special characters)...")
    start_time = time.time()
    
    # Use parallel processing for cleaning operations
    with concurrent.futures.ThreadPoolExecutor(max_workers=NUM_CORES) as executor:
        # Clean URLs in parallel
        df['text'] = list(executor.map(remove_urls, df['text']))
        # Clean special characters in parallel
        df['text'] = list(executor.map(clean_special_chars, df['text']))
    
    print(f"Text cleaning completed in {time.time() - start_time:.2f} seconds")
    return df

def process_text(text):
    """Process a single text entry (for parallel processing)"""
    return ' '.join([word for word in text.lower().split() if word not in STOP_WORDS])

def preprocess_text(df):
    """Text normalization pipeline with parallel processing"""
    print("Preprocessing text (lowercasing and stopword removal)...")
    start_time = time.time()
    
    # Use parallel processing
    with concurrent.futures.ThreadPoolExecutor(max_workers=NUM_CORES) as executor:
        df['text'] = list(tqdm(
            executor.map(process_text, df['text']), 
            total=len(df), 
            desc="Processing texts"
        ))
    
    print(f"Text preprocessing completed in {time.time() - start_time:.2f} seconds")
    return df

def prepare_data(file_path, test_size=0.2, random_state=42):
    """End-to-end data preparation
    
    Returns:
        X_train, X_test, y_train, y_test: Stratified splits
    """
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)  # Load raw data
    
    # Print dataset information
    print(f"Dataset shape: {df.shape}")
    print("Original column dtypes:")
    print(df.dtypes)
    
    print("Cleaning data...")
    df = clean_data(df)  # Sanitization
    
    print("Preprocessing text...")
    df = preprocess_text(df)  # Normalization
    
    # Double check class types
    print("\nClass value counts:")
    print(df['class'].value_counts())
    print("Class column dtype:", df['class'].dtype)
    
    # Ensure class labels are strings before splitting
    df['class'] = df['class'].astype(str)
    
    print("Splitting into train/test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        df['text'], df['class'], 
        test_size=test_size, 
        random_state=random_state,
        stratify=df['class']  # Ensure balanced classes
    )
    
    # Verify split data types
    print(f"Train set: {len(X_train)} samples, labels dtype: {y_train.dtype}")
    print(f"Test set: {len(X_test)} samples, labels dtype: {y_test.dtype}")
    
    return X_train, X_test, y_train, y_test

def create_short_sample(args):
    """Create a shortened version of a text (for parallel processing)"""
    i, text, label = args
    
    # Only process longer texts (>30 words)
    if len(text.split()) > 30:
        # Take just the first 15-25 words to create a short sample
        words = text.split()
        short_length = min(len(words), 15 + i % 10)  # Vary between 15-25 words
        short_text = ' '.join(words[:short_length])
        return short_text, label
    
    return None, None

def augment_data_lengths(X_train, y_train):
    """Create augmented samples with different text lengths
    to improve robustness to variable length inputs - with parallel processing
    
    Args:
        X_train: Training text samples
        y_train: Training labels
        
    Returns:
        X_augmented, y_augmented: Augmented datasets with shorter text samples
    """
    print("Augmenting data with varied text lengths...")
    start_time = time.time()
    
    # Ensure y_train is string type
    if isinstance(y_train, pd.Series):
        y_train = y_train.astype(str)
    else:
        y_train = np.array(y_train, dtype=str)
    
    X_augmented = X_train.copy()
    y_augmented = y_train.copy()
    
    # Prepare arguments for parallel processing - only process every 5th sample
    process_indices = range(0, len(X_train), 5)
    args_list = [(i, X_train.iloc[i] if hasattr(X_train, 'iloc') else X_train[i], 
                 y_train.iloc[i] if hasattr(y_train, 'iloc') else y_train[i])
                for i in process_indices]
    
    short_texts = []
    short_labels = []
    
    # Use parallel processing for creating short samples
    with concurrent.futures.ThreadPoolExecutor(max_workers=NUM_CORES) as executor:
        results = list(tqdm(
            executor.map(create_short_sample, args_list),
            total=len(args_list),
            desc="Creating augmented samples"
        ))
    
    # Filter out None results and collect valid samples
    for text, label in results:
        if text is not None:
            short_texts.append(text)
            short_labels.append(label)
    
    # Convert to Series if X_train is a Series
    if isinstance(X_train, pd.Series):
        short_texts = pd.Series(short_texts)
        short_labels = pd.Series(short_labels)
    
    # Combine original and augmented data
    X_augmented = pd.concat([X_augmented, short_texts])
    y_augmented = pd.concat([y_augmented, short_labels])
    
    # Ensure all labels are string type
    y_augmented = y_augmented.astype(str)
    
    print(f"Added {len(short_texts)} augmented samples with varied lengths")
    print(f"Data augmentation completed in {time.time() - start_time:.2f} seconds")
    print(f"Label types in augmented dataset: {y_augmented.dtype}")
    
    return X_augmented, y_augmented

if __name__ == "__main__":
    # Test the preprocessing function
    X_train, X_test, y_train, y_test = prepare_data("../data/reddit_mental_health.csv")
    print("Data preparation complete!")
    
    # Sample augmentation
    X_augmented, y_augmented = augment_data_lengths(X_train, y_train)
    print(f"Original training set: {len(X_train)} samples")
    print(f"Augmented training set: {len(X_augmented)} samples") 