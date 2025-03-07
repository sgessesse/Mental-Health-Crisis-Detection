"""
Mental Health Crisis Detection - Model Evaluation Module

This module evaluates the DistilBERT model with a focus on:
1. Overall performance metrics
2. Robustness to different text lengths
3. Memory usage analysis
4. Real-world text testing
"""
import torch
import numpy as np
import pandas as pd
from transformers import DistilBertTokenizerFast
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os
import gc
import joblib
import psutil
import re
import nltk
from nltk.corpus import stopwords
from tqdm import tqdm

# Default model directory
DEFAULT_MODEL_DIR = "../app/model"

# Set up device - use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Evaluation using device: {device}")

def load_model(model_dir=DEFAULT_MODEL_DIR):
    """Load the trained model from disk
    
    Args:
        model_dir: Directory containing the model files
        
    Returns:
        model: Loaded model
        tokenizer: Tokenizer
        label_encoder: Label encoder
    """
    print(f"Loading model from {model_dir}...")
    
    # Try to load TorchScript model first (faster inference)
    ts_path = os.path.join(model_dir, "model_optimized.pt")
    if os.path.exists(ts_path):
        model = torch.jit.load(ts_path, map_location=device)
        print("Loaded TorchScript model")
    else:
        # Fallback to regular model
        from transformers import DistilBertForSequenceClassification
        model = DistilBertForSequenceClassification.from_pretrained(model_dir).to(device)
        model.eval()  # Set to evaluation mode
        print("Loaded standard model")
    
    # Load tokenizer and label encoder
    tokenizer = DistilBertTokenizerFast.from_pretrained(model_dir)
    label_encoder = joblib.load(os.path.join(model_dir, "label_encoder.joblib"))
    
    return model, tokenizer, label_encoder

def evaluate_model(model, tokenizer, X_test, y_test):
    """Evaluate model performance on test set
    
    Args:
        model: Trained model
        tokenizer: Tokenizer
        X_test: Test text data
        y_test: Test labels
        
    Returns:
        Dictionary with evaluation results
    """
    print("Evaluating model on test set...")
    
    # Convert model to evaluation mode if possible
    if hasattr(model, 'eval'):
        model.eval()
    
    all_preds = []
    all_labels = []
    total_time = 0
    
    # Process in batches to avoid memory issues
    batch_size = 32
    if torch.cuda.is_available():
        batch_size = 64
    
    # Create progress bar
    num_batches = (len(X_test) + batch_size - 1) // batch_size
    progress_bar = tqdm(total=num_batches, desc="Evaluating batches")
    
    for i in range(0, len(X_test), batch_size):
        batch_texts = X_test.iloc[i:i+batch_size] if hasattr(X_test, 'iloc') else X_test[i:i+batch_size]
        batch_labels = y_test[i:i+batch_size]
        
        # Preprocess and tokenize
        inputs = tokenizer(batch_texts.tolist() if hasattr(batch_texts, 'tolist') else batch_texts, 
                         padding=True, truncation=True, max_length=256, return_tensors="pt")
        
        # Move inputs to device
        inputs = {key: val.to(device) for key, val in inputs.items()}
        
        # Record inference time
        start_time = time.time()
        with torch.no_grad():
            if isinstance(model, torch.jit.ScriptModule):
                outputs = model(inputs["input_ids"], inputs["attention_mask"])
                logits = outputs[0] if isinstance(outputs, tuple) else outputs
            else:
                outputs = model(**inputs)
                logits = outputs.logits
        
        elapsed = time.time() - start_time
        total_time += elapsed
        
        # Get predictions
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        
        all_preds.extend(preds)
        all_labels.extend(batch_labels)
        
        # Clean up to save memory
        del inputs, outputs, logits, preds
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Update progress bar
        progress_bar.update(1)
    
    # Close progress bar
    progress_bar.close()
    
    # Calculate average inference time
    avg_time_per_sample = total_time / len(X_test)
    print(f"Average inference time per sample: {avg_time_per_sample*1000:.2f} ms")
    
    # Ensure labels are numpy arrays
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(all_labels, all_preds),
        'f1_weighted': f1_score(all_labels, all_preds, average='weighted'),
        'precision_weighted': precision_score(all_labels, all_preds, average='weighted'),
        'recall_weighted': recall_score(all_labels, all_preds, average='weighted'),
        'avg_inference_time_ms': avg_time_per_sample * 1000
    }
    
    print("\nEvaluation Results:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    return metrics

def test_text_length_robustness(model, tokenizer, X_test, y_test):
    """Test model performance across different text lengths
    
    Args:
        model: Trained model
        tokenizer: Tokenizer
        X_test: Test text data
        y_test: Test labels
        
    Returns:
        Dictionary with results by text length bin
    """
    print("\nTesting model robustness to different text lengths...")
    
    # Get text lengths
    text_lengths = [len(text.split()) for text in X_test]
    
    # Create bins for text lengths
    bins = [(0, 10), (11, 30), (31, 50), (51, 100), (101, 200), (201, float('inf'))]
    bin_labels = ['Very Short', 'Short', 'Medium', 'Long', 'Very Long', 'Extra Long']
    
    results = {}
    
    for i, (min_len, max_len) in enumerate(bins):
        # Filter samples by length
        indices = [idx for idx, length in enumerate(text_lengths) 
                  if min_len <= length <= max_len]
        
        if not indices:  # Skip if no samples in this bin
            continue
            
        bin_X = X_test.iloc[indices] if hasattr(X_test, 'iloc') else X_test[indices]
        bin_y = y_test.iloc[indices] if hasattr(y_test, 'iloc') else y_test[indices]
        
        print(f"\n{bin_labels[i]} texts ({min_len}-{max_len} words): {len(bin_X)} samples")
        
        # Process in small batches
        all_preds = []
        batch_size = 16 if not torch.cuda.is_available() else 32
        
        # Create progress bar
        progress_bar = tqdm(total=(len(bin_X) + batch_size - 1) // batch_size, 
                          desc=f"Testing {bin_labels[i]} texts")
        
        for j in range(0, len(bin_X), batch_size):
            batch_texts = bin_X.iloc[j:j+batch_size].tolist() if hasattr(bin_X, 'iloc') else bin_X[j:j+batch_size]
            
            # Dynamic max_length based on text length
            max_length = min(256, max(32, max([len(text.split()) for text in batch_texts]) + 20))
            
            inputs = tokenizer(batch_texts, padding=True, truncation=True, 
                              max_length=max_length, return_tensors="pt")
            
            # Move inputs to device
            inputs = {key: val.to(device) for key, val in inputs.items()}
            
            with torch.no_grad():
                if isinstance(model, torch.jit.ScriptModule):
                    outputs = model(inputs["input_ids"], inputs["attention_mask"])
                    logits = outputs[0] if isinstance(outputs, tuple) else outputs
                else:
                    outputs = model(**inputs)
                    logits = outputs.logits
            
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            
            del inputs, outputs, logits
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Update progress bar
            progress_bar.update(1)
        
        # Close progress bar
        progress_bar.close()
        
        # Calculate metrics for this bin
        bin_report = classification_report(bin_y, all_preds, 
                                        target_names=["NON-SUICIDAL", "SUICIDAL"], 
                                        output_dict=True)
        
        print(f"F1 Score: {bin_report['weighted avg']['f1-score']:.3f}")
        print(f"Accuracy: {bin_report['accuracy']:.3f}")
        
        # Store results
        results[bin_labels[i]] = {
            'f1': bin_report['weighted avg']['f1-score'],
            'accuracy': bin_report['accuracy'],
            'count': len(bin_X)
        }
    
    # Plot results by text length
    if results:
        plt.figure(figsize=(12, 6))
        
        bins = list(results.keys())
        f1_scores = [results[b]['f1'] for b in bins]
        accuracies = [results[b]['accuracy'] for b in bins]
        
        x = np.arange(len(bins))
        width = 0.35
        
        plt.bar(x - width/2, f1_scores, width, label='F1 Score')
        plt.bar(x + width/2, accuracies, width, label='Accuracy')
        
        plt.xlabel('Text Length')
        plt.ylabel('Score')
        plt.title('Model Performance by Text Length')
        plt.xticks(x, bins)
        plt.ylim(0, 1.1)
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig("text_length_performance.png")
        plt.close()
        print(f"Text length performance plot saved to text_length_performance.png")
    
    return results

def measure_memory_usage(model, tokenizer):
    """Measure memory usage during model loading and inference
    
    Args:
        model: Trained model
        tokenizer: Tokenizer
        
    Returns:
        Dictionary with memory usage stats
    """
    print("\nMeasuring memory usage...")
    
    memory_stats = {}
    
    # Report model size
    if hasattr(model, 'parameters'):
        model_size_mb = sum(p.element_size() * p.nelement() for p in model.parameters()) / (1024 * 1024)
        print(f"Model parameter size: {model_size_mb:.2f} MB")
        memory_stats['model_size_mb'] = model_size_mb
    
    # Current memory usage
    process = psutil.Process(os.getpid())
    base_memory = process.memory_info().rss / (1024 * 1024)
    print(f"Current memory usage: {base_memory:.2f} MB")
    memory_stats['base_memory_mb'] = base_memory
    
    # GPU memory if available
    if torch.cuda.is_available():
        gpu_memory_allocated = torch.cuda.memory_allocated() / (1024 * 1024)
        gpu_memory_reserved = torch.cuda.memory_reserved() / (1024 * 1024)
        print(f"GPU memory allocated: {gpu_memory_allocated:.2f} MB")
        print(f"GPU memory reserved: {gpu_memory_reserved:.2f} MB")
        memory_stats['gpu_memory_allocated_mb'] = gpu_memory_allocated
        memory_stats['gpu_memory_reserved_mb'] = gpu_memory_reserved
    
    # Test with different batch sizes
    batch_sizes = [1, 4, 8, 16, 32]
    test_text = "This is a sample text to test memory usage during inference. "
    test_text = test_text * 10  # Make it longer
    
    batch_results = {}
    
    print("\nMemory usage during inference:")
    for batch_size in batch_sizes:
        # Clear memory before test
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Measure baseline
        process = psutil.Process(os.getpid())
        start_memory = process.memory_info().rss / (1024 * 1024)
        
        if torch.cuda.is_available():
            start_gpu_memory = torch.cuda.memory_allocated() / (1024 * 1024)
        
        # Create batch
        texts = [test_text] * batch_size
        
        # Tokenize
        inputs = tokenizer(texts, padding=True, truncation=True, max_length=256, return_tensors="pt")
        
        # Move inputs to device
        inputs = {key: val.to(device) for key, val in inputs.items()}
        
        # Run inference
        with torch.no_grad():
            if isinstance(model, torch.jit.ScriptModule):
                # TorchScript model
                outputs = model(inputs["input_ids"], inputs["attention_mask"])
            else:
                # Regular model
                outputs = model(**inputs)
        
        # Measure peak memory
        peak_memory = process.memory_info().rss / (1024 * 1024)
        
        if torch.cuda.is_available():
            peak_gpu_memory = torch.cuda.memory_allocated() / (1024 * 1024)
            gpu_memory_used = peak_gpu_memory - start_gpu_memory
            print(f"GPU - Batch size {batch_size}: +{gpu_memory_used:.2f} MB total, {gpu_memory_used / batch_size:.2f} MB per sample")
        
        # Calculate memory usage
        memory_used = peak_memory - start_memory
        per_sample = memory_used / batch_size
        
        print(f"CPU - Batch size {batch_size}: +{memory_used:.2f} MB total, {per_sample:.2f} MB per sample")
        
        batch_results[batch_size] = {
            'total_mb': memory_used,
            'per_sample_mb': per_sample
        }
        
        if torch.cuda.is_available():
            batch_results[batch_size]['gpu_total_mb'] = gpu_memory_used
            batch_results[batch_size]['gpu_per_sample_mb'] = gpu_memory_used / batch_size
        
        # Clean up
        del inputs, outputs
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    memory_stats['batch_results'] = batch_results
    return memory_stats

def predict_text(model, tokenizer, text, label_encoder=None):
    """Make prediction on a single text sample
    
    Args:
        model: Trained model
        tokenizer: Tokenizer
        text: Text to predict
        label_encoder: Optional label encoder for class names
        
    Returns:
        Dictionary with prediction results
    """
    # Download stopwords if needed
    try:
        nltk.download('stopwords', quiet=True)
        STOP_WORDS = set(stopwords.words('english'))
    except:
        # Fallback if NLTK fails
        STOP_WORDS = set()
    
    # Clean text
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'[^A-Za-z0-9\s]+', '', text)  # Keep alphanumeric
    text = ' '.join([word for word in text.lower().split() if word not in STOP_WORDS])
    
    # Dynamic max_length based on text length
    max_length = min(256, max(32, len(text.split()) + 20))
    
    # Tokenize
    inputs = tokenizer(text, padding='max_length', truncation=True, 
                      max_length=max_length, return_tensors="pt")
    
    # Move inputs to device
    inputs = {key: val.to(device) for key, val in inputs.items()}
    
    # Run inference
    with torch.no_grad():
        if isinstance(model, torch.jit.ScriptModule):
            outputs = model(inputs["input_ids"], inputs["attention_mask"])
            logits = outputs[0] if isinstance(outputs, tuple) else outputs
        else:
            outputs = model(**inputs)
            logits = outputs.logits
    
    # Get prediction
    probs = torch.nn.functional.softmax(logits, dim=-1)
    pred_id = torch.argmax(probs, dim=1).item()
    confidence = float(probs[0][pred_id].item())
    
    return {
        "prediction": "SUICIDAL" if pred_id == 1 else "NON-SUICIDAL",
        "confidence": confidence,
        "input_length": len(text.split())
    }

def test_real_world_examples(model, tokenizer):
    """Test the model on real-world examples of varying lengths
    
    Args:
        model: Trained model
        tokenizer: Tokenizer
        
    Returns:
        Dictionary with test results
    """
    print("\nTesting model on real-world examples:")
    
    # Test samples of varying lengths and content
    test_samples = [
        "I'm feeling sad today",  # Very short non-suicidal
        "I want to end my life",  # Short suicidal
        "I've been feeling down lately and can't seem to get motivated. Work is stressful but I'm trying to hang in there.",  # Medium non-suicidal
        "I can't take it anymore. I've been thinking about ending it all. The pain is too much to bear and no one understands what I'm going through.",  # Medium suicidal
        "I had a great day today! I went to the park and met with friends. We had lunch and talked about our future plans. Then I came home and watched a movie.",  # Medium positive
        # Add a longer example with mixed signals
        "I've been struggling a lot lately with work and personal relationships. Sometimes it feels overwhelming and I wonder what's the point of continuing. But then I remember my family and the good moments that make life worth living. It's a constant battle between darkness and light."
    ]
    
    results = []
    
    for sample in test_samples:
        result = predict_text(model, tokenizer, sample)
        print(f"\nText ({result['input_length']} words): {sample[:100]}{'...' if len(sample) > 100 else ''}")
        print(f"Prediction: {result['prediction']} (Confidence: {result['confidence']:.4f})")
        results.append(result)
    
    return results

def run_full_evaluation(X_test, y_test, model_dir=DEFAULT_MODEL_DIR):
    """Run a full evaluation of the model
    
    Args:
        X_test: Test text data
        y_test: Test labels
        model_dir: Directory containing the model
        
    Returns:
        Dictionary with all evaluation results
    """
    # Ensure consistent label types
    print("\nPreparing test data...")
    if isinstance(y_test, pd.Series):
        y_test = y_test.astype(str).str.lower()
    else:
        y_test = np.array([str(label).lower() for label in y_test])
    
    # Print info about test data
    print(f"Test data info: {len(X_test)} samples")
    print(f"Test labels dtype: {y_test.dtype}")
    print(f"Unique label values: {np.unique(y_test)}")
    
    # Load model and label encoder
    print("\nLoading model and label encoder...")
    model, tokenizer, label_encoder = load_model(model_dir)
    
    # Print info about label encoder
    print(f"Label encoder classes: {label_encoder.classes_}")
    
    # Encode test labels to match training format
    print("\nEncoding test labels...")
    try:
        y_test_encoded = label_encoder.transform(y_test)
        print(f"Encoded test labels shape: {y_test_encoded.shape}")
        print(f"Unique encoded values: {np.unique(y_test_encoded)}")
    except ValueError as e:
        print(f"Error encoding labels: {e}")
        print("Original labels:", np.unique(y_test))
        print("Label encoder classes:", label_encoder.classes_)
        raise
    
    # Run evaluations with encoded labels
    print("\nRunning evaluations...")
    basic_metrics = evaluate_model(model, tokenizer, X_test, y_test_encoded)
    length_robustness = test_text_length_robustness(model, tokenizer, X_test, y_test_encoded)
    memory_usage = measure_memory_usage(model, tokenizer)
    real_world_tests = test_real_world_examples(model, tokenizer)
    
    return {
        'basic_metrics': basic_metrics,
        'length_robustness': length_robustness,
        'memory_usage': memory_usage,
        'real_world_tests': real_world_tests
    }

if __name__ == "__main__":
    # This script should be run by main_distilbert.py
    print("This module should be imported by main_distilbert.py rather than run directly.")
    print("Run 'python main_distilbert.py --evaluate' to evaluate the model.") 