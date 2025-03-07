"""
Mental Health Crisis Detection - Main Training Script

This script orchestrates the complete process of:
1. Data preprocessing
2. Model training
3. Model evaluation
4. Model export for deployment

Usage:
    python main_distilbert.py [--train] [--evaluate] [--data_path PATH] [--output_dir DIR]
"""
import argparse
import os
import time
import json
import sys
import torch
import pandas as pd
import numpy as np
from preprocess import prepare_data, augment_data_lengths
from model import train_distilbert_model, save_optimized_model
from evaluate import run_full_evaluation

# Default paths
DEFAULT_DATA_PATH = "../data/reddit_mental_health.csv"
DEFAULT_MODEL_DIR = "../app/model"

def check_system_info():
    """Check system information for GPU and memory"""
    print("\n" + "="*80)
    print("SYSTEM INFORMATION")
    print("="*80)
    
    # Check for CUDA
    if torch.cuda.is_available():
        print(f"GPU detected: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
        print(f"PyTorch CUDA: {torch.version.cuda}")
        
        # Get VRAM info
        gpu_props = torch.cuda.get_device_properties(0)
        print(f"GPU memory: {gpu_props.total_memory / 1e9:.2f} GB")
        
        # Current memory usage
        print(f"Current GPU memory allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
        print(f"Current GPU memory reserved: {torch.cuda.memory_reserved(0) / 1e9:.2f} GB")
    else:
        print("No GPU detected. Training will run on CPU (much slower).")
    
    # CPU information
    import psutil
    print(f"\nCPU cores: {psutil.cpu_count(logical=False)} physical, {psutil.cpu_count()} logical")
    print(f"Available memory: {psutil.virtual_memory().available / 1e9:.2f} GB / {psutil.virtual_memory().total / 1e9:.2f} GB")
    
    print("="*80 + "\n")

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Mental Health Crisis Detection - DistilBERT Training")
    
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate the model")
    parser.add_argument("--data_path", type=str, default=DEFAULT_DATA_PATH, 
                        help=f"Path to the data CSV file (default: {DEFAULT_DATA_PATH})")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_MODEL_DIR,
                        help=f"Directory to save the model (default: {DEFAULT_MODEL_DIR})")
    parser.add_argument("--cpu_only", action="store_true", help="Force CPU usage even if GPU is available")
    
    return parser.parse_args()

def train_pipeline(data_path, output_dir):
    """Run the complete training pipeline"""
    total_start_time = time.time()
    print("\n" + "="*80)
    print(f"TRAINING PIPELINE - {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    # 1. Prepare data
    print("\n--- Data Preparation ---")
    data_start_time = time.time()
    X_train, X_test, y_train, y_test = prepare_data(data_path)
    print(f"Data preparation completed in {time.time() - data_start_time:.2f} seconds")
    
    # Verify and convert label types
    print("\nVerifying label types:")
    print(f"y_train type: {type(y_train)}, dtype: {y_train.dtype}")
    print(f"y_test type: {type(y_test)}, dtype: {y_test.dtype}")
    
    # Ensure labels are strings and lowercase
    if isinstance(y_train, pd.Series):
        y_train = y_train.astype(str).str.lower()
        y_test = y_test.astype(str).str.lower()
    else:
        y_train = np.array([str(label).lower() for label in y_train])
        y_test = np.array([str(label).lower() for label in y_test])
    
    print("\nAfter conversion:")
    print(f"Training labels: {np.unique(y_train)}")
    print(f"Test labels: {np.unique(y_test)}")
    
    # 2. Augment data with varied text lengths
    print("\n--- Data Augmentation ---")
    aug_start_time = time.time()
    X_augmented, y_augmented = augment_data_lengths(X_train, y_train)
    print(f"Data augmentation completed in {time.time() - aug_start_time:.2f} seconds")
    
    # Verify augmented labels
    print("\nVerifying augmented labels:")
    print(f"y_augmented type: {type(y_augmented)}, dtype: {y_augmented.dtype}")
    print(f"Unique augmented values: {np.unique(y_augmented)}")
    
    # 3. Train model
    print("\n--- Model Training ---")
    training_start_time = time.time()
    model, tokenizer, label_encoder = train_distilbert_model(X_augmented, y_augmented)
    print(f"Model training completed in {(time.time() - training_start_time) / 60:.2f} minutes")
    
    # Save label encoder classes for verification
    print("\nSaved label encoder classes:", label_encoder.classes_)
    
    # 4. Save model
    print("\n--- Model Export ---")
    export_start_time = time.time()
    model_path = save_optimized_model(model, tokenizer, label_encoder, output_dir)
    print(f"Model export completed in {time.time() - export_start_time:.2f} seconds")
    
    total_time = time.time() - total_start_time
    
    print("\n" + "="*80)
    print(f"TRAINING COMPLETE - Model saved to {model_path}")
    print(f"Total pipeline time: {total_time / 60:.2f} minutes")
    print("="*80 + "\n")
    
    # Return test data for potential immediate evaluation
    return model_path, X_test, y_test

def evaluate_pipeline(data_path, model_dir, X_test=None, y_test=None):
    """Run the evaluation pipeline"""
    eval_start_time = time.time()
    print("\n" + "="*80)
    print(f"EVALUATION PIPELINE - {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    # 1. Prepare test data if not provided
    if X_test is None or y_test is None:
        print("\n--- Loading Test Data ---")
        _, X_test, _, y_test = prepare_data(data_path)
    
    # Ensure consistent label types
    if isinstance(y_test, pd.Series):
        y_test = y_test.astype(str).str.lower()
    else:
        y_test = np.array([str(label).lower() for label in y_test])
    
    print(f"\nTest data prepared: {len(X_test)} samples")
    print(f"Test labels dtype: {y_test.dtype}")
    print(f"Unique test labels: {np.unique(y_test)}")
    
    # 2. Run evaluation
    print("\n--- Running Evaluations ---")
    try:
        results = run_full_evaluation(X_test, y_test, model_dir)
        
        # 3. Save results
        results_file = os.path.join(os.path.dirname(model_dir), "evaluation_results.json")
        
        # Convert results to serializable format
        serializable_results = {
            'accuracy': float(results['basic_metrics']['accuracy']),
            'f1_weighted': float(results['basic_metrics']['f1_weighted']),
            'precision_weighted': float(results['basic_metrics']['precision_weighted']),
            'recall_weighted': float(results['basic_metrics']['recall_weighted']),
            'avg_inference_time_ms': float(results['basic_metrics']['avg_inference_time_ms']),
            'model_size_mb': float(results['memory_usage'].get('model_size_mb', 0)),
            'evaluation_time_seconds': float(time.time() - eval_start_time),
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        with open(results_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"\nEvaluation results saved to {results_file}")
        
    except Exception as e:
        print(f"\nError during evaluation: {str(e)}")
        print("Traceback:")
        import traceback
        traceback.print_exc()
        return None
    
    print("\n" + "="*80)
    print(f"EVALUATION COMPLETE - Time taken: {(time.time() - eval_start_time) / 60:.2f} minutes")
    print("="*80 + "\n")
    
    return results

def main():
    """Main function"""
    start_time = time.time()
    args = parse_args()
    
    # Check for valid arguments
    if not (args.train or args.evaluate):
        print("Please specify at least one action: --train or --evaluate")
        return
    
    # Check system information
    check_system_info()
    
    # Force CPU if requested
    if args.cpu_only and torch.cuda.is_available():
        print("Forcing CPU usage as requested")
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    
    # Check if data file exists
    if not os.path.exists(args.data_path):
        print(f"Error: Data file not found at {args.data_path}")
        print("Please provide a valid path to the data CSV file with --data_path")
        return
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output_dir), exist_ok=True)
    
    # Execute requested actions
    X_test = None
    y_test = None
    
    if args.train:
        model_path, X_test, y_test = train_pipeline(args.data_path, args.output_dir)
    
    if args.evaluate:
        # Ensure model exists if only evaluating
        if not args.train and not os.path.exists(args.output_dir):
            print(f"Error: Model directory not found at {args.output_dir}")
            print("Please train the model first or specify a valid model directory with --output_dir")
            return
        
        # Use the test data from training if available
        evaluate_pipeline(args.data_path, args.output_dir, X_test, y_test)
    
    total_time = time.time() - start_time
    print(f"All operations completed successfully! Total time: {total_time / 60:.2f} minutes")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProcess interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1) 