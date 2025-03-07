"""
Mental Health Crisis Detection - DistilBERT Model Training

This module handles the training of an optimized DistilBERT model that is:
1. Smaller in size (<100MB)
2. Efficient in memory usage
3. Robust to variable text lengths
"""
from datasets import Dataset
from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizerFast,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    EarlyStoppingCallback
)
import numpy as np
import torch
import os
import joblib
from sklearn.metrics import recall_score, f1_score, precision_score, accuracy_score
from sklearn.preprocessing import LabelEncoder
import pandas as pd

# Define model paths
DEFAULT_MODEL_DIR = "../app/model"

# Set up GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Print CUDA information if available
if torch.cuda.is_available():
    print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"Available GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print(f"Current GPU memory usage: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    print(f"GPU memory reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")

def get_optimal_batch_size():
    """Determine optimal batch size based on available GPU memory"""
    if not torch.cuda.is_available():
        return 32  # Default CPU batch size
    
    total_memory = torch.cuda.get_device_properties(0).total_memory
    # Use 70% of available memory to be more conservative
    usable_memory = total_memory * 0.70
    
    # More conservative memory estimation
    # DistilBERT base model with sequence length 256 typically uses about 0.6GB for batch size 32
    memory_per_sample = (0.6 * 1024 * 1024 * 1024) / 32  # Convert 0.6GB to bytes and divide by 32
    
    # Calculate maximum batch size
    max_batch_size = int((usable_memory - 3 * 1024 * 1024 * 1024) / memory_per_sample)  # Leave 3GB for other operations
    
    # Round down to nearest multiple of 16 for better GPU utilization
    optimal_batch_size = max(32, (max_batch_size // 16) * 16)
    
    # Cap the batch size at 128 to be safe
    optimal_batch_size = min(optimal_batch_size, 128)
    
    print(f"Calculated optimal batch size: {optimal_batch_size}")
    return optimal_batch_size

def compute_metrics(eval_pred):
    """Comprehensive evaluation metrics"""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)  # Convert logits to class IDs
    
    # Calculate multiple metrics
    recall = recall_score(labels, predictions, average='binary')
    precision = precision_score(labels, predictions, average='binary')
    f1 = f1_score(labels, predictions, average='binary')
    accuracy = accuracy_score(labels, predictions)
    
    return {
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def train_distilbert_model(X_train, y_train, output_dir=DEFAULT_MODEL_DIR):
    """Train a DistilBERT model optimized for size and performance
    
    Args:
        X_train: Training text samples (pandas Series or numpy array)
        y_train: Training labels (pandas Series or numpy array)
        output_dir: Directory to save the model
        
    Returns:
        model: Trained DistilBERT model
        tokenizer: Trained tokenizer
        label_encoder: Class label encoder
    """
    print("Initializing DistilBERT training...")
    
    # Clear GPU cache before starting
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # 1. Prepare tokenizer
    print("Loading DistilBERT tokenizer...")
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
    
    # 2. Ensure labels are strings and convert to lowercase for consistency
    print("Preparing labels...")
    if isinstance(y_train, pd.Series):
        y_train = y_train.astype(str).str.lower()
    else:
        y_train = np.array([str(label).lower() for label in y_train])
    
    print("Label value counts:")
    if isinstance(y_train, pd.Series):
        print(y_train.value_counts())
    else:
        unique, counts = np.unique(y_train, return_counts=True)
        print(dict(zip(unique, counts)))
    
    # 3. Encode labels
    print("Encoding labels...")
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    
    print(f"Label encoder classes: {label_encoder.classes_}")
    print(f"Encoded label values: {np.unique(y_train_encoded)}")
    
    # 4. Create Hugging Face dataset
    print("Creating dataset...")
    train_texts = X_train.tolist() if hasattr(X_train, 'tolist') else X_train
    train_labels = y_train_encoded.tolist() if hasattr(y_train_encoded, 'tolist') else y_train_encoded
    
    dataset = Dataset.from_dict({
        'text': train_texts,
        'labels': train_labels
    })
    
    # 5. Define tokenize function with dynamic handling
    def tokenize_function(examples):
        """Convert text to DistilBERT-compatible tokens with dynamic padding"""
        return tokenizer(
            examples['text'], 
            truncation=True,
            padding=False,  # No padding here - will be done dynamically during training
            max_length=256  # Reduce from 512 to 256 to handle memory better
        )
    
    # 6. Tokenize dataset
    print("Tokenizing dataset...")
    tokenized_dataset = dataset.map(tokenize_function, batched=True, num_proc=4)
    
    # 7. Initialize model
    print("Initializing DistilBERT model...")
    model = DistilBertForSequenceClassification.from_pretrained(
        'distilbert-base-uncased',
        num_labels=len(label_encoder.classes_)
    ).to(device)  # Move model to GPU
    
    # 8. Configure model architecture
    print("Configuring model architecture...")
    model.config.hidden_dropout_prob = 0.2  # Reduce overfitting
    model.config.attention_probs_dropout_prob = 0.1  # Sparse attention
    model.config.torchscript = True  # Enable TorchScript optimization
    
    # 9. Setup training arguments - with optimized GPU configuration
    print("Setting up training arguments...")
    
    # Get optimal batch size
    batch_size = get_optimal_batch_size()
    print(f"Using batch size: {batch_size}")
    
    training_args = TrainingArguments(
        output_dir='./results',
        eval_strategy='epoch',
        save_strategy='epoch',
        learning_rate=5e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=5,
        weight_decay=0.01,
        logging_dir='./logs',
        fp16=torch.cuda.is_available(),  # Use mixed precision training
        gradient_accumulation_steps=1,
        load_best_model_at_end=True,
        metric_for_best_model='f1',
        save_total_limit=1,  # Reduce from 2 to 1 to save memory
        dataloader_num_workers=2,  # Reduce from 4 to 2
        dataloader_pin_memory=True if torch.cuda.is_available() else False,
        logging_steps=50,
        # Speed optimizations
        dataloader_drop_last=True,
        group_by_length=True,
        remove_unused_columns=True,
        # Memory optimizations
        gradient_checkpointing=True,  # Re-enable for memory efficiency
        max_grad_norm=1.0,  # Re-enable for stability
        optim='adamw_torch',  # More memory efficient optimizer
    )
    
    # 10. Split dataset for train/validation
    print("Splitting dataset into train/validation sets...")
    train_size = int(0.9 * len(tokenized_dataset))
    train_dataset = tokenized_dataset.shuffle(seed=42).select(range(train_size))
    eval_dataset = tokenized_dataset.shuffle(seed=42).select(range(train_size, len(tokenized_dataset)))
    
    # 11. Create data collator with dynamic padding
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding=True)
    
    # 12. Setup early stopping callback
    early_stopping_callback = EarlyStoppingCallback(early_stopping_patience=3)
    
    # 13. Initialize trainer
    print("Initializing trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[early_stopping_callback],
    )
    
    # 14. Train model
    print("Starting model training...")
    trainer.train()
    
    # 15. Evaluate model
    print("Evaluating model...")
    results = trainer.evaluate()
    print(f"Evaluation results: {results}")
    
    return model, tokenizer, label_encoder

def save_optimized_model(model, tokenizer, label_encoder, output_dir=DEFAULT_MODEL_DIR):
    """Save the model in an optimized format for deployment
    
    Saves:
    1. DistilBERT model
    2. Tokenizer
    3. Label encoder
    4. TorchScript optimized model
    
    Args:
        model: Trained DistilBERT model
        tokenizer: DistilBERT tokenizer
        label_encoder: Label encoder used for training
        output_dir: Directory to save the model
    """
    print(f"Saving model to {output_dir}...")
    
    # Move model to CPU for saving
    model = model.cpu()
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Save model and tokenizer
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # 2. Save label encoder
    joblib.dump(label_encoder, os.path.join(output_dir, "label_encoder.joblib"))
    
    # 3. Create TorchScript version for faster inference
    print("Creating TorchScript optimized model...")
    
    # Prepare model for tracing
    model.eval()
    
    # Create a dummy input for tracing
    dummy_input = tokenizer("This is a test", return_tensors="pt")
    
    # Trace the model
    try:
        traced_model = torch.jit.trace(
            model, 
            (dummy_input["input_ids"], dummy_input["attention_mask"])
        )
        
        # Save the traced model
        torch.jit.save(traced_model, os.path.join(output_dir, "model_optimized.pt"))
        print("TorchScript model created successfully")
    except Exception as e:
        print(f"Error creating TorchScript model: {e}")
        print("Continuing without TorchScript optimization")
    
    # 4. Calculate and report model size
    model_size_mb = sum(p.element_size() * p.nelement() for p in model.parameters()) / (1024 * 1024)
    print(f"Model size: {model_size_mb:.2f} MB")
    
    print(f"Model saved successfully to {output_dir}")
    return output_dir

if __name__ == "__main__":
    # Test module if run directly
    print("This module should be imported by main_distilbert.py rather than run directly.")
    print("Run 'python main_distilbert.py' to train and save the model.") 