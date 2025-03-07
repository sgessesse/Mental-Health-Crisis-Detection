"""
Mental Health Crisis Detection API - Optimized Version

FastAPI web service for real-time analysis of text content using a 
fine-tuned DistilBERT model. Designed for Docker deployment on AWS Elastic Beanstalk.

Key Features:
- Async model loading with thread synchronization
- CPU-optimized inference with TorchScript
- Memory-efficient text processing
- Dynamic text length handling
- NLTK text preprocessing pipeline
- React-style frontend integration
"""
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import torch
from transformers import DistilBertTokenizerFast
import nltk
from nltk.corpus import stopwords
import numpy as np
import re
import threading
import os
import gc
import joblib

# --- Global Configuration ---
model = None  # Will hold the TorchScript model
tokenizer = None  # DistilBERT tokenizer instance
label_encoder = None  # For class labels
model_loaded_event = threading.Event()  # Threading event for model load status

# Configure NLTK for Docker environments
nltk.data.path.append('/usr/share/nltk_data')  # Shared volume in Docker
nltk.download('stopwords')  # Ensure stopwords are available
STOP_WORDS = set(stopwords.words('english'))  # English stopwords set

# --- FastAPI Setup ---
app = FastAPI(title="Mental Health Detection API")
app.mount("/static", StaticFiles(directory="static"), name="static")

def load_model_async():
    """
    Asynchronously load optimized DistilBERT model from disk
    
    Uses threading to prevent blocking server startup
    Loads the TorchScript version for faster inference
    Forces CPU inference for compatibility with EB deployment
    """
    global model, tokenizer, label_encoder
    try:
        # Memory optimization: explicitly set mode and device
        torch.set_grad_enabled(False)  # Disable gradient calculation globally
        
        # Load TorchScript model if available, otherwise load standard model
        model_path = os.path.join('model', 'model_optimized.pt')
        if os.path.exists(model_path):
            model = torch.jit.load(model_path, map_location='cpu')
            print("Loaded optimized TorchScript model")
        else:
            # Fallback to regular model
            from transformers import DistilBertForSequenceClassification
            model = DistilBertForSequenceClassification.from_pretrained(
                'model', 
                local_files_only=True,
                torchscript=True  # Prepare for TorchScript conversion
            ).to('cpu')
            print("Loaded standard model")

        # Force model to evaluation mode
        if hasattr(model, 'eval'):
            model.eval()
        
        # Load tokenizer and label encoder
        tokenizer = DistilBertTokenizerFast.from_pretrained('model')
        label_encoder = joblib.load(os.path.join('model', 'label_encoder.joblib'))
        
        # Run garbage collection after loading
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # Signal completion to other threads
        model_loaded_event.set()
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Model loading failed: {e}")
        raise  # Critical failure - crash the app

@app.on_event("startup")
def startup_event():
    """
    Startup handler - initiates async model loading
    
    Runs in separate thread to maintain API responsiveness
    during potentially slow model initialization
    """
    thread = threading.Thread(target=load_model_async)
    thread.start()

class TextRequest(BaseModel):
    """Request model for prediction endpoint"""
    text: str  # Raw input text to analyze

def preprocess_input(text: str) -> str:
    """
    Replicate notebook preprocessing pipeline
    
    1. Remove URLs and special characters
    2. Convert to lowercase
    3. Remove stopwords
    
    Returns:
        str: Cleaned text ready for tokenization
    """
    # URL removal (common in social media posts)
    text = re.sub(r'http\S+', '', text)

    # Retain only alphanumeric + whitespace
    text = re.sub(r'[^A-Za-z0-9\s]+', '', text)

    # Lowercase + stopword removal (matches training preprocessing)
    text = ' '.join([word for word in text.lower().split() if word not in STOP_WORDS])
    return text

@app.post("/predict")
async def predict(request: TextRequest):
    """
    Prediction endpoint for text analysis
    
    Steps:
    1. Validate model readiness
    2. Clean input text
    3. Tokenize for DistilBERT
    4. Run optimized model inference
    5. Format probabilities
    
    Returns:
        JSON: Prediction and confidence score
    """
    if not model_loaded_event.is_set():
        raise HTTPException(
            status_code=503,
            detail="Model is still loading. Try again later."
        )
    
    try:
        # Replicate training preprocessing
        cleaned_text = preprocess_input(request.text)
        
        # Memory-optimized tokenization - use shorter max_length (256) for shorter texts
        max_length = min(256, max(32, len(cleaned_text.split()) + 20))  # Dynamic sizing
        
        # DistilBERT tokenization with optimization
        inputs = tokenizer(
            cleaned_text,
            truncation=True,  # Enforce max token limit
            padding='max_length',  # Pad to max length
            max_length=max_length,  # Dynamic max length
            return_tensors='pt'  # PyTorch tensors
        )
        
        # Memory-efficient inference
        with torch.no_grad():
            if isinstance(model, torch.jit.ScriptModule):
                # TorchScript model expects separate tensors
                outputs = model(inputs["input_ids"], inputs["attention_mask"])
                # TorchScript may return a tuple or single tensor depending on how it was traced
                logits = outputs[0] if isinstance(outputs, tuple) else outputs
            else:
                # Regular model expects a dictionary
                outputs = model(**inputs)
                logits = outputs.logits
        
        # Convert logits to probabilities (memory-efficient)
        probs = torch.nn.functional.softmax(logits, dim=-1)
        
        # Get prediction
        prediction_id = int(torch.argmax(probs, dim=1).item())
        confidence = float(probs[0][prediction_id].item())
        
        # Convert to label
        prediction_label = "SUICIDAL" if prediction_id == 1 else "NON-SUICIDAL"
        
        # Clean up memory
        del inputs, outputs, logits, probs
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        gc.collect()
        
        return {
            "prediction": prediction_label,
            "confidence": confidence,
            "input_length": len(cleaned_text.split()),  # Return input length for transparency
            "max_length_used": max_length  # Return max length used
        }
    except Exception as e:
        print(f"Error during prediction: {e}")
        return {"error": f"Error analyzing text: {str(e)}"}

@app.get("/health")
def health_check():
    """Dynamic health check reflecting model status"""
    return JSONResponse(
        content={"status": "ready" if model_loaded_event.is_set() else "loading"},
        status_code=200 if model_loaded_event.is_set() else 503
    )

@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Serve frontend interface"""
    with open("static/index.html") as f:
        return HTMLResponse(content=f.read(), status_code=200)