"""
Mental Health Crisis Detection API

FastAPI web service for real-time analysis of text content using a 
fine-tuned BERT model. Designed for Docker deployment on AWS Elastic Beanstalk.

Key Features:
- Async model loading with thread synchronization
- CPU-optimized inference
- NLTK text preprocessing pipeline
- Dynamic health checks
- React-style frontend integration
"""
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import torch
from transformers import BertForSequenceClassification, BertTokenizerFast
from nltk.corpus import stopwords
import nltk
import numpy as np
import re
import threading

# --- Global Configuration ---
model = None # Will hold the loaded BERT model
tokenizer = None # BERT tokenizer instance
model_loaded_event = threading.Event()  # Threading event for model load status

# Configure NLTK for Docker environments
nltk.data.path.append('/usr/share/nltk_data') # Shared volume in Docker
nltk.download('stopwords') # Ensure stopwords are available
STOP_WORDS = set(stopwords.words('english')) # English stopwords set

# --- FastAPI Setup ---
app = FastAPI(title="Mental Health Detection API")
app.mount("/static", StaticFiles(directory="static"), name="static")

def load_model_async():
    """
    Asynchronously load BERT model and tokenizer from disk
    
    Uses threading to prevent blocking server startup
    Forces CPU inference for compatibility with EB deployment
    """
    global model, tokenizer
    try:
        # Load model artifacts from 'model' directory
        model = BertForSequenceClassification.from_pretrained(
            'model', 
            local_files_only=True  # Ensure local files only
        ).to('cpu')  # Explicit CPU allocation
        
        tokenizer = BertTokenizerFast.from_pretrained('model')
        
        # Signal completion to other threads
        model_loaded_event.set()  
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Model loading failed: {e}")
        raise # Critical failure - crash the app

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
    text: str # Raw input text to analyze

def preprocess_input(text: str) -> str:
    """
    Replicate notebook preprocessing pipeline
    
    1. Remove URLs and special characters
    2. Convert to lowercase
    3. Remove stopwords
    
    Returns:
        str: Cleaned text ready for BERT tokenization
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
    3. Tokenize for BERT
    4. Run model inference
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

        # BERT tokenization with same params as training
        inputs = tokenizer(
            cleaned_text,
            truncation=True,  # Enforce 512 token limit
            padding='max_length',  # Pad to max length
            max_length=512,  # Match model architecture
            return_tensors='pt'  # PyTorch tensors
        )
        
        # Disable gradient calculation for inference
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Convert logits to probabilities
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        prediction = np.argmax(probs.numpy())
        
        return {
            "prediction": "SUICIDAL" if prediction == 1 else "NON-SUICIDAL",
            "confidence": float(probs[0][prediction]) # Convert torch tensor
        }
    except Exception as e:
        print(f"Error during prediction: {e}")
        return {"error": "Error analyzing text."}

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