"""
Mental Health API - Step 3: Model Inference and UI
Adding real model inference and user interface
"""
import logging
import os
import numpy as np
import joblib
import torch
from fastapi import FastAPI, HTTPException, Request, Form
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from typing import Optional
import time
from pydantic import BaseModel

# Import model specific libraries
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(title="Mental Health API - Step 3: Model Inference and UI")

# Mount static directory
app.mount("/static", StaticFiles(directory="static"), name="static")

# Define models for API
class PredictionRequest(BaseModel):
    text: str
    include_details: Optional[bool] = False

class PredictionResponse(BaseModel):
    prediction: str
    confidence: float
    processing_time: float
    details: Optional[dict] = None

# Global variables for model and tokenizer
MODEL_PATH = "model"
tokenizer = None
model = None
label_encoder = None
model_loaded = False
model_error = None

# Load label encoder
try:
    logger.info("Loading label encoder...")
    label_encoder_path = os.path.join(MODEL_PATH, "label_encoder.joblib")
    if os.path.exists(label_encoder_path):
        label_encoder = joblib.load(label_encoder_path)
        logger.info(f"Label encoder loaded with classes: {label_encoder.classes_}")
    else:
        logger.warning("Label encoder file not found, using default classes")
        from sklearn.preprocessing import LabelEncoder
        label_encoder = LabelEncoder()
        label_encoder.classes_ = np.array(['NON-SUICIDAL', 'SUICIDAL'])
        # Save the default label encoder for future use
        try:
            os.makedirs(MODEL_PATH, exist_ok=True)
            joblib.dump(label_encoder, label_encoder_path)
            logger.info(f"Created and saved default label encoder to {label_encoder_path}")
        except Exception as e:
            logger.warning(f"Could not save default label encoder: {str(e)}")
except Exception as e:
    logger.error(f"Error loading label encoder: {str(e)}")
    model_error = f"Label encoder error: {str(e)}"
    # Create a fallback label encoder
    from sklearn.preprocessing import LabelEncoder
    label_encoder = LabelEncoder()
    label_encoder.classes_ = np.array(['NON-SUICIDAL', 'SUICIDAL'])
    logger.info("Created fallback label encoder with default classes")

# Function to load model
def load_model():
    global tokenizer, model, model_loaded, model_error
    
    try:
        logger.info(f"Loading model and tokenizer from {MODEL_PATH}...")
        logger.info(f"Current directory: {os.getcwd()}")
        logger.info(f"MODEL_PATH directory contents: {os.listdir(MODEL_PATH) if os.path.exists(MODEL_PATH) else 'Not found'}")
        
        # Load tokenizer
        tokenizer_path = MODEL_PATH
        logger.info(f"Loading tokenizer from {tokenizer_path}")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        logger.info(f"Tokenizer loaded: {tokenizer.__class__.__name__}")
        
        # Load model
        model_path = MODEL_PATH
        logger.info(f"Loading model from {model_path}")
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        model.eval()  # Set model to evaluation mode
        logger.info(f"Model loaded: {model.__class__.__name__}, Number of parameters: {sum(p.numel() for p in model.parameters())}")
        
        # Log model configuration
        if hasattr(model, 'config'):
            logger.info(f"Model config: {model.config}")
        
        model_loaded = True
        return True
    except Exception as e:
        import traceback
        logger.error(f"Error loading model: {str(e)}")
        logger.error(traceback.format_exc())
        model_error = f"Model loading error: {str(e)}"
        model_loaded = False
        return False

# Load model on startup
@app.on_event("startup")
async def startup_event():
    load_model()

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the HTML UI"""
    return FileResponse("static/index.html")

@app.get("/api/info")
async def info():
    """API information endpoint"""
    return {
        "api_name": "Mental Health Detection API",
        "version": "3.0",
        "model_loaded": model_loaded,
        "model_error": model_error,
        "numpy_version": np.__version__,
        "joblib_version": joblib.__version__,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "available_endpoints": [
            {"path": "/", "method": "GET", "description": "HTML UI"},
            {"path": "/api/info", "method": "GET", "description": "API information"},
            {"path": "/health", "method": "GET", "description": "Health check"},
            {"path": "/predict", "method": "POST", "description": "Make a prediction"},
            {"path": "/reload-model", "method": "POST", "description": "Reload the model (admin only)"}
        ]
    }

@app.get("/health")
async def health():
    """Health check endpoint"""
    # Get memory usage
    mem_info = os.popen('free -m').readlines() if os.name != 'nt' else ["100", "200", "300"]
    try:
        available_memory = int(mem_info[1].split()[6]) if os.name != 'nt' else 1000
    except:
        available_memory = 0
        
    status = "ready" if model_loaded else "error"
    status_code = 200 if model_loaded else 500
    
    response = {
        "status": status,
        "model_loaded": model_loaded,
        "model_error": model_error,
        "available_memory_mb": available_memory,
        "cuda_available": torch.cuda.is_available()
    }
    
    return JSONResponse(content=response, status_code=status_code)

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Real prediction endpoint using the loaded model"""
    if not model_loaded:
        raise HTTPException(status_code=503, 
                           detail=f"Model not loaded. Error: {model_error}")
    
    # Check if label encoder is available
    if label_encoder is None:
        logger.error("Label encoder is not available")
        raise HTTPException(status_code=503,
                           detail="Label encoder is not available. Please reload the model.")
    
    start_time = time.time()
    
    try:
        text = request.text
        include_details = request.include_details
        
        # Log the received text (truncated for privacy)
        truncated_text = text[:30] + "..." if len(text) > 30 else text
        logger.info(f"Processing request. Text: {truncated_text}")
        
        # Tokenize the input text
        inputs = tokenizer(text, return_tensors="pt", 
                          truncation=True, max_length=512, 
                          padding="max_length")
        
        # Make prediction
        with torch.no_grad():
            outputs = model(**inputs)
            
            # Handle different output formats
            # If outputs is a tuple, the first element typically contains the logits
            if isinstance(outputs, tuple):
                logits = outputs[0]
            # If outputs has a logits attribute (e.g., transformers.modeling_outputs.SequenceClassifierOutput)
            elif hasattr(outputs, 'logits'):
                logits = outputs.logits
            else:
                # If we can't determine the format, try to use outputs directly
                logits = outputs
                
            probabilities = torch.softmax(logits, dim=1)
            confidence, prediction_idx = torch.max(probabilities, dim=1)
        
        # Convert prediction index to class name
        prediction_idx_value = prediction_idx.item()
        logger.info(f"Raw prediction index: {prediction_idx_value}")
        
        # Ensure the index is within bounds of label_encoder.classes_
        if prediction_idx_value < 0 or prediction_idx_value >= len(label_encoder.classes_):
            logger.warning(f"Prediction index {prediction_idx_value} out of bounds for classes {label_encoder.classes_}")
            prediction = "UNKNOWN"
        else:
            prediction = label_encoder.classes_[prediction_idx_value]
            
        confidence_value = confidence.item()
        logger.info(f"Prediction: {prediction}, Confidence: {confidence_value}")
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Prepare response
        response = {
            "prediction": prediction,
            "confidence": confidence_value,
            "processing_time": processing_time
        }
        
        # Add details if requested
        if include_details:
            # Get all class probabilities
            all_probs = probabilities[0].tolist()
            
            # Ensure we have the right number of probabilities
            if len(all_probs) == len(label_encoder.classes_):
                class_probs = {cls: prob for cls, prob in zip(label_encoder.classes_, all_probs)}
            else:
                logger.warning(f"Number of probabilities ({len(all_probs)}) doesn't match number of classes ({len(label_encoder.classes_)})")
                class_probs = {f"class_{i}": prob for i, prob in enumerate(all_probs)}
            
            # Add details to response
            response["details"] = {
                "class_probabilities": class_probs,
                "input_length": len(text),
                "token_count": inputs.input_ids.shape[1],
                "truncated": len(text) > 512,
                "model_type": "DistilBERT for Sequence Classification",
                "raw_prediction_index": prediction_idx_value
            }
        
        return response
    
    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}")
        raise HTTPException(status_code=500, 
                           detail=f"Error processing request: {str(e)}")

@app.post("/reload-model")
async def reload_model(request: Request):
    """Admin endpoint to reload the model"""
    try:
        # In a real application, you would add authentication here
        success = load_model()
        if success:
            return {"status": "success", "message": "Model reloaded successfully"}
        else:
            raise HTTPException(status_code=500, 
                               detail=f"Failed to reload model: {model_error}")
    except Exception as e:
        logger.error(f"Error reloading model: {str(e)}")
        raise HTTPException(status_code=500, 
                           detail=f"Error reloading model: {str(e)}")

# Fallback route for 404 errors
@app.exception_handler(404)
async def custom_404_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={"error": "The requested resource was not found",
                "available_endpoints": ["/", "/api/info", "/health", "/predict"]}
    )

# Fallback for 500 errors
@app.exception_handler(500)
async def custom_500_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"error": f"Internal server error: {str(exc)}",
                "model_loaded": model_loaded,
                "model_error": model_error}
    )

# Make sure transformers logging is set to warning only
import transformers
transformers.logging.set_verbosity_warning()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080) 