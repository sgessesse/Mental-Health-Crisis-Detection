# Mental Health Crisis Detection

![Python](https://img.shields.io/badge/Python-3.9-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104.1-green.svg)
![DistilBERT](https://img.shields.io/badge/DistilBERT-Latest-red.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

An AI-powered web application that analyzes social media text for signs of mental health crisis using NLP and deep learning. The system uses a fine-tuned DistilBERT model optimized for deployment, achieving 98% accuracy while maintaining fast inference times.

**Disclaimer:** This project is developed strictly for educational and research purposes. It is not intended for clinical use, diagnosis, or as a substitute for professional medical advice, diagnosis, or treatment. The outputs of this model should never be used to make decisions regarding suicidal ideation or any mental health condition. Always consult with a qualified mental health professional for any health concerns.


## Features

- 🧠 DistilBERT-based text classification (SUICIDAL/NON-SUICIDAL)
- 🎯 98.1% Accuracy, 98.1% F1 Score
- ⚡ Fast inference (0.12ms per prediction)
- 🚀 Scaled to 4.1 RPS (246 RPM) on 1GB RAM instance
- 🌐 FastAPI backend with CPU optimization
- 📊 Comprehensive model training pipeline
- 🐳 Docker-ready deployment configuration
- 💾 Optimized model size (255MB)
- 📈 Consistent performance across text lengths

## Project Structure

```
Mental-Health-Crisis-Detection/
├── app/                        # Web application
│   ├── static/                 # Frontend files
│   ├── main.py                # FastAPI application
│   ├── requirements.txt       # Deployment dependencies
│   └── Dockerfile            # Container configuration
│
└── model-training/            # Model training pipeline
    ├── preprocess.py         # Data preprocessing
    ├── model.py             # Model architecture
    ├── train.py            # Training script
    ├── evaluate.py         # Evaluation script
    └── requirements.txt    # Training dependencies
```

## Installation

1. **Clone Repository**
```bash
git clone https://github.com/sgessesse/Mental-Health-Crisis-Detection.git
cd Mental-Health-Crisis-Detection
```

2. **Download Dataset**
- Get the dataset from [Kaggle](https://www.kaggle.com/datasets/nikhileswarkomati/suicide-watch/data)
- Place the CSV file in: `data/reddit_mental_health.csv`

3. **Training Setup**
```bash
cd model-training
pip install -r requirements.txt
python -m nltk.downloader stopwords
```

4. **Deployment Setup**
```bash
cd app
pip install -r requirements.txt
```

## Training the Model

1. **Preprocess Data**
```bash
cd model-training
python preprocess.py
```

2. **Train Model**
```bash
python train.py
```

3. **Evaluate Model**
```bash
python evaluate.py
```

### Model Performance

Latest evaluation results:
- Accuracy: 98.09%
- F1 Score (weighted): 98.09%
- Precision (weighted): 98.09%
- Recall (weighted): 98.09%
- Inference Time: 0.12ms per prediction
- Model Size: 255.4MB

The model demonstrates consistently high performance across varying text lengths, from very short (15 words) to long (400+ words) texts. A detailed visualization of the model's performance across different text lengths can be found in `model-training/text_length_performance.png`.

## Running the Web Application

1. **After Training**
- Ensure the trained model is in `app/model/`
- Verify `label_encoder.joblib` is present

2. **Start FastAPI Server**
```bash
cd app
uvicorn main:app --reload
```

3. **Access Web Interface**
- Visit http://localhost:8000 in your browser

## Docker Deployment

1. **Build Image**
```bash
cd app
docker build -t mental-health-detection .
```

2. **Run Container**
```bash
docker run -p 8000:8080 mental-health-detection
```


## Important Notes

- The model and data files are not included in the repository due to size limitations
- You must train the model yourself using the provided training pipeline
- The system is optimized for CPU inference in production
- Different requirements files are maintained for training (GPU) and deployment (CPU)

## Ethical Considerations & Important Disclaimer

**CRITICAL DISCLAIMER:** This project is for **educational and research purposes only**. It is explicitly **NOT intended for clinical use, diagnosis, or as a substitute for professional medical advice, diagnosis, or treatment.** The model's outputs are based on patterns learned from data and should **NEVER** be used to make decisions regarding suicidal ideation or any mental health condition. Always consult with a qualified mental health professional for any health concerns.

- This tool provides preliminary insights only.
- Predictions must **always** be validated by a human expert.
- It is crucial to provide appropriate mental health resources alongside any results.
- Strict user privacy and data protection measures must be maintained.
- This project is not a replacement for professional mental health assessment or intervention.
- The developers and contributors of this project are not liable for any misuse or misinterpretation of the information provided by this tool.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## License

MIT License - see LICENSE for details

## Acknowledgments

- Dataset: Reddit Mental Health Dataset from Kaggle
- Base Model: DistilBERT from Hugging Face
- FastAPI framework