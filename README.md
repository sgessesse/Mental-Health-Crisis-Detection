# Mental-Health-Crisis-Detection

![Live Demo](https://img.shields.io/badge/Demo-Live%20Demo-brightgreen)](http://mental-health-env.eba-3mrbnqp2.us-east-2.elasticbeanstalk.com/)

An AI-powered web application that analyzes social media text for signs of mental health crisis using NLP and deep learning.

## Features

- 🧠 BERT-based text classification (SUICIDAL/NON-SUICIDAL)
- 🌐 Web interface for real-time analysis
- ⚡ FastAPI backend with CPU optimization
- 📊 Model training notebooks with 98%+ accuracy
- 🐳 Docker-ready deployment configuration
- ☁️ AWS Elastic Beanstalk deployment support

## Live Demo

Access the deployed web application:  
[http://mental-health-env.eba-3mrbnqp2.us-east-2.elasticbeanstalk.com/](http://mental-health-env.eba-3mrbnqp2.us-east-2.elasticbeanstalk.com/)

## Project Structure
Mental-Health-Crisis-Detection/
├── app/ # FastAPI application
│ ├── static/ # Frontend assets
│ └── main.py # API endpoints
│ └── application.py # API entry point
├── notebooks/ # Jupyter notebooks
│ ├── preprocess.ipynb # Data cleaning
│ ├── model.ipynb # Model training
│ ├── evaluate.ipynb # Performance metrics
│ └── main.ipynb # End-to-end pipeline
├── requirements.txt # Python dependencies
└── README.md # Project documentation

## Installation

1. **Clone Repository**

git clone https://github.com/yourusername/mental-health-detection.git
cd mental-health-detection

2. **Install Dependencies**

pip install -r requirements.txt
python -m nltk.downloader stopwords

3. **Download Dataset**

Get the dataset from Kaggle:
https://www.kaggle.com/datasets/nikhileswarkomati/suicide-watch/data

Place the CSV file in:
notebooks/data/reddit_mental_health.csv

## Training the Model

1. **Preprocess Data**

Run preprocess.ipynb to clean and prepare the dataset

2. **Model Training**

Execute model.ipynb to:

- Fine-tune BERT-base model

- Save trained model to app/model/

- Generate label encoder

3. **Evaluation**

Use evaluate.ipynb to validate model performance

run main.ipynb to run all the three processes

## Running the Web Application

1. **After Training**

Ensure you have:

- app/model/ folder with trained model

- label_encoder.joblib file

2. **Start FastAPI Server**

cd app
uvicorn main:app --reload

3. **Access Web Interface**

Visit http://localhost:8000 in your browser

## Deployment

The system is configured for Docker deployment. To deploy on AWS Elastic Beanstalk:

1. Build Docker image

2. Configure EB CLI

3. Deploy using Dockerfile configuration

## Contributing

1. Fork the repository

2. Create feature branch

3. Submit a pull request

## Ethical Considerations

- Predictions should never be used without human validation

- Always provide mental health resources with results

- Maintain strict user anonymity

## Troubleshooting

- **Dataset Not Found**: Verify CSV file path in notebooks

- **Model Training Issues**: Ensure GPU support for PyTorch

- **Dependency Conflicts**: Use exact versions from requirements.txt

## License

MIT License - see LICENSE for details