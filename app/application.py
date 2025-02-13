"""
FastAPI Application Entry Point (Simplified Instance)

This minimal setup allows Elastic Beanstalk/AWS deployment systems 
to recognize and deploy the application. The main logic resides in main.py.
"""

from fastapi import FastAPI

# Primary application instance - AWS deployment systems often look for 
# an 'application' object in this file
app = FastAPI()

@app.get("/health")
def health_check():
    """Basic health check endpoint for load balancer validation"""
    return {"status": "OK"}