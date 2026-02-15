"""FastAPI service exposing model predictions."""

from __future__ import annotations

import time
import joblib
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from typing import Dict, Any

from src.api.pydantic_models import PredictionResponse, Transaction, HealthResponse, MetricsResponse
from src.data_processing import engineer_features
from src.logger import setup_logger
from src.monitoring import monitor
from src.validators import validate_model_input
import pandas as pd

logger = setup_logger(__name__)

MODEL_PATH = Path("main/model.pkl")
MODEL_VERSION = "1.0.0"

app = FastAPI(
    title="Credit Risk Scoring API",
    version="1.0.0",
    description="Production-grade API for credit risk prediction",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event() -> None:
    """Load model on startup."""
    try:
        if not MODEL_PATH.exists():
            raise RuntimeError("Model artifact not found. Train the model first.")
        app.state.model = joblib.load(MODEL_PATH)
        app.state.model_version = MODEL_VERSION
        logger.info(f"Model loaded successfully (version {MODEL_VERSION})")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise


@app.get("/", tags=["General"])
async def root() -> Dict[str, str]:
    """Root endpoint."""
    return {
        "message": "Credit Risk Scoring API",
        "version": MODEL_VERSION,
        "status": "operational",
        "docs": "/docs"
    }


@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
async def health_check() -> HealthResponse:
    """Health check endpoint."""
    try:
        # Check if model is loaded
        if not hasattr(app.state, "model"):
            return HealthResponse(
                status="unhealthy",
                model_loaded=False,
                model_version=None
            )
        
        return HealthResponse(
            status="healthy",
            model_loaded=True,
            model_version=app.state.model_version
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthResponse(
            status="unhealthy",
            model_loaded=False,
            model_version=None
        )


@app.get("/metrics", response_model=MetricsResponse, tags=["Monitoring"])
async def get_metrics() -> MetricsResponse:
    """Get performance metrics."""
    try:
        metrics = monitor.calculate_metrics(window_minutes=60)
        return MetricsResponse(
            total_predictions=metrics.total_predictions,
            avg_response_time_ms=metrics.avg_response_time_ms,
            p95_response_time_ms=metrics.p95_response_time_ms,
            avg_risk_score=metrics.avg_risk_score,
            high_risk_percentage=metrics.high_risk_percentage
        )
    except Exception as e:
        logger.error(f"Failed to get metrics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve metrics"
        )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict_risk(tx: Transaction) -> PredictionResponse:
    """Predict credit risk for a transaction.
    
    Parameters
    ----------
    tx : Transaction
        Transaction data
        
    Returns
    -------
    PredictionResponse
        Risk probability and recommendation
    """
    start_time = time.time()
    
    try:
        # Validate input
        validate_model_input(tx.dict())
        
        # Prepare data
        df = pd.DataFrame([tx.dict()])
        X, _ = engineer_features(df)
        
        # Handle missing features
        model = app.state.model
        if hasattr(model, 'feature_names_in_'):
            for col in model.feature_names_in_:
                if col not in X.columns:
                    X[col] = 0
            X = X[model.feature_names_in_]
        
        # Make prediction
        prob = float(model.predict_proba(X)[:, 1][0])
        
        # Determine recommendation
        if prob < 0.3:
            recommendation = "APPROVE"
            risk_level = "LOW"
        elif prob < 0.7:
            recommendation = "REVIEW"
            risk_level = "MEDIUM"
        else:
            recommendation = "DECLINE"
            risk_level = "HIGH"
        
        # Calculate response time
        response_time_ms = (time.time() - start_time) * 1000
        
        # Log prediction
        monitor.log_prediction(
            input_features=tx.dict(),
            prediction=prob,
            model_version=app.state.model_version,
            response_time_ms=response_time_ms
        )
        
        logger.info(
            f"Prediction: {prob:.3f} ({risk_level}) - "
            f"Response time: {response_time_ms:.2f}ms"
        )
        
        return PredictionResponse(
            risk_probability=prob,
            risk_level=risk_level,
            recommendation=recommendation,
            model_version=app.state.model_version
        )
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


@app.post("/predict/batch", tags=["Prediction"])
async def predict_batch(transactions: list[Transaction]) -> Dict[str, Any]:
    """Batch prediction endpoint.
    
    Parameters
    ----------
    transactions : list
        List of transactions
        
    Returns
    -------
    dict
        Batch prediction results
    """
    try:
        results = []
        
        for tx in transactions:
            response = await predict_risk(tx)
            results.append({
                "input": tx.dict(),
                "prediction": response.dict()
            })
        
        return {
            "count": len(results),
            "results": results
        }
        
    except Exception as e:
        logger.error(f"Batch prediction failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction failed: {str(e)}"
        )