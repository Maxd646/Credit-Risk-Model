"""Pydantic models for API request/response validation."""

from pydantic import BaseModel, Field
from typing import Optional


class Transaction(BaseModel):
    """Transaction input model."""
    Amount: float = Field(..., gt=0, description="Transaction amount (must be positive)")
    FraudResult: int = Field(..., ge=0, le=1, description="Fraud result flag (0 or 1)")
    
    class Config:
        schema_extra = {
            "example": {
                "Amount": 1000.0,
                "FraudResult": 0
            }
        }


class PredictionResponse(BaseModel):
    """Prediction response model."""
    risk_probability: float = Field(..., ge=0, le=1, description="Risk probability (0-1)")
    risk_level: str = Field(..., description="Risk level (LOW, MEDIUM, HIGH)")
    recommendation: str = Field(..., description="Recommendation (APPROVE, REVIEW, DECLINE)")
    model_version: str = Field(..., description="Model version used")
    
    class Config:
        schema_extra = {
            "example": {
                "risk_probability": 0.23,
                "risk_level": "LOW",
                "recommendation": "APPROVE",
                "model_version": "1.0.0"
            }
        }


class HealthResponse(BaseModel):
    """Health check response model."""
    status: str = Field(..., description="Service status (healthy/unhealthy)")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    model_version: Optional[str] = Field(None, description="Loaded model version")
    
    class Config:
        schema_extra = {
            "example": {
                "status": "healthy",
                "model_loaded": True,
                "model_version": "1.0.0"
            }
        }


class MetricsResponse(BaseModel):
    """Metrics response model."""
    total_predictions: int = Field(..., description="Total predictions in window")
    avg_response_time_ms: float = Field(..., description="Average response time (ms)")
    p95_response_time_ms: float = Field(..., description="95th percentile response time (ms)")
    avg_risk_score: float = Field(..., description="Average risk score")
    high_risk_percentage: float = Field(..., description="Percentage of high-risk predictions")
    
    class Config:
        schema_extra = {
            "example": {
                "total_predictions": 1523,
                "avg_response_time_ms": 45.2,
                "p95_response_time_ms": 89.5,
                "avg_risk_score": 0.34,
                "high_risk_percentage": 28.5
            }
        }
