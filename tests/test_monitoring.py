"""Tests for monitoring module."""

import pytest
from src.monitoring import ModelMonitor, PredictionLog, PerformanceMetrics


def test_prediction_log_creation():
    """Test PredictionLog creation."""
    log = PredictionLog(
        timestamp="2026-02-15T20:00:00",
        input_features={"Amount": 1000.0},
        prediction=0.25,
        prediction_class=0,
        model_version="1.0.0",
        response_time_ms=45.2
    )
    
    assert log.prediction == 0.25
    assert log.prediction_class == 0
    assert log.response_time_ms == 45.2


def test_model_monitor_log_prediction():
    """Test logging predictions."""
    monitor = ModelMonitor()
    
    monitor.log_prediction(
        input_features={"Amount": 1000.0},
        prediction=0.25,
        model_version="1.0.0",
        response_time_ms=45.2
    )
    
    assert len(monitor.predictions) > 0


def test_model_monitor_calculate_metrics():
    """Test metrics calculation."""
    monitor = ModelMonitor()
    
    # Log some predictions
    for i in range(10):
        monitor.log_prediction(
            input_features={"Amount": 1000.0 + i},
            prediction=0.2 + (i * 0.05),
            model_version="1.0.0",
            response_time_ms=40.0 + i
        )
    
    metrics = monitor.calculate_metrics()
    
    assert isinstance(metrics, PerformanceMetrics)
    assert metrics.total_predictions == 10
    assert metrics.avg_response_time_ms > 0


def test_model_monitor_detect_drift():
    """Test drift detection."""
    monitor = ModelMonitor()
    
    # Log predictions with drift
    for i in range(100):
        monitor.log_prediction(
            input_features={"Amount": 1000.0},
            prediction=0.8,  # High predictions
            model_version="1.0.0",
            response_time_ms=45.0
        )
    
    # Should detect drift from baseline of 0.3
    drift_detected = monitor.detect_drift(baseline_mean=0.3, threshold=0.1)
    assert drift_detected is True
