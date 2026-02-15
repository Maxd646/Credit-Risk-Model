"""Model performance monitoring and metrics tracking."""

from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import json
import pandas as pd
import numpy as np

from src.logger import setup_logger

logger = setup_logger(__name__)


@dataclass
class PredictionLog:
    """Log entry for a single prediction."""
    timestamp: str
    input_features: Dict
    prediction: float
    prediction_class: int
    model_version: str
    response_time_ms: float


@dataclass
class PerformanceMetrics:
    """Performance metrics for monitoring."""
    timestamp: str
    total_predictions: int
    avg_response_time_ms: float
    p95_response_time_ms: float
    p99_response_time_ms: float
    error_rate: float
    avg_risk_score: float
    high_risk_percentage: float


class ModelMonitor:
    """Monitor model performance and predictions."""
    
    def __init__(self, log_dir: Path = Path("logs/predictions")):
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.current_log_file = self.log_dir / f"predictions_{datetime.now().strftime('%Y%m%d')}.jsonl"
        self.metrics_file = self.log_dir / "metrics.json"
        self.predictions: List[PredictionLog] = []
    
    def log_prediction(
        self,
        input_features: Dict,
        prediction: float,
        model_version: str,
        response_time_ms: float
    ) -> None:
        """Log a prediction.
        
        Parameters
        ----------
        input_features : dict
            Input features used for prediction
        prediction : float
            Predicted probability
        model_version : str
            Version of model used
        response_time_ms : float
            Response time in milliseconds
        """
        log_entry = PredictionLog(
            timestamp=datetime.now().isoformat(),
            input_features=input_features,
            prediction=float(prediction),
            prediction_class=int(prediction > 0.5),
            model_version=model_version,
            response_time_ms=response_time_ms
        )
        
        # Append to file
        with open(self.current_log_file, 'a') as f:
            f.write(json.dumps(asdict(log_entry)) + '\n')
        
        # Keep in memory for metrics
        self.predictions.append(log_entry)
        
        # Limit memory usage
        if len(self.predictions) > 10000:
            self.predictions = self.predictions[-5000:]
    
    def calculate_metrics(self, window_minutes: int = 60) -> PerformanceMetrics:
        """Calculate performance metrics.
        
        Parameters
        ----------
        window_minutes : int
            Time window for metrics calculation
            
        Returns
        -------
        PerformanceMetrics
            Calculated metrics
        """
        if not self.predictions:
            return PerformanceMetrics(
                timestamp=datetime.now().isoformat(),
                total_predictions=0,
                avg_response_time_ms=0.0,
                p95_response_time_ms=0.0,
                p99_response_time_ms=0.0,
                error_rate=0.0,
                avg_risk_score=0.0,
                high_risk_percentage=0.0
            )
        
        # Filter by time window
        cutoff_time = datetime.now().timestamp() - (window_minutes * 60)
        recent_predictions = [
            p for p in self.predictions
            if datetime.fromisoformat(p.timestamp).timestamp() > cutoff_time
        ]
        
        if not recent_predictions:
            recent_predictions = self.predictions[-100:]  # Use last 100 if no recent
        
        # Calculate metrics
        response_times = [p.response_time_ms for p in recent_predictions]
        predictions = [p.prediction for p in recent_predictions]
        
        metrics = PerformanceMetrics(
            timestamp=datetime.now().isoformat(),
            total_predictions=len(recent_predictions),
            avg_response_time_ms=float(np.mean(response_times)),
            p95_response_time_ms=float(np.percentile(response_times, 95)),
            p99_response_time_ms=float(np.percentile(response_times, 99)),
            error_rate=0.0,  # Would track actual errors
            avg_risk_score=float(np.mean(predictions)),
            high_risk_percentage=float(sum(1 for p in predictions if p > 0.5) / len(predictions) * 100)
        )
        
        # Save metrics
        self._save_metrics(metrics)
        
        return metrics
    
    def _save_metrics(self, metrics: PerformanceMetrics) -> None:
        """Save metrics to file."""
        metrics_data = asdict(metrics)
        
        # Load existing metrics
        if self.metrics_file.exists():
            with open(self.metrics_file, 'r') as f:
                all_metrics = json.load(f)
        else:
            all_metrics = []
        
        # Append new metrics
        all_metrics.append(metrics_data)
        
        # Keep last 1000 entries
        all_metrics = all_metrics[-1000:]
        
        # Save
        with open(self.metrics_file, 'w') as f:
            json.dump(all_metrics, f, indent=2)
    
    def get_metrics_history(self, limit: int = 100) -> List[Dict]:
        """Get metrics history.
        
        Parameters
        ----------
        limit : int
            Number of recent metrics to return
            
        Returns
        -------
        list
            List of metrics dictionaries
        """
        if not self.metrics_file.exists():
            return []
        
        with open(self.metrics_file, 'r') as f:
            all_metrics = json.load(f)
        
        return all_metrics[-limit:]
    
    def detect_drift(self, baseline_mean: float, threshold: float = 0.1) -> bool:
        """Detect prediction drift.
        
        Parameters
        ----------
        baseline_mean : float
            Baseline mean prediction value
        threshold : float
            Drift threshold (percentage)
            
        Returns
        -------
        bool
            True if drift detected
        """
        if not self.predictions:
            return False
        
        recent_predictions = [p.prediction for p in self.predictions[-1000:]]
        current_mean = np.mean(recent_predictions)
        
        drift_percentage = abs(current_mean - baseline_mean) / baseline_mean
        
        if drift_percentage > threshold:
            logger.warning(
                f"Prediction drift detected: {drift_percentage:.2%} "
                f"(baseline: {baseline_mean:.3f}, current: {current_mean:.3f})"
            )
            return True
        
        return False
    
    def generate_report(self) -> Dict:
        """Generate monitoring report.
        
        Returns
        -------
        dict
            Monitoring report
        """
        metrics = self.calculate_metrics()
        
        report = {
            "generated_at": datetime.now().isoformat(),
            "current_metrics": asdict(metrics),
            "total_predictions_today": len(self.predictions),
            "model_health": "healthy" if metrics.avg_response_time_ms < 100 else "degraded",
            "recommendations": []
        }
        
        # Add recommendations
        if metrics.avg_response_time_ms > 100:
            report["recommendations"].append("Consider model optimization - response time high")
        
        if metrics.high_risk_percentage > 50:
            report["recommendations"].append("High percentage of risky predictions - review model")
        
        return report


# Global monitor instance
monitor = ModelMonitor()
