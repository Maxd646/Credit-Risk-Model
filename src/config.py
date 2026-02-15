"""Configuration management using dataclasses."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any
import os


@dataclass
class DataConfig:
    """Data processing configuration."""
    raw_data_path: Path = Path("data/raw/data.csv")
    processed_data_path: Path = Path("data/processed")
    test_size: float = 0.2
    random_state: int = 42
    stratify: bool = True
    
    def __post_init__(self):
        """Validate configuration."""
        if not 0 < self.test_size < 1:
            raise ValueError("test_size must be between 0 and 1")


@dataclass
class ModelConfig:
    """Model training configuration."""
    model_output_path: Path = Path("main/model.pkl")
    mlflow_experiment: str = "credit-risk"
    mlflow_tracking_uri: str = "file:./mlruns"
    
    # Model hyperparameters
    logistic_regression_params: Dict[str, Any] = field(default_factory=lambda: {
        "C": [0.1, 1, 10],
        "max_iter": 1000,
        "solver": "lbfgs"
    })
    
    random_forest_params: Dict[str, Any] = field(default_factory=lambda: {
        "n_estimators": [100, 300],
        "max_depth": [None, 10],
        "random_state": 42
    })
    
    gradient_boosting_params: Dict[str, Any] = field(default_factory=lambda: {
        "n_estimators": [100, 300],
        "learning_rate": [0.05, 0.1],
        "random_state": 42
    })
    
    # Training settings
    cv_folds: int = 3
    scoring_metric: str = "roc_auc"
    n_jobs: int = -1


@dataclass
class APIConfig:
    """API server configuration."""
    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = False
    workers: int = 4
    log_level: str = "info"
    
    # Model settings
    model_path: Path = Path("main/model.pkl")
    max_request_size: int = 1024 * 1024  # 1MB
    
    # Rate limiting
    rate_limit_requests: int = 100
    rate_limit_period: int = 60  # seconds


@dataclass
class DashboardConfig:
    """Dashboard configuration."""
    title: str = "Credit Risk Scoring Dashboard"
    page_icon: str = "💳"
    layout: str = "wide"
    
    # Performance settings
    cache_ttl: int = 3600  # 1 hour
    max_rows_display: int = 100


@dataclass
class SHAPConfig:
    """SHAP explainability configuration."""
    output_dir: Path = Path("outputs/shap")
    sample_size: int = 100
    random_state: int = 42
    
    # Plot settings
    dpi: int = 300
    figure_width: int = 10
    figure_height: int = 6


@dataclass
class LoggingConfig:
    """Logging configuration."""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    log_file: Path = Path("logs/app.log")
    max_bytes: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5


@dataclass
class AppConfig:
    """Main application configuration."""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    api: APIConfig = field(default_factory=APIConfig)
    dashboard: DashboardConfig = field(default_factory=DashboardConfig)
    shap: SHAPConfig = field(default_factory=SHAPConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    
    # Environment
    environment: str = field(default_factory=lambda: os.getenv("ENVIRONMENT", "development"))
    debug: bool = field(default_factory=lambda: os.getenv("DEBUG", "False").lower() == "true")
    
    def __post_init__(self):
        """Create necessary directories."""
        self.model.model_output_path.parent.mkdir(parents=True, exist_ok=True)
        self.shap.output_dir.mkdir(parents=True, exist_ok=True)
        self.logging.log_file.parent.mkdir(parents=True, exist_ok=True)
        self.data.processed_data_path.mkdir(parents=True, exist_ok=True)


# Global configuration instance
config = AppConfig()
