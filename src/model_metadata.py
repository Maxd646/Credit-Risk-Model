"""Model metadata and versioning."""

from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import json
import joblib

from src.logger import setup_logger

logger = setup_logger(__name__)


@dataclass
class ModelMetadata:
    """Model metadata for tracking and versioning."""
    model_name: str
    version: str
    created_at: str
    algorithm: str
    hyperparameters: Dict[str, Any]
    
    # Performance metrics
    roc_auc: float
    precision: float
    recall: float
    f1_score: float
    
    # Training info
    training_samples: int
    validation_samples: int
    feature_count: int
    feature_names: list
    
    # Data info
    data_source: str
    data_version: Optional[str] = None
    
    # Additional info
    description: Optional[str] = None
    tags: Optional[list] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)
    
    def save(self, path: Path) -> None:
        """Save metadata to JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            f.write(self.to_json())
        logger.info(f"Saved model metadata to {path}")
    
    @classmethod
    def load(cls, path: Path) -> 'ModelMetadata':
        """Load metadata from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        logger.info(f"Loaded model metadata from {path}")
        return cls(**data)


class ModelRegistry:
    """Model registry for managing multiple model versions."""
    
    def __init__(self, registry_dir: Path = Path("main/registry")):
        self.registry_dir = registry_dir
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        self.index_file = self.registry_dir / "index.json"
        self._load_index()
    
    def _load_index(self) -> None:
        """Load registry index."""
        if self.index_file.exists():
            with open(self.index_file, 'r') as f:
                self.index = json.load(f)
        else:
            self.index = {"models": [], "latest": None}
    
    def _save_index(self) -> None:
        """Save registry index."""
        with open(self.index_file, 'w') as f:
            json.dump(self.index, f, indent=2)
    
    def register_model(
        self,
        model: Any,
        metadata: ModelMetadata,
        set_as_latest: bool = True
    ) -> Path:
        """Register a new model version.
        
        Parameters
        ----------
        model : Any
            Trained model object
        metadata : ModelMetadata
            Model metadata
        set_as_latest : bool
            Whether to set this as the latest version
            
        Returns
        -------
        Path
            Path to saved model
        """
        # Create version directory
        version_dir = self.registry_dir / metadata.version
        version_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        model_path = version_dir / "model.pkl"
        joblib.dump(model, model_path)
        
        # Save metadata
        metadata_path = version_dir / "metadata.json"
        metadata.save(metadata_path)
        
        # Update index
        model_entry = {
            "version": metadata.version,
            "created_at": metadata.created_at,
            "algorithm": metadata.algorithm,
            "roc_auc": metadata.roc_auc,
            "path": str(model_path)
        }
        
        self.index["models"].append(model_entry)
        
        if set_as_latest:
            self.index["latest"] = metadata.version
        
        self._save_index()
        
        logger.info(f"Registered model version {metadata.version}")
        return model_path
    
    def load_model(self, version: Optional[str] = None) -> tuple:
        """Load model and metadata.
        
        Parameters
        ----------
        version : str, optional
            Model version to load. If None, loads latest.
            
        Returns
        -------
        tuple
            (model, metadata)
        """
        if version is None:
            version = self.index["latest"]
        
        if version is None:
            raise ValueError("No models registered")
        
        version_dir = self.registry_dir / version
        model_path = version_dir / "model.pkl"
        metadata_path = version_dir / "metadata.json"
        
        model = joblib.load(model_path)
        metadata = ModelMetadata.load(metadata_path)
        
        logger.info(f"Loaded model version {version}")
        return model, metadata
    
    def list_models(self) -> list:
        """List all registered models."""
        return self.index["models"]
    
    def get_latest_version(self) -> Optional[str]:
        """Get latest model version."""
        return self.index["latest"]


def create_model_metadata(
    model_name: str,
    algorithm: str,
    hyperparameters: Dict[str, Any],
    metrics: Dict[str, float],
    training_info: Dict[str, Any],
    data_source: str
) -> ModelMetadata:
    """Create model metadata object.
    
    Parameters
    ----------
    model_name : str
        Name of the model
    algorithm : str
        Algorithm used (e.g., 'RandomForest')
    hyperparameters : dict
        Model hyperparameters
    metrics : dict
        Performance metrics (roc_auc, precision, recall, f1_score)
    training_info : dict
        Training information (training_samples, validation_samples, etc.)
    data_source : str
        Source of training data
        
    Returns
    -------
    ModelMetadata
        Model metadata object
    """
    version = datetime.now().strftime("%Y%m%d_%H%M%S")
    created_at = datetime.now().isoformat()
    
    return ModelMetadata(
        model_name=model_name,
        version=version,
        created_at=created_at,
        algorithm=algorithm,
        hyperparameters=hyperparameters,
        roc_auc=metrics.get("roc_auc", 0.0),
        precision=metrics.get("precision", 0.0),
        recall=metrics.get("recall", 0.0),
        f1_score=metrics.get("f1_score", 0.0),
        training_samples=training_info.get("training_samples", 0),
        validation_samples=training_info.get("validation_samples", 0),
        feature_count=training_info.get("feature_count", 0),
        feature_names=training_info.get("feature_names", []),
        data_source=data_source,
        description=f"{algorithm} model for credit risk prediction",
        tags=["credit-risk", "classification", algorithm.lower()]
    )
