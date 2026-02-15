"""Tests for configuration module."""

from src.config import AppConfig, DataConfig, ModelConfig


def test_data_config_defaults():
    """Test DataConfig default values."""
    config = DataConfig()
    
    assert config.test_size == 0.2
    assert config.random_state == 42
    assert config.stratify is True


def test_data_config_validation():
    """Test DataConfig validation."""
    try:
        config = DataConfig(test_size=1.5)
        assert False, "Should raise ValueError"
    except ValueError as e:
        assert "test_size" in str(e)


def test_model_config_defaults():
    """Test ModelConfig default values."""
    config = ModelConfig()
    
    assert config.cv_folds == 3
    assert config.scoring_metric == "roc_auc"
    assert config.n_jobs == -1


def test_app_config_initialization():
    """Test AppConfig initialization."""
    config = AppConfig()
    
    assert config.data is not None
    assert config.model is not None
    assert config.api is not None
    assert config.dashboard is not None
    assert config.shap is not None
    assert config.logging is not None
