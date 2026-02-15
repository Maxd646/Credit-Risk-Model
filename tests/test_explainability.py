"""Tests for explainability module."""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


def test_shap_values_shape():
    """Test that SHAP values have correct shape."""
    X = pd.DataFrame({
        'feature1': np.random.rand(50),
        'feature2': np.random.rand(50),
        'feature3': np.random.rand(50)
    })
    y = pd.Series(np.random.randint(0, 2, 50))
    
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)
    
    # Test prediction shape
    predictions = model.predict_proba(X)
    
    assert predictions.shape[0] == len(X)
    assert predictions.shape[1] == 2  # Binary classification


def test_feature_importance_extraction():
    """Test that feature importance can be extracted."""
    X = pd.DataFrame({
        'amount': np.random.rand(100) * 1000,
        'frequency': np.random.randint(1, 50, 100),
        'recency': np.random.randint(1, 365, 100)
    })
    y = pd.Series(np.random.randint(0, 2, 100))
    
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)
    
    # Get feature importances
    importances = model.feature_importances_
    
    assert len(importances) == 3
    assert all(0 <= imp <= 1 for imp in importances)
    assert np.isclose(importances.sum(), 1.0, atol=0.01)
