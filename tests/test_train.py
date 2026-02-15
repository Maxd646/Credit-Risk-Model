"""Tests for model training module."""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


def test_model_training_basic():
    """Test that a basic model can be trained."""
    X = pd.DataFrame({
        'feature1': np.random.rand(100),
        'feature2': np.random.rand(100)
    })
    y = pd.Series(np.random.randint(0, 2, 100))
    
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)
    
    assert hasattr(model, 'coef_')
    assert model.coef_.shape[1] == 2


def test_model_prediction_shape():
    """Test that model predictions have correct shape."""
    X_train = pd.DataFrame({
        'feature1': np.random.rand(100),
        'feature2': np.random.rand(100)
    })
    y_train = pd.Series(np.random.randint(0, 2, 100))
    
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X_train, y_train)
    
    X_test = pd.DataFrame({
        'feature1': np.random.rand(20),
        'feature2': np.random.rand(20)
    })
    
    predictions = model.predict_proba(X_test)
    
    assert predictions.shape == (20, 2)
    assert np.allclose(predictions.sum(axis=1), 1.0)
