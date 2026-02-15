"""Tests for prediction module."""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier


def test_prediction_output_range():
    """Test that predictions are valid probabilities."""
    X_train = pd.DataFrame({
        'amount': np.random.rand(100) * 1000,
        'frequency': np.random.randint(1, 50, 100)
    })
    y_train = pd.Series(np.random.randint(0, 2, 100))
    
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X_train, y_train)
    
    X_test = pd.DataFrame({
        'amount': [500.0, 1000.0],
        'frequency': [10, 25]
    })
    
    probs = model.predict_proba(X_test)[:, 1]
    
    assert all(0 <= p <= 1 for p in probs)
    assert len(probs) == 2


def test_batch_prediction():
    """Test batch prediction functionality."""
    X_train = pd.DataFrame({
        'feature1': np.random.rand(50),
        'feature2': np.random.rand(50)
    })
    y_train = pd.Series(np.random.randint(0, 2, 50))
    
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X_train, y_train)
    
    batch_size = 10
    X_batch = pd.DataFrame({
        'feature1': np.random.rand(batch_size),
        'feature2': np.random.rand(batch_size)
    })
    
    predictions = model.predict(X_batch)
    
    assert len(predictions) == batch_size
    assert all(p in [0, 1] for p in predictions)
