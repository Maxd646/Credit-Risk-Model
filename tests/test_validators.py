"""Tests for validation module."""

import pytest
import pandas as pd
import numpy as np
from pydantic import ValidationError

from src.validators import (
    TransactionData,
    DataFrameValidator,
    validate_model_input,
    validate_training_data
)


def test_transaction_data_valid():
    """Test valid transaction data."""
    data = TransactionData(Amount=1000.0, FraudResult=0)
    
    assert data.Amount == 1000.0
    assert data.FraudResult == 0


def test_transaction_data_invalid_amount():
    """Test invalid transaction amount."""
    with pytest.raises(ValidationError):
        TransactionData(Amount=-100.0, FraudResult=0)


def test_transaction_data_invalid_fraud_result():
    """Test invalid fraud result."""
    with pytest.raises(ValidationError):
        TransactionData(Amount=1000.0, FraudResult=2)


def test_dataframe_validator_required_columns():
    """Test required columns validation."""
    df = pd.DataFrame({"Amount": [100, 200]})
    validator = DataFrameValidator()
    
    with pytest.raises(ValueError):
        validator.validate_required_columns(df, ["Amount", "FraudResult"])


def test_dataframe_validator_no_nulls():
    """Test no nulls validation."""
    df = pd.DataFrame({
        "Amount": [100, None, 300],
        "FraudResult": [0, 1, 0]
    })
    validator = DataFrameValidator()
    
    with pytest.raises(ValueError):
        validator.validate_no_nulls(df)


def test_dataframe_validator_value_ranges():
    """Test value ranges validation."""
    df = pd.DataFrame({
        "Amount": [100, 200, -50],
        "FraudResult": [0, 1, 0]
    })
    validator = DataFrameValidator()
    
    with pytest.raises(ValueError):
        validator.validate_value_ranges(df, {"Amount": (0, float('inf'))})


def test_validate_model_input_valid():
    """Test valid model input."""
    data = {"Amount": 1000.0, "FraudResult": 0}
    result = validate_model_input(data)
    
    assert isinstance(result, TransactionData)
    assert result.Amount == 1000.0


def test_validate_training_data_valid():
    """Test valid training data."""
    df = pd.DataFrame({
        "Amount": [100.0, 200.0, 300.0],
        "FraudResult": [0, 1, 0]
    })
    
    # Should not raise
    validate_training_data(df)
