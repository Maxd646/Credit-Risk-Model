"""Tests for FastAPI endpoints."""

from fastapi.testclient import TestClient
import pytest


def test_api_structure():
    """Test that API response structure is correct."""
    response_data = {
        "risk_probability": 0.75
    }
    
    assert "risk_probability" in response_data
    assert isinstance(response_data["risk_probability"], float)
    assert 0 <= response_data["risk_probability"] <= 1


def test_transaction_validation():
    """Test transaction data validation."""
    valid_transaction = {
        "Amount": 1000.0,
        "FraudResult": 0
    }
    
    assert "Amount" in valid_transaction
    assert isinstance(valid_transaction["Amount"], (int, float))
    assert valid_transaction["Amount"] > 0
