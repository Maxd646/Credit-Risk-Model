"""Data validation utilities."""

from typing import Any, Dict, List, Optional
import pandas as pd
import numpy as np
from pydantic import BaseModel, validator, Field

from src.logger import setup_logger

logger = setup_logger(__name__)


class TransactionData(BaseModel):
    """Validation model for transaction data."""
    Amount: float = Field(..., gt=0, description="Transaction amount must be positive")
    FraudResult: int = Field(..., ge=0, le=1, description="Fraud result must be 0 or 1")
    
    class Config:
        schema_extra = {
            "example": {
                "Amount": 1000.0,
                "FraudResult": 0
            }
        }
    
    @validator('Amount')
    def validate_amount(cls, v):
        """Validate transaction amount."""
        if v > 1000000:
            logger.warning(f"Unusually large transaction amount: {v}")
        return v


class DataFrameValidator:
    """Validator for pandas DataFrames."""
    
    @staticmethod
    def validate_required_columns(
        df: pd.DataFrame,
        required_columns: List[str]
    ) -> None:
        """Validate that required columns exist.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame to validate
        required_columns : list
            List of required column names
            
        Raises
        ------
        ValueError
            If required columns are missing
        """
        missing = set(required_columns) - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        logger.info("All required columns present")
    
    @staticmethod
    def validate_no_nulls(
        df: pd.DataFrame,
        columns: Optional[List[str]] = None
    ) -> None:
        """Validate that specified columns have no null values.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame to validate
        columns : list, optional
            Columns to check. If None, checks all columns.
            
        Raises
        ------
        ValueError
            If null values are found
        """
        cols_to_check = columns or df.columns
        null_counts = df[cols_to_check].isnull().sum()
        
        if null_counts.any():
            null_cols = null_counts[null_counts > 0]
            raise ValueError(f"Null values found: {null_cols.to_dict()}")
        
        logger.info("No null values found")
    
    @staticmethod
    def validate_data_types(
        df: pd.DataFrame,
        expected_types: Dict[str, type]
    ) -> None:
        """Validate column data types.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame to validate
        expected_types : dict
            Dictionary mapping column names to expected types
            
        Raises
        ------
        ValueError
            If data types don't match
        """
        for col, expected_type in expected_types.items():
            if col not in df.columns:
                continue
            
            actual_type = df[col].dtype
            
            # Handle numeric types
            if expected_type in [int, float, np.number]:
                if not pd.api.types.is_numeric_dtype(actual_type):
                    raise ValueError(
                        f"Column '{col}' expected numeric, got {actual_type}"
                    )
            
            # Handle string types
            elif expected_type == str:
                if not pd.api.types.is_string_dtype(actual_type) and actual_type != object:
                    raise ValueError(
                        f"Column '{col}' expected string, got {actual_type}"
                    )
        
        logger.info("Data types validated")
    
    @staticmethod
    def validate_value_ranges(
        df: pd.DataFrame,
        ranges: Dict[str, tuple]
    ) -> None:
        """Validate that values are within expected ranges.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame to validate
        ranges : dict
            Dictionary mapping column names to (min, max) tuples
            
        Raises
        ------
        ValueError
            If values are out of range
        """
        for col, (min_val, max_val) in ranges.items():
            if col not in df.columns:
                continue
            
            out_of_range = (df[col] < min_val) | (df[col] > max_val)
            
            if out_of_range.any():
                count = out_of_range.sum()
                raise ValueError(
                    f"Column '{col}' has {count} values outside range [{min_val}, {max_val}]"
                )
        
        logger.info("Value ranges validated")
    
    @staticmethod
    def validate_no_duplicates(
        df: pd.DataFrame,
        subset: Optional[List[str]] = None
    ) -> None:
        """Validate that there are no duplicate rows.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame to validate
        subset : list, optional
            Columns to consider for duplicates
            
        Raises
        ------
        ValueError
            If duplicates are found
        """
        duplicates = df.duplicated(subset=subset)
        
        if duplicates.any():
            count = duplicates.sum()
            raise ValueError(f"Found {count} duplicate rows")
        
        logger.info("No duplicates found")


def validate_model_input(data: Dict[str, Any]) -> TransactionData:
    """Validate model input data.
    
    Parameters
    ----------
    data : dict
        Input data dictionary
        
    Returns
    -------
    TransactionData
        Validated transaction data
        
    Raises
    ------
    ValidationError
        If validation fails
    """
    return TransactionData(**data)


def validate_training_data(df: pd.DataFrame) -> None:
    """Validate training data.
    
    Parameters
    ----------
    df : pd.DataFrame
        Training data
        
    Raises
    ------
    ValueError
        If validation fails
    """
    validator = DataFrameValidator()
    
    # Check required columns
    validator.validate_required_columns(df, ["Amount", "FraudResult"])
    
    # Check data types
    validator.validate_data_types(df, {
        "Amount": float,
        "FraudResult": int
    })
    
    # Check value ranges
    validator.validate_value_ranges(df, {
        "Amount": (0, float('inf')),
        "FraudResult": (0, 1)
    })
    
    logger.info("Training data validation passed")
