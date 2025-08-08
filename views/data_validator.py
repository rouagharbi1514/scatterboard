#!/usr/bin/env python3
"""
Data Validation Utility Module
=============================

This module provides comprehensive data validation utilities for all view components.
It handles common data validation scenarios including:
- Missing or empty data
- Invalid data types
- Out-of-range values
- Null/NaN handling
- Excel import artifacts

Usage:
    from views.data_validator import DataValidator, ValidationResult
    
    validator = DataValidator()
    result = validator.validate_dataframe(df, required_columns=['Date', 'TotalRevenue'])
    
    if not result.is_valid:
        print(f"Validation errors: {result.errors}")
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of data validation."""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    cleaned_data: Optional[pd.DataFrame] = None
    validation_summary: Dict[str, Any] = None


class DataValidator:
    """Comprehensive data validator for hotel dashboard data."""
    
    def __init__(self):
        """Initialize the data validator."""
        self.required_numeric_columns = [
            'TotalRevenue', 'TotalCosts', 'ADR', 'RevPAR', 'MarketingSpend',
            'CustomerAcquisitionCost', 'CustomerLifetimeValue', 'MaintenanceCosts'
        ]
        
        self.required_percentage_columns = [
            'OccupancyRate'
        ]
        
        self.required_integer_columns = [
            'RoomsSold', 'RoomsAvailable', 'StaffCount'
        ]
        
        self.required_rating_columns = [
            'GuestSatisfaction'
        ]
        
        self.date_columns = [
            'Date', 'CheckInDate', 'CheckOutDate', 'BookingDate'
        ]
    
    def validate_dataframe(self, df: pd.DataFrame, 
                          required_columns: Optional[List[str]] = None,
                          auto_clean: bool = True) -> ValidationResult:
        """Validate a complete dataframe.
        
        Args:
            df: DataFrame to validate
            required_columns: List of columns that must be present
            auto_clean: Whether to automatically clean the data
            
        Returns:
            ValidationResult with validation status and cleaned data
        """
        errors = []
        warnings = []
        cleaned_df = df.copy() if df is not None and not df.empty else pd.DataFrame()
        
        # Basic structure validation
        if df is None:
            errors.append("DataFrame is None")
            return ValidationResult(False, errors, warnings)
        
        if df.empty:
            errors.append("DataFrame is empty")
            return ValidationResult(False, errors, warnings)
        
        # Check required columns
        if required_columns:
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                errors.append(f"Missing required columns: {missing_columns}")
        
        # Validate data types and ranges
        type_validation = self._validate_data_types(cleaned_df)
        errors.extend(type_validation['errors'])
        warnings.extend(type_validation['warnings'])
        
        if auto_clean and type_validation['cleaned_data'] is not None:
            cleaned_df = type_validation['cleaned_data']
        
        # Validate data ranges
        range_validation = self._validate_data_ranges(cleaned_df)
        errors.extend(range_validation['errors'])
        warnings.extend(range_validation['warnings'])
        
        if auto_clean and range_validation['cleaned_data'] is not None:
            cleaned_df = range_validation['cleaned_data']
        
        # Validate business logic
        business_validation = self._validate_business_logic(cleaned_df)
        warnings.extend(business_validation['warnings'])
        
        # Generate validation summary
        summary = self._generate_validation_summary(df, cleaned_df, errors, warnings)
        
        is_valid = len(errors) == 0
        
        return ValidationResult(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            cleaned_data=cleaned_df if is_valid else None,
            validation_summary=summary
        )
    
    def _validate_data_types(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate and clean data types."""
        errors = []
        warnings = []
        cleaned_df = df.copy()
        
        # Validate and clean numeric columns
        for col in self.required_numeric_columns:
            if col in df.columns:
                try:
                    # Convert to numeric, coercing errors to NaN
                    numeric_series = pd.to_numeric(df[col], errors='coerce')
                    
                    # Check for conversion failures
                    conversion_failures = df[col].notna() & numeric_series.isna()
                    if conversion_failures.any():
                        failure_count = conversion_failures.sum()
                        warnings.append(f"Column '{col}': {failure_count} values could not be converted to numeric")
                    
                    cleaned_df[col] = numeric_series
                    
                except Exception as e:
                    errors.append(f"Error processing numeric column '{col}': {str(e)}")
        
        # Validate and clean integer columns
        for col in self.required_integer_columns:
            if col in df.columns:
                try:
                    # Convert to numeric first, then to integer
                    numeric_series = pd.to_numeric(df[col], errors='coerce')
                    
                    # Check for non-integer values
                    non_integer_mask = numeric_series.notna() & (numeric_series != numeric_series.astype(int))
                    if non_integer_mask.any():
                        warnings.append(f"Column '{col}': Contains non-integer values that will be rounded")
                    
                    cleaned_df[col] = numeric_series.round().astype('Int64')  # Nullable integer
                    
                except Exception as e:
                    errors.append(f"Error processing integer column '{col}': {str(e)}")
        
        # Validate and clean date columns
        for col in self.date_columns:
            if col in df.columns:
                try:
                    # Convert to datetime, coercing errors to NaT
                    datetime_series = pd.to_datetime(df[col], errors='coerce')
                    
                    # Check for conversion failures
                    conversion_failures = df[col].notna() & datetime_series.isna()
                    if conversion_failures.any():
                        failure_count = conversion_failures.sum()
                        warnings.append(f"Column '{col}': {failure_count} values could not be converted to datetime")
                    
                    cleaned_df[col] = datetime_series
                    
                except Exception as e:
                    errors.append(f"Error processing date column '{col}': {str(e)}")
        
        return {
            'errors': errors,
            'warnings': warnings,
            'cleaned_data': cleaned_df
        }
    
    def _validate_data_ranges(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate data ranges and boundaries."""
        errors = []
        warnings = []
        cleaned_df = df.copy()
        
        # Validate percentage columns (0-1 range)
        for col in self.required_percentage_columns:
            if col in df.columns:
                series = df[col]
                if series.notna().any():
                    # Check for values outside 0-1 range
                    out_of_range = (series < 0) | (series > 1)
                    if out_of_range.any():
                        out_count = out_of_range.sum()
                        warnings.append(f"Column '{col}': {out_count} values outside valid range (0-1)")
                        
                        # Clip values to valid range
                        cleaned_df[col] = series.clip(0, 1)
        
        # Validate rating columns (typically 1-5 range)
        for col in self.required_rating_columns:
            if col in df.columns:
                series = df[col]
                if series.notna().any():
                    # Check for values outside 1-5 range
                    out_of_range = (series < 1) | (series > 5)
                    if out_of_range.any():
                        out_count = out_of_range.sum()
                        warnings.append(f"Column '{col}': {out_count} values outside typical range (1-5)")
        
        # Validate non-negative columns
        non_negative_columns = self.required_numeric_columns + self.required_integer_columns
        for col in non_negative_columns:
            if col in df.columns:
                series = df[col]
                if series.notna().any():
                    negative_values = series < 0
                    if negative_values.any():
                        negative_count = negative_values.sum()
                        warnings.append(f"Column '{col}': {negative_count} negative values found")
                        
                        # Set negative values to 0 or NaN based on context
                        if col in ['TotalRevenue', 'TotalCosts']:
                            cleaned_df.loc[negative_values, col] = 0
                        else:
                            cleaned_df.loc[negative_values, col] = np.nan
        
        return {
            'errors': errors,
            'warnings': warnings,
            'cleaned_data': cleaned_df
        }
    
    def _validate_business_logic(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate business logic constraints."""
        warnings = []
        
        # Check rooms sold vs rooms available
        if 'RoomsSold' in df.columns and 'RoomsAvailable' in df.columns:
            oversold = df['RoomsSold'] > df['RoomsAvailable']
            if oversold.any():
                oversold_count = oversold.sum()
                warnings.append(f"Business logic warning: {oversold_count} records show more rooms sold than available")
        
        # Check occupancy rate vs rooms calculation
        if all(col in df.columns for col in ['RoomsSold', 'RoomsAvailable', 'OccupancyRate']):
            calculated_occupancy = df['RoomsSold'] / df['RoomsAvailable']
            occupancy_diff = abs(calculated_occupancy - df['OccupancyRate'])
            significant_diff = occupancy_diff > 0.1  # 10% difference threshold
            
            if significant_diff.any():
                diff_count = significant_diff.sum()
                warnings.append(f"Business logic warning: {diff_count} records show significant difference between calculated and reported occupancy rate")
        
        # Check revenue calculations
        if all(col in df.columns for col in ['ADR', 'RoomsSold', 'TotalRevenue']):
            calculated_revenue = df['ADR'] * df['RoomsSold']
            revenue_diff = abs(calculated_revenue - df['TotalRevenue'])
            significant_diff = revenue_diff > (df['TotalRevenue'] * 0.1)  # 10% difference
            
            if significant_diff.any():
                diff_count = significant_diff.sum()
                warnings.append(f"Business logic warning: {diff_count} records show significant difference between calculated and reported revenue")
        
        # Check cost-to-revenue ratio
        if 'TotalCosts' in df.columns and 'TotalRevenue' in df.columns:
            cost_ratio = df['TotalCosts'] / df['TotalRevenue']
            high_cost_ratio = cost_ratio > 1.0  # Costs higher than revenue
            
            if high_cost_ratio.any():
                high_cost_count = high_cost_ratio.sum()
                warnings.append(f"Business logic warning: {high_cost_count} records show costs higher than revenue")
        
        return {
            'warnings': warnings
        }
    
    def _generate_validation_summary(self, original_df: pd.DataFrame, 
                                   cleaned_df: pd.DataFrame,
                                   errors: List[str], 
                                   warnings: List[str]) -> Dict[str, Any]:
        """Generate a comprehensive validation summary."""
        summary = {
            'original_shape': original_df.shape if original_df is not None else (0, 0),
            'cleaned_shape': cleaned_df.shape if cleaned_df is not None else (0, 0),
            'total_errors': len(errors),
            'total_warnings': len(warnings),
            'validation_timestamp': datetime.now().isoformat()
        }
        
        if original_df is not None and not original_df.empty:
            summary['original_columns'] = list(original_df.columns)
            summary['missing_data_summary'] = {
                col: original_df[col].isna().sum() 
                for col in original_df.columns
            }
        
        if cleaned_df is not None and not cleaned_df.empty:
            summary['cleaned_columns'] = list(cleaned_df.columns)
            summary['data_types'] = {
                col: str(cleaned_df[col].dtype) 
                for col in cleaned_df.columns
            }
        
        return summary
    
    def validate_excel_import(self, file_path: str) -> ValidationResult:
        """Validate Excel file import and handle common Excel artifacts.
        
        Args:
            file_path: Path to Excel file
            
        Returns:
            ValidationResult with validation status and cleaned data
        """
        errors = []
        warnings = []
        
        try:
            # Try to read Excel file
            df = pd.read_excel(file_path)
            
            # Handle common Excel artifacts
            df = self._clean_excel_artifacts(df)
            
            # Validate the cleaned data
            return self.validate_dataframe(df, auto_clean=True)
            
        except FileNotFoundError:
            errors.append(f"Excel file not found: {file_path}")
        except pd.errors.EmptyDataError:
            errors.append("Excel file is empty or contains no data")
        except Exception as e:
            errors.append(f"Error reading Excel file: {str(e)}")
        
        return ValidationResult(False, errors, warnings)
    
    def _clean_excel_artifacts(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean common Excel import artifacts."""
        if df.empty:
            return df
        
        cleaned_df = df.copy()
        
        # Remove completely empty rows and columns
        cleaned_df = cleaned_df.dropna(how='all').dropna(axis=1, how='all')
        
        # Handle merged cell artifacts (columns ending with .1, .2, etc.)
        columns_to_drop = []
        for col in cleaned_df.columns:
            if isinstance(col, str) and '.' in col and col.split('.')[-1].isdigit():
                base_col = col.split('.')[0]
                if base_col in cleaned_df.columns:
                    # Merge data from duplicate column if it contains non-null values
                    mask = cleaned_df[base_col].isna() & cleaned_df[col].notna()
                    if mask.any():
                        cleaned_df.loc[mask, base_col] = cleaned_df.loc[mask, col]
                    columns_to_drop.append(col)
        
        # Drop duplicate columns
        cleaned_df = cleaned_df.drop(columns=columns_to_drop)
        
        # Clean column names (remove extra spaces, special characters)
        cleaned_df.columns = [str(col).strip() for col in cleaned_df.columns]
        
        # Handle currency formatting (remove $ and commas)
        for col in cleaned_df.columns:
            if cleaned_df[col].dtype == 'object':
                # Check if column contains currency-formatted strings
                sample_values = cleaned_df[col].dropna().astype(str).head(10)
                if any('$' in str(val) or ',' in str(val) for val in sample_values):
                    # Clean currency formatting
                    cleaned_df[col] = cleaned_df[col].astype(str).str.replace('$', '').str.replace(',', '')
                    # Try to convert to numeric
                    cleaned_df[col] = pd.to_numeric(cleaned_df[col], errors='ignore')
        
        # Handle percentage formatting
        for col in cleaned_df.columns:
            if cleaned_df[col].dtype == 'object':
                sample_values = cleaned_df[col].dropna().astype(str).head(10)
                if any('%' in str(val) for val in sample_values):
                    # Clean percentage formatting and convert to decimal
                    cleaned_df[col] = cleaned_df[col].astype(str).str.replace('%', '')
                    cleaned_df[col] = pd.to_numeric(cleaned_df[col], errors='ignore') / 100
        
        return cleaned_df
    
    def get_validation_report(self, result: ValidationResult) -> str:
        """Generate a human-readable validation report.
        
        Args:
            result: ValidationResult to generate report for
            
        Returns:
            Formatted validation report string
        """
        report = []
        report.append("=" * 50)
        report.append("DATA VALIDATION REPORT")
        report.append("=" * 50)
        
        # Overall status
        status = "✅ PASSED" if result.is_valid else "❌ FAILED"
        report.append(f"Validation Status: {status}")
        report.append("")
        
        # Summary statistics
        if result.validation_summary:
            summary = result.validation_summary
            report.append("Summary:")
            report.append(f"  Original Data Shape: {summary.get('original_shape', 'N/A')}")
            report.append(f"  Cleaned Data Shape: {summary.get('cleaned_shape', 'N/A')}")
            report.append(f"  Total Errors: {summary.get('total_errors', 0)}")
            report.append(f"  Total Warnings: {summary.get('total_warnings', 0)}")
            report.append("")
        
        # Errors
        if result.errors:
            report.append("Errors:")
            for i, error in enumerate(result.errors, 1):
                report.append(f"  {i}. {error}")
            report.append("")
        
        # Warnings
        if result.warnings:
            report.append("Warnings:")
            for i, warning in enumerate(result.warnings, 1):
                report.append(f"  {i}. {warning}")
            report.append("")
        
        # Recommendations
        report.append("Recommendations:")
        if result.errors:
            report.append("  - Fix data errors before proceeding with analysis")
            report.append("  - Check data source for missing or corrupted values")
            report.append("  - Verify column names and data types")
        if result.warnings:
            report.append("  - Review warnings for potential data quality issues")
            report.append("  - Consider data cleaning or preprocessing steps")
        if result.is_valid:
            report.append("  - Data validation passed successfully")
            report.append("  - Data is ready for analysis")
        
        report.append("=" * 50)
        
        return "\n".join(report)


# Convenience functions for common validation scenarios
def validate_revenue_data(df: pd.DataFrame) -> ValidationResult:
    """Validate data for revenue analysis."""
    validator = DataValidator()
    required_columns = ['Date', 'TotalRevenue']
    return validator.validate_dataframe(df, required_columns=required_columns)


def validate_forecast_data(df: pd.DataFrame, target_column: str) -> ValidationResult:
    """Validate data for forecasting."""
    validator = DataValidator()
    required_columns = ['Date', target_column]
    result = validator.validate_dataframe(df, required_columns=required_columns)
    
    # Additional forecast-specific validation
    if result.is_valid and result.cleaned_data is not None:
        # Check for sufficient data points
        if len(result.cleaned_data) < 30:
            result.warnings.append(f"Insufficient data for forecasting: {len(result.cleaned_data)} points (recommended: 30+)")
        
        # Check for target column data availability
        target_data = result.cleaned_data[target_column].dropna()
        if len(target_data) < len(result.cleaned_data) * 0.8:  # Less than 80% data availability
            result.warnings.append(f"Target column '{target_column}' has significant missing data")
    
    return result


def validate_kpi_data(df: pd.DataFrame) -> ValidationResult:
    """Validate data for KPI calculations."""
    validator = DataValidator()
    required_columns = ['Date', 'TotalRevenue', 'OccupancyRate']
    return validator.validate_dataframe(df, required_columns=required_columns)


def validate_operations_data(df: pd.DataFrame) -> ValidationResult:
    """Validate data for operations analysis."""
    validator = DataValidator()
    required_columns = ['Date', 'TotalRevenue', 'TotalCosts', 'StaffCount']
    return validator.validate_dataframe(df, required_columns=required_columns)