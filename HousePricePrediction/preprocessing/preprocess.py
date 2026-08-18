"""
House Price Prediction - Data Preprocessing Module

This script loads raw house price data, cleans it, prepares features for Linear Regression,
and saves the processed dataset.
"""

import pandas as pd
import numpy as np
import os
import sys
from typing import Tuple, List, Optional


def load_data(filepath: str) -> pd.DataFrame:
    """
    Load the dataset from CSV file.
    
    Args:
        filepath: Path to the CSV file
        
    Returns:
        Loaded DataFrame
    """
    print(f"Loading data from: {filepath}")
    df = pd.read_csv(filepath)
    print(f"Data loaded successfully. Shape: {df.shape}")
    return df


def inspect_data(df: pd.DataFrame) -> None:
    """
    Display comprehensive information about the dataset.
    
    Args:
        df: Input DataFrame
    """
    print("\n" + "="*60)
    print("DATASET INSPECTION")
    print("="*60)
    
    print(f"\n1. Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns")
    
    print(f"\n2. Column Names ({len(df.columns)}):")
    for i, col in enumerate(df.columns, 1):
        print(f"   {i}. {col}")
    
    print(f"\n3. Data Types:")
    print(df.dtypes.to_string())
    
    print(f"\n4. Missing Values per Column:")
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    missing_df = pd.DataFrame({'Missing Count': missing, 'Missing %': missing_pct})
    print(missing_df[missing_df['Missing Count'] > 0].to_string())
    if missing.sum() == 0:
        print("   No missing values found.")
    
    print(f"\n5. Duplicate Rows: {df.duplicated().sum()}")
    
    print(f"\n6. First 5 Rows:")
    print(df.head().to_string())
    
    print(f"\n7. Basic Statistics (Numerical Columns):")
    print(df.describe().to_string())


def identify_target_column(df: pd.DataFrame) -> str:
    """
    Identify the target column representing house price.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Name of the target column
    """
    # Common price column names (case-insensitive)
    price_keywords = ['price', 'cost', 'value', 'amount', 'sale_price', 'selling_price']
    
    # Check for exact matches first (case-insensitive)
    for col in df.columns:
        if col.lower() in price_keywords:
            print(f"\nTarget column identified: '{col}' (exact match)")
            return col
    
    # Check for partial matches
    for col in df.columns:
        col_lower = col.lower()
        for keyword in price_keywords:
            if keyword in col_lower:
                print(f"\nTarget column identified: '{col}' (partial match with '{keyword}')")
                return col
    
    # If no obvious price column, look for numeric columns that could be price
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        # Heuristic: price is usually a larger value column
        # Check which numeric column has the highest mean (likely price)
        means = df[numeric_cols].mean()
        likely_price = means.idxmax()
        print(f"\nTarget column inferred: '{likely_price}' (highest mean among numeric columns)")
        return likely_price
    
    raise ValueError("Could not identify a target price column in the dataset.")


def clean_data(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """
    Clean the dataset by handling duplicates and missing values.
    
    Args:
        df: Input DataFrame
        target_col: Name of the target column
        
    Returns:
        Cleaned DataFrame
    """
    print("\n" + "="*60)
    print("DATA CLEANING")
    print("="*60)
    
    original_shape = df.shape
    original_duplicates = df.duplicated().sum()
    
    # Remove duplicate rows
    df = df.drop_duplicates()
    duplicates_removed = original_duplicates - df.duplicated().sum()
    print(f"Removed {duplicates_removed} duplicate rows.")
    
    # Handle missing values in target column - remove rows where target is missing
    target_missing_before = df[target_col].isnull().sum()
    if target_missing_before > 0:
        df = df.dropna(subset=[target_col])
        print(f"Removed {target_missing_before} rows with missing target values.")
    
    # Handle missing values in features
    missing_before = df.isnull().sum().sum()
    
    # Separate numerical and categorical columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    # Remove target from numeric columns for imputation
    if target_col in numeric_cols:
        numeric_cols.remove(target_col)
    
    # Impute numerical columns with median
    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            print(f"  Imputed '{col}' with median: {median_val}")
    
    # Impute categorical columns with most frequent value
    for col in categorical_cols:
        if df[col].isnull().sum() > 0:
            mode_val = df[col].mode()
            if len(mode_val) > 0:
                df[col] = df[col].fillna(mode_val[0])
                print(f"  Imputed '{col}' with mode: {mode_val[0]}")
            else:
                df[col] = df[col].fillna('Unknown')
                print(f"  Imputed '{col}' with 'Unknown' (no mode found)")
    
    missing_after = df.isnull().sum().sum()
    print(f"Total missing values handled: {missing_before - missing_after}")
    
    # Validate target column has valid values
    if df[target_col].isnull().sum() > 0:
        raise ValueError(f"Target column '{target_col}' still has missing values after cleaning.")
    
    # Check for invalid target values (negative or zero prices)
    invalid_target = (df[target_col] <= 0).sum()
    if invalid_target > 0:
        print(f"Warning: Found {invalid_target} rows with invalid target values (<= 0). Removing them.")
        df = df[df[target_col] > 0]
    
    print(f"Cleaning complete. Shape: {original_shape} -> {df.shape}")
    return df


def prepare_features(df: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, List[str]]:
    """
    Prepare features for Linear Regression training.
    
    Args:
        df: Cleaned DataFrame
        target_col: Name of the target column
        
    Returns:
        Tuple of (processed DataFrame, list of feature column names)
    """
    print("\n" + "="*60)
    print("FEATURE PREPARATION")
    print("="*60)
    
    # Separate features and target
    feature_cols = [col for col in df.columns if col != target_col]
    
    # Identify column types
    numeric_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df[feature_cols].select_dtypes(include=['object']).columns.tolist()
    
    print(f"Numerical features ({len(numeric_cols)}): {numeric_cols}")
    print(f"Categorical features ({len(categorical_cols)}): {categorical_cols}")
    
    # Convert numerical columns to appropriate types
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        # Fill any NaN created during conversion with median
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].median())
    
    # Handle categorical features - One-hot encoding for Linear Regression
    if categorical_cols:
        print(f"\nApplying one-hot encoding to categorical features...")
        df = pd.get_dummies(df, columns=categorical_cols, drop_first=True, dtype=int)
        print(f"One-hot encoding complete. New shape: {df.shape}")
    
    # Get final feature columns (excluding target)
    final_feature_cols = [col for col in df.columns if col != target_col]
    
    print(f"\nFinal feature columns ({len(final_feature_cols)}):")
    for i, col in enumerate(final_feature_cols, 1):
        print(f"  {i}. {col}")
    
    return df, final_feature_cols


def save_processed_data(df: pd.DataFrame, output_path: str) -> None:
    """
    Save the processed dataset to CSV.
    
    Args:
        df: Processed DataFrame
        output_path: Output file path
    """
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    df.to_csv(output_path, index=False)
    print(f"\nProcessed data saved to: {output_path}")
    print(f"Final dataset shape: {df.shape}")


def print_summary(
    original_shape: Tuple[int, int],
    final_shape: Tuple[int, int],
    duplicates_removed: int,
    missing_handled: int,
    target_col: str,
    feature_cols: List[str],
    output_path: str
) -> None:
    """
    Print a comprehensive preprocessing summary.
    
    Args:
        original_shape: Original dataset shape
        final_shape: Final dataset shape
        duplicates_removed: Number of duplicates removed
        missing_handled: Number of missing values handled
        target_col: Target column name
        feature_cols: List of final feature columns
        output_path: Output file path
    """
    print("\n" + "="*60)
    print("PREPROCESSING SUMMARY")
    print("="*60)
    print(f"Original dataset shape:     {original_shape[0]} rows x {original_shape[1]} columns")
    print(f"Final dataset shape:        {final_shape[0]} rows x {final_shape[1]} columns")
    print(f"Rows removed:               {original_shape[0] - final_shape[0]}")
    print(f"Duplicates removed:         {duplicates_removed}")
    print(f"Missing values handled:     {missing_handled}")
    print(f"Selected target column:     {target_col}")
    print(f"Final feature columns:      {len(feature_cols)}")
    print(f"Output file path:           {output_path}")
    print("="*60)


def find_data_file() -> str:
    """
    Find the appropriate data file to process.
    
    Returns:
        Path to the data file
    """
    # Priority order for data files
    possible_paths = [
        "data/raw/house_prices.csv",
        "data/raw/Bangalore house data.csv",
        "data/raw/Delhi house data.csv",
        "data/raw/Pune house data.csv",
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            return path
    
    raise FileNotFoundError(
        "No data file found. Expected one of: " + ", ".join(possible_paths)
    )


def main():
    """Main preprocessing pipeline."""
    print("="*60)
    print("HOUSE PRICE PREDICTION - DATA PREPROCESSING")
    print("="*60)
    
    # Find and load data
    data_path = find_data_file()
    print(f"Found data file: {data_path}")
    
    df = load_data(data_path)
    original_shape = df.shape
    
    # Inspect data
    inspect_data(df)
    
    # Identify target column
    target_col = identify_target_column(df)
    
    # Clean data
    df_cleaned = clean_data(df.copy(), target_col)
    
    # Prepare features
    df_processed, feature_cols = prepare_features(df_cleaned, target_col)
    final_shape = df_processed.shape
    
    # Calculate metrics
    duplicates_removed = original_shape[0] - df_cleaned.shape[0]
    missing_handled = df.isnull().sum().sum() - df_processed.isnull().sum().sum()
    
    # Save processed data
    output_path = "data/processed/cleaned_house_prices.csv"
    save_processed_data(df_processed, output_path)
    
    # Print summary
    print_summary(
        original_shape=original_shape,
        final_shape=final_shape,
        duplicates_removed=duplicates_removed,
        missing_handled=missing_handled,
        target_col=target_col,
        feature_cols=feature_cols,
        output_path=output_path
    )
    
    print("\n✅ Preprocessing completed successfully!")
    return df_processed, target_col, feature_cols


if __name__ == "__main__":
    main()