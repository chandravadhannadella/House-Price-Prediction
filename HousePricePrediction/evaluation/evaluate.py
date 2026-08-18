"""
House Price Prediction - Model Evaluation Module

This script evaluates the trained Linear Regression model on test data,
calculates evaluation metrics, and generates reports and predictions.
"""

import pandas as pd
import os
import sys
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import joblib
import numpy as np


def load_data(filepath: str) -> pd.DataFrame:
    """
    Load the processed dataset from CSV file.
    
    Args:
        filepath: Path to the processed CSV file
        
    Returns:
        Loaded DataFrame
        
    Raises:
        FileNotFoundError: If the processed dataset file doesn't exist
        ValueError: If the dataset is empty
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Processed dataset not found at: {filepath}")
    
    try:
        df = pd.read_csv(filepath)
        print(f"✓ Loaded processed dataset from: {filepath}")
        print(f"  Dataset shape: {df.shape}")
        
        if df.empty:
            raise ValueError("Loaded dataset is empty.")
        
        return df
    
    except pd.errors.EmptyDataError:
        raise ValueError(f"Processed dataset at {filepath} is empty.")
    except Exception as e:
        raise Exception(f"Error loading processed dataset: {str(e)}")


def load_model(filepath: str):
    """
    Load the trained model from a pickle file.
    
    Args:
        filepath: Path to the trained model pickle file
        
    Returns:
        Loaded model object
        
    Raises:
        FileNotFoundError: If the model file doesn't exist
        Exception: If model loading fails
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Trained model not found at: {filepath}")
    
    try:
        model = joblib.load(filepath)
        print(f"✓ Loaded trained model from: {filepath}")
        print(f"  Model type: {type(model).__name__}")
        return model
    
    except Exception as e:
        raise Exception(f"Error loading trained model: {str(e)}")


def identify_target_column(df: pd.DataFrame) -> str:
    """
    Identify the target column from the processed dataset.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Name of the target column
        
    Raises:
        ValueError: If no target column can be identified
    """
    price_keywords = ['price', 'cost', 'value', 'amount', 'sale_price', 'selling_price']
    
    # Check for exact matches first (case-insensitive)
    for col in df.columns:
        if col.lower() in price_keywords:
            print(f"✓ Target column identified: '{col}'")
            return col
    
    # Check for partial matches
    for col in df.columns:
        col_lower = col.lower()
        for keyword in price_keywords:
            if keyword in col_lower:
                print(f"✓ Target column identified: '{col}'")
                return col
    
    raise ValueError(
        "Could not identify target column. Expected a column with 'price' in the name. "
        f"Available columns: {df.columns.tolist()}"
    )


def prepare_test_data(df: pd.DataFrame, target_col: str) -> tuple:
    """
    Prepare the test dataset with the same train/test split used during training.
    
    This function recreates the exact same split configuration to ensure
    consistency with the model training process.
    
    Args:
        df: Input DataFrame (complete processed dataset)
        target_col: Name of the target column
        
    Returns:
        Tuple of (X_test, y_test) - test features and target values
        
    Raises:
        ValueError: If target column doesn't exist or if no features remain
    """
    if target_col not in df.columns:
        raise ValueError(
            f"Target column '{target_col}' not found in dataset. "
            f"Available columns: {df.columns.tolist()}"
        )
    
    # Separate features and target
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    if X.empty or len(X.columns) == 0:
        raise ValueError("No features available after removing target column.")
    
    print(f"✓ Features shape: {X.shape}")
    print(f"✓ Target shape: {y.shape}")
    
    # Recreate the same train/test split as used during training
    # This is crucial for consistent evaluation
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42
    )
    
    print(f"✓ Train/test split completed (same configuration as training):")
    print(f"  Training samples: {len(X_train)}")
    print(f"  Testing samples: {len(X_test)}")
    
    return X_test, y_test


def evaluate_model(model, X_test, y_test) -> dict:
    """
    Evaluate the trained model on test data and calculate metrics.
    
    Args:
        model: Trained model object
        X_test: Test features
        y_test: Test target values
        
    Returns:
        Dictionary containing all evaluation metrics
        
    Raises:
        Exception: If prediction or metric calculation fails
    """
    print("\n" + "="*60)
    print("MODEL EVALUATION")
    print("="*60)
    
    try:
        # Generate predictions
        y_pred = model.predict(X_test)
        print(f"✓ Generated predictions for {len(y_pred)} test samples")
        
        # Calculate evaluation metrics
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        
        print(f"✓ Calculated evaluation metrics:")
        print(f"  R² Score: {r2:.6f}")
        print(f"  MAE: {mae:,.2f}")
        print(f"  MSE: {mse:,.2f}")
        print(f"  RMSE: {rmse:,.2f}")
        
        # Return metrics and predictions
        metrics = {
            'r2_score': r2,
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'y_pred': y_pred,
            'y_test': y_test,
            'num_test_samples': len(y_test)
        }
        
        return metrics
    
    except Exception as e:
        raise Exception(f"Error during model evaluation: {str(e)}")


def save_evaluation_report(
    metrics: dict,
    target_col: str,
    output_path: str,
    dataset_path: str = "data/processed/cleaned_house_prices.csv"
) -> None:
    """
    Save the evaluation report to a text file.
    
    Args:
        metrics: Dictionary containing evaluation metrics
        target_col: Name of the target column
        output_path: Path where the report should be saved
        dataset_path: Path to the processed dataset
        
    Raises:
        Exception: If report saving fails
    """
    try:
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Generate report content
        report = []
        report.append("="*60)
        report.append("HOUSE PRICE PREDICTION - MODEL EVALUATION REPORT")
        report.append("="*60)
        report.append("")
        report.append(f"Evaluation Date/Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        report.append("-"*60)
        report.append("PROJECT INFORMATION")
        report.append("-"*60)
        report.append(f"Project Name: House Price Prediction")
        report.append(f"Model Name: Linear Regression")
        report.append(f"Dataset Path: {dataset_path}")
        report.append("")
        report.append("-"*60)
        report.append("EVALUATION DETAILS")
        report.append("-"*60)
        report.append(f"Target Column: {target_col}")
        report.append(f"Number of Test Samples: {metrics['num_test_samples']}")
        report.append("")
        report.append("-"*60)
        report.append("EVALUATION METRICS")
        report.append("-"*60)
        report.append(f"R² Score: {metrics['r2_score']:.6f}")
        report.append(f"Mean Absolute Error (MAE): {metrics['mae']:,.2f}")
        report.append(f"Mean Squared Error (MSE): {metrics['mse']:,.2f}")
        report.append(f"Root Mean Squared Error (RMSE): {metrics['rmse']:,.2f}")
        report.append("")
        report.append("-"*60)
        report.append("METRIC INTERPRETATION")
        report.append("-"*60)
        report.append(f"R² Score (Coefficient of Determination):")
        report.append(f"  Value: {metrics['r2_score']:.6f} (range: 0 to 1)")
        report.append(f"  Interpretation: Explains {metrics['r2_score']*100:.2f}% of variance in house prices")
        report.append("")
        report.append(f"MAE (Mean Absolute Error):")
        report.append(f"  Value: {metrics['mae']:,.2f}")
        report.append(f"  Interpretation: Average prediction error (absolute)")
        report.append("")
        report.append(f"MSE (Mean Squared Error):")
        report.append(f"  Value: {metrics['mse']:,.2f}")
        report.append(f"  Interpretation: Average squared prediction error")
        report.append("")
        report.append(f"RMSE (Root Mean Squared Error):")
        report.append(f"  Value: {metrics['rmse']:,.2f}")
        report.append(f"  Interpretation: Square root of MSE, same units as target")
        report.append("")
        report.append("="*60)
        
        # Write report to file
        with open(output_path, 'w') as f:
            f.write('\n'.join(report))
        
        print(f"\n✓ Evaluation report saved to: {output_path}")
    
    except Exception as e:
        raise Exception(f"Error saving evaluation report: {str(e)}")


def save_predictions(
    metrics: dict,
    output_path: str
) -> None:
    """
    Save actual and predicted prices to a CSV file.
    
    Args:
        metrics: Dictionary containing predictions and actual values
        output_path: Path where the predictions CSV should be saved
        
    Raises:
        Exception: If prediction saving fails
    """
    try:
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Create DataFrame with actual and predicted values
        predictions_df = pd.DataFrame({
            'Actual_Price': metrics['y_test'].values,
            'Predicted_Price': metrics['y_pred']
        })
        
        # Add additional columns for analysis
        predictions_df['Difference'] = predictions_df['Actual_Price'] - predictions_df['Predicted_Price']
        predictions_df['Absolute_Error'] = predictions_df['Difference'].abs()
        predictions_df['Percentage_Error'] = (predictions_df['Absolute_Error'] / predictions_df['Actual_Price'] * 100).round(2)
        
        # Save to CSV
        predictions_df.to_csv(output_path, index=False)
        
        print(f"✓ Predictions saved to: {output_path}")
        print(f"  Total predictions: {len(predictions_df)}")
        print(f"  Columns: {', '.join(predictions_df.columns.tolist())}")
    
    except Exception as e:
        raise Exception(f"Error saving predictions: {str(e)}")


def print_evaluation_summary(metrics: dict) -> None:
    """
    Print a formatted evaluation summary to the console.
    
    Args:
        metrics: Dictionary containing evaluation metrics
    """
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    print(f"Model Evaluation Results")
    print("-"*60)
    print(f"R² Score: {metrics['r2_score']:.6f}")
    print(f"MAE: {metrics['mae']:,.2f}")
    print(f"MSE: {metrics['mse']:,.2f}")
    print(f"RMSE: {metrics['rmse']:,.2f}")
    print(f"Test Samples Evaluated: {metrics['num_test_samples']}")
    print("="*60 + "\n")


def main():
    """
    Main function to orchestrate the entire evaluation pipeline.
    """
    try:
        # Define paths
        processed_data_path = "data/processed/cleaned_house_prices.csv"
        model_path = "models/house_price_model.pkl"
        report_output_path = "outputs/evaluation_report.txt"
        predictions_output_path = "outputs/predictions.csv"
        
        # Step 1: Load processed data
        print("="*60)
        print("STEP 1: LOADING PROCESSED DATA")
        print("="*60)
        df = load_data(processed_data_path)
        
        # Step 2: Load trained model
        print("\n" + "="*60)
        print("STEP 2: LOADING TRAINED MODEL")
        print("="*60)
        model = load_model(model_path)
        
        # Step 3: Identify target column
        print("\n" + "="*60)
        print("STEP 3: IDENTIFYING TARGET COLUMN")
        print("="*60)
        target_col = identify_target_column(df)
        
        # Step 4: Prepare test data (recreate same split as training)
        print("\n" + "="*60)
        print("STEP 4: PREPARING TEST DATA")
        print("="*60)
        X_test, y_test = prepare_test_data(df, target_col)
        
        # Step 5: Evaluate model
        metrics = evaluate_model(model, X_test, y_test)
        
        # Step 6: Save evaluation report
        print("\n" + "="*60)
        print("STEP 5: SAVING EVALUATION REPORT")
        print("="*60)
        save_evaluation_report(metrics, target_col, report_output_path, processed_data_path)
        
        # Step 7: Save predictions
        print("\n" + "="*60)
        print("STEP 6: SAVING PREDICTIONS")
        print("="*60)
        save_predictions(metrics, predictions_output_path)
        
        # Step 8: Print summary
        print_evaluation_summary(metrics)
        
        print("✓ MODEL EVALUATION COMPLETED SUCCESSFULLY")
        print(f"✓ Report saved to: {report_output_path}")
        print(f"✓ Predictions saved to: {predictions_output_path}")
        
    except FileNotFoundError as e:
        print(f"❌ Error: {str(e)}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"❌ Validation Error: {str(e)}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected Error: {str(e)}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
