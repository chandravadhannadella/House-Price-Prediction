"""
House Price Prediction - Model Training Module

This script loads the preprocessed dataset, trains a Linear Regression model,
and saves it for later use in evaluation and deployment.
"""

import pandas as pd
import os
import sys
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib


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


def identify_target_column(df: pd.DataFrame) -> str:
    """
    Identify the target column from the processed dataset.
    
    The target column should be identified during preprocessing.
    This function looks for the price column by common naming patterns.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Name of the target column
        
    Raises:
        ValueError: If no target column can be identified
    """
    # Common price column names (case-insensitive)
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


def prepare_data(df: pd.DataFrame, target_col: str) -> tuple:
    """
    Separate the dataset into features (X) and target (y).
    
    Args:
        df: Input DataFrame
        target_col: Name of the target column
        
    Returns:
        Tuple of (X, y) where X is features DataFrame and y is target Series
        
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
    
    print(f"✓ Features (X) shape: {X.shape}")
    print(f"✓ Target (y) shape: {y.shape}")
    
    return X, y


def split_data(X, y, test_size: float = 0.2, random_state: int = 42) -> tuple:
    """
    Split the dataset into training and testing sets.
    
    Args:
        X: Features DataFrame
        y: Target Series
        test_size: Proportion of data for testing (default 0.2 = 20%)
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=random_state
    )
    
    print(f"✓ Data split completed:")
    print(f"  Training samples: {len(X_train)} ({(len(X_train)/len(X)*100):.1f}%)")
    print(f"  Testing samples:  {len(X_test)} ({(len(X_test)/len(X)*100):.1f}%)")
    
    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train) -> LinearRegression:
    """
    Train a Linear Regression model on the training data.
    
    Args:
        X_train: Training features
        y_train: Training target values
        
    Returns:
        Trained LinearRegression model
    """
    print("\n" + "="*60)
    print("MODEL TRAINING")
    print("="*60)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    print("✓ Linear Regression model trained successfully")
    print(f"  Model coefficients shape: {model.coef_.shape}")
    print(f"  Model intercept: {model.intercept_:.2f}")
    
    return model


def save_model(model: LinearRegression, output_path: str) -> None:
    """
    Save the trained model to a pickle file using joblib.
    
    Args:
        model: Trained model
        output_path: Path where the model should be saved
        
    Raises:
        Exception: If model saving fails
    """
    try:
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        joblib.dump(model, output_path)
        print(f"\n✓ Model saved successfully: {output_path}")
        
        # Verify the model can be loaded back
        loaded_model = joblib.load(output_path)
        print(f"✓ Model verification: Successfully loaded from {output_path}")
        
    except Exception as e:
        raise Exception(f"Error saving model: {str(e)}")


def print_training_summary(
    df_shape: tuple,
    num_features: int,
    target_col: str,
    num_train_samples: int,
    num_test_samples: int,
    model_name: str,
    model_path: str
) -> None:
    """
    Print a comprehensive training summary.
    
    Args:
        df_shape: Original dataset shape
        num_features: Number of input features
        target_col: Name of target column
        num_train_samples: Number of training samples
        num_test_samples: Number of testing samples
        model_name: Name of the model
        model_path: Path where model is saved
    """
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    print(f"Dataset shape: {df_shape}")
    print(f"Number of features: {num_features}")
    print(f"Target column: '{target_col}'")
    print(f"Training samples: {num_train_samples}")
    print(f"Testing samples: {num_test_samples}")
    print(f"Model: {model_name}")
    print(f"Model saved successfully: {model_path}")
    print("="*60 + "\n")


def main():
    """
    Main function to orchestrate the entire training pipeline.
    """
    try:
        # Define paths
        processed_data_path = "data/processed/cleaned_house_prices.csv"
        model_output_path = "models/house_price_model.pkl"
        
        # Step 1: Load processed data
        print("="*60)
        print("STEP 1: LOADING PROCESSED DATA")
        print("="*60)
        df = load_data(processed_data_path)
        original_shape = df.shape
        
        # Step 2: Identify target column
        print("\n" + "="*60)
        print("STEP 2: IDENTIFYING TARGET COLUMN")
        print("="*60)
        target_col = identify_target_column(df)
        
        # Step 3: Prepare data (separate X and y)
        print("\n" + "="*60)
        print("STEP 3: PREPARING DATA")
        print("="*60)
        X, y = prepare_data(df, target_col)
        num_features = X.shape[1]
        
        # Step 4: Split data into train and test sets
        print("\n" + "="*60)
        print("STEP 4: SPLITTING DATA")
        print("="*60)
        X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2, random_state=42)
        num_train_samples = len(X_train)
        num_test_samples = len(X_test)
        
        # Step 5: Train the model
        model = train_model(X_train, y_train)
        
        # Step 6: Save the model
        print("\n" + "="*60)
        print("STEP 5: SAVING MODEL")
        print("="*60)
        save_model(model, model_output_path)
        
        # Step 7: Print training summary
        print_training_summary(
            df_shape=original_shape,
            num_features=num_features,
            target_col=target_col,
            num_train_samples=num_train_samples,
            num_test_samples=num_test_samples,
            model_name="Linear Regression",
            model_path=model_output_path
        )
        
        print("✓ MODEL TRAINING COMPLETED SUCCESSFULLY")
        print(f"✓ Next step: Implement evaluation in evaluation/evaluate.py")
        
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
