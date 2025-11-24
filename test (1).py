"""
Test the trained DecisionTreeClassifier model.

This script:
1. Loads the saved model from savedmodel.pth
2. Loads the test dataset
3. Computes and displays the test accuracy
"""

import numpy as np
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os

def load_test_data():
    """
    Load Olivetti faces dataset and return the test split.
    
    Returns:
        X_test, y_test: Test datasets
    """
    print("Loading Olivetti faces dataset for testing...")
    
    # Load the same dataset with same random state
    faces = fetch_olivetti_faces(shuffle=True, random_state=42)
    X = faces.data
    y = faces.target
    
    # Use the same split as training (70-30 split with same random_state)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.3,
        random_state=42,
        stratify=y
    )
    
    print(f"Test set loaded: {X_test.shape[0]} images")
    
    return X_test, y_test

def load_model(filename='savedmodel.pth'):
    """
    Load the saved model.
    
    Args:
        filename: Name of the saved model file
    
    Returns:
        Loaded model
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Model file '{filename}' not found. Please run train.py first.")
    
    print(f"\nLoading model from '{filename}'...")
    model = joblib.load(filename)
    print("Model loaded successfully!")
    
    return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model on test data and display results.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
    """
    print("\n" + "="*60)
    print("Model Evaluation on Test Set")
    print("="*60)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate test accuracy
    test_accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n📊 Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    
    # Additional metrics
    print(f"\n📈 Performance Metrics:")
    print(f"   - Correct predictions: {np.sum(y_pred == y_test)} out of {len(y_test)}")
    print(f"   - Incorrect predictions: {np.sum(y_pred != y_test)}")
    
    # Show detailed classification report
    print("\n📋 Detailed Classification Report:")
    print("-" * 60)
    report = classification_report(y_test, y_pred, zero_division=0)
    print(report)
    
    return test_accuracy

if __name__ == "__main__":
    try:
        print("="*60)
        print("ML Model Testing Pipeline")
        print("="*60)
        
        # Step 1: Load the test data
        X_test, y_test = load_test_data()
        
        # Step 2: Load the saved model
        model = load_model()
        
        # Step 3: Evaluate the model
        test_accuracy = evaluate_model(model, X_test, y_test)
        
        print("\n" + "="*60)
        print("Testing pipeline completed successfully!")
        print("="*60)
        
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("Please run train.py first to generate the model file.")
    except Exception as e:
        print(f"\n❌ An error occurred: {e}")
