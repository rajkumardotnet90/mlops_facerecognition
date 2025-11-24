"""
Train a DecisionTreeClassifier on the Olivetti faces dataset.

This script:
1. Loads the Olivetti faces dataset from sklearn
2. Splits data into 70% train and 30% test
3. Trains a DecisionTreeClassifier
4. Saves the trained model using joblib
"""

import numpy as np
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import joblib

def load_and_split_data():
    """
    Load Olivetti faces dataset and split into train/test sets.
    
    Returns:
        X_train, X_test, y_train, y_test: Split datasets
    """
    print("Loading Olivetti faces dataset...")
    # Load the Olivetti faces dataset
    # This dataset contains 400 images of 40 people (10 images per person)
    # Each image is 64x64 pixels
    faces = fetch_olivetti_faces(shuffle=True, random_state=42)
    
    X = faces.data  # Features: flattened 64x64 images (4096 features per image)
    y = faces.target  # Labels: person ID (0-39)
    
    print(f"Dataset loaded: {X.shape[0]} images, {X.shape[1]} features per image")
    print(f"Number of classes (people): {len(np.unique(y))}")
    
    # Split into 70% training and 30% testing
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.3,  # 30% for testing
        random_state=42,  # For reproducibility
        stratify=y  # Ensure each class is proportionally represented
    )
    
    print(f"Training set size: {X_train.shape[0]} images")
    print(f"Test set size: {X_test.shape[0]} images")
    
    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train):
    """
    Train a DecisionTreeClassifier model.
    
    Args:
        X_train: Training features
        y_train: Training labels
    
    Returns:
        Trained model
    """
    print("\nTraining DecisionTreeClassifier...")
    
    # Create a Decision Tree Classifier
    # max_depth limits tree depth to prevent overfitting
    # random_state ensures reproducibility
    model = DecisionTreeClassifier(
        max_depth=20,  # Increased depth for better learning
        random_state=42,
        min_samples_split=2,  # Minimum samples required to split a node
        min_samples_leaf=1  # Minimum samples required at leaf node
    )
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Calculate training accuracy
    train_accuracy = model.score(X_train, y_train)
    print(f"Training completed!")
    print(f"Training accuracy: {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
    
    return model

def save_model(model, filename='savedmodel.pth'):
    """
    Save the trained model using joblib.
    
    Args:
        model: Trained model to save
        filename: Name of the file to save the model
    """
    print(f"\nSaving model as '{filename}'...")
    joblib.dump(model, filename)
    print(f"Model saved successfully!")

if __name__ == "__main__":
    # Main execution flow
    print("="*60)
    print("ML Model Training Pipeline")
    print("="*60)
    
    # Step 1: Load and split data
    X_train, X_test, y_train, y_test = load_and_split_data()
    
    # Step 2: Train the model
    model = train_model(X_train, y_train)
    
    # Step 3: Save the model
    save_model(model)
    
    print("\n" + "="*60)
    print("Training pipeline completed successfully!")
    print("="*60)
