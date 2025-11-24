"""
Flask Web Application for Face Recognition

This application:
1. Serves a web page to upload face images
2. Uses the trained DecisionTreeClassifier model to predict the person
3. Displays the predicted class (person ID) to the user
"""

from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
from PIL import Image
import io
import os

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
MODEL_PATH = 'savedmodel.pth'
model = None

def load_model():
    """Load the trained model at startup"""
    global model
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        print(f"✓ Model loaded successfully from {MODEL_PATH}")
    else:
        print(f"✗ Error: Model file '{MODEL_PATH}' not found!")
        print("Please run train.py first to generate the model.")

def preprocess_image(image_file):
    """
    Preprocess uploaded image to match model's expected input.
    
    The model expects:
    - Grayscale image
    - 64x64 pixels
    - Flattened to 4096 features
    - Pixel values normalized to [0, 1]
    
    Args:
        image_file: Uploaded image file
    
    Returns:
        Preprocessed image as numpy array
    """
    # Read image
    img = Image.open(io.BytesIO(image_file.read()))
    
    # Convert to grayscale
    img = img.convert('L')
    
    # Resize to 64x64 (same as training data)
    img = img.resize((64, 64))
    
    # Convert to numpy array
    img_array = np.array(img)
    
    # Flatten to 1D array (4096 features)
    img_flat = img_array.flatten()
    
    # Normalize pixel values to [0, 1]
    img_normalized = img_flat / 255.0
    
    # Reshape to (1, 4096) for model prediction
    img_reshaped = img_normalized.reshape(1, -1)
    
    return img_reshaped

@app.route('/')
def home():
    """Home page route"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """
    Handle image upload and prediction.
    
    Returns:
        JSON response with predicted person ID
    """
    try:
        # Check if model is loaded
        if model is None:
            return jsonify({
                'success': False,
                'error': 'Model not loaded. Please contact administrator.'
            }), 500
        
        # Check if file was uploaded
        if 'file' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No file uploaded'
            }), 400
        
        file = request.files['file']
        
        # Check if filename is empty
        if file.filename == '':
            return jsonify({
                'success': False,
                'error': 'No file selected'
            }), 400
        
        # Preprocess the image
        img_array = preprocess_image(file)
        
        # Make prediction
        prediction = model.predict(img_array)
        predicted_class = int(prediction[0])
        
        # Get prediction probability (confidence)
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(img_array)
            confidence = float(np.max(probabilities) * 100)
        else:
            confidence = None
        
        # Return prediction result
        return jsonify({
            'success': True,
            'predicted_person': predicted_class,
            'confidence': f"{confidence:.2f}%" if confidence else "N/A"
        })
    
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Prediction failed: {str(e)}'
        }), 500

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None
    })

if __name__ == '__main__':
    # Load model before starting server
    load_model()
    
    # Start Flask server
    # host='0.0.0.0' allows external access (needed for Docker)
    # port=5000 is the default Flask port
    print("\n" + "="*60)
    print("Starting Face Recognition Web Application")
    print("="*60)
    print("Server running on: http://0.0.0.0:5000")
    print("Press Ctrl+C to stop the server")
    print("="*60 + "\n")
    
    app.run(host='0.0.0.0', port=5000, debug=True)
