import os
from flask import Flask, render_template, request, jsonify, send_file
import pandas as pd
import joblib
import io
import random

app = Flask(__name__)

# Load the trained model
MODEL_PATH = os.path.join(os.path.dirname(__file__), '../models/best_model.pkl')
DATA_PATH = os.path.join(os.path.dirname(__file__), '../data/student_performance.csv')
model = None

if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict_single', methods=['POST'])
def predict_single():
    if not model:
        return jsonify({'error': 'Model not trained yet.'}), 500
    
    data = request.json
    # Convert incoming JSON to DataFrame
    df = pd.DataFrame([data])
    
    try:
        raw_prediction = model.predict(df)[0]
        # Clamp prediction between 0 and 100
        prediction = max(0.0, min(100.0, raw_prediction))
        return jsonify({'prediction': round(prediction, 2)})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/random_student', methods=['GET'])
def random_student():
    try:
        df = pd.read_csv(DATA_PATH)
        # Drop ID columns for frontend population
        df = df.drop(columns=['StudentID', 'Name', 'FinalGrade', 'Gender'], errors='ignore')
        random_row = df.sample(1).iloc[0].to_dict()
        return jsonify(random_row)
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    if not model:
        return jsonify({'error': 'Model not trained yet.'}), 500
        
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
        
    if file:
        try:
            df = pd.read_csv(file)
            # Store original for merging
            original_df = df.copy()
            
            # Preprocess if needed
            df_features = df.drop(columns=['StudentID', 'Name', 'FinalGrade', 'Gender'], errors='ignore')
            
            predictions = model.predict(df_features)
            # Clamp predictions between 0 and 100
            clamped_predictions = [max(0.0, min(100.0, p)) for p in predictions]
            original_df['Predicted_FinalGrade'] = [round(p, 2) for p in clamped_predictions]
            
            # Return JSON for table rendering
            return jsonify({'data': original_df.to_dict(orient='records')})
        except Exception as e:
            return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)
