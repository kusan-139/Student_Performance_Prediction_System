# Student Performance Prediction System

An industry-oriented machine learning project that predicts student performance and serves the model via an interactive, dynamically-themed Flask dashboard.

## Overview

This project aims to predict a student's final grade based on various factors such as attendance rate, study hours, previous grades, extracurricular activities, and parental support. It involves a complete machine learning pipeline, from exploratory data analysis (EDA) and data preprocessing to model training and deployment.

### Key Features
- **Exploratory Data Analysis (EDA):** Jupyter notebook for analyzing the dataset, handling missing values, capping outliers, and visualizing relationships.
- **Machine Learning Pipeline:** Trains both Random Forest and XGBoost regressors, selecting the best model based on R² score. Features are preprocessed using `StandardScaler` and `OneHotEncoder`.
- **Interactive Web Dashboard:** A Flask-based web application with a modern UI offering multiple themes (Glassmorphism, Beige Editorial, Cyberpunk, Neo-Brutalism).
- **Prediction Modes:**
  - **Single Prediction:** Input manual details to predict the grade for an individual student. Includes a feature to load a random student from the dataset.
  - **Bulk Prediction:** Upload a CSV file to predict grades for multiple students at once and export the results.

## Project Structure

```
├── data/
│   ├── sample_test_20.csv      # Sample data for batch prediction
│   └── student_performance.csv # Main dataset
├── images/                     # Generated EDA and feature importance plots
├── models/
│   └── best_model.pkl          # Trained model (Random Forest)
├── notebooks/
│   └── EDA_and_Preprocessing.ipynb # EDA notebook
├── outputs/
│   └── model_evaluation_report.txt # Evaluation metrics report
├── src/
│   └── train_and_eda.py        # Main training script
├── web/
│   ├── app.py                  # Flask application
│   ├── static/                 # CSS and JS files
│   └── templates/              # HTML templates
├── requirements.txt            # Project dependencies
└── README.md
```

## Setup and Installation

1. **Clone the repository:**
   ```bash
   git clone <repository_url>
   cd "Student Performance Prediction System"
   ```

2. **Install dependencies:**
   It is recommended to use a virtual environment.
   ```bash
   pip install -r requirements.txt
   ```

3. **Train the Model (Optional):**
   If you want to retrain the model or regenerate plots:
   ```bash
   python src/train_and_eda.py
   ```

4. **Run the Web App:**
   ```bash
   python web/app.py
   ```
   Open your browser and navigate to `http://127.0.0.1:5000/`.

## Model Evaluation

The pipeline trains Random Forest and XGBoost models. The best model (currently Random Forest) is saved and evaluated using RMSE, MAE, and R² metrics. The evaluation report is saved in `outputs/model_evaluation_report.txt`.

---
<div align="center">
  <p>Built with ❤️ by <b>Kusan Chakraborty</b></p>
  <p>GitHub: <a href="https://github.com/kusan-139">@kusan-139</a></p>
</div>
