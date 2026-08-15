import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import os

# Create directories if they don't exist
os.makedirs('images', exist_ok=True)
os.makedirs('models', exist_ok=True)
os.makedirs('outputs', exist_ok=True)

print("Loading data...")
df = pd.read_csv('data/student_performance.csv')

print(f"Initial shape: {df.shape}")

# Drop identifiers
df = df.drop(columns=['StudentID', 'Name'])

print("Data Cleaning (Handling Outliers)...")
# Let's cap outliers using the IQR method for AttendanceRate, StudyHoursPerWeek, PreviousGrade
def cap_outliers(series):
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return np.clip(series, lower_bound, upper_bound)

numerical_cols = ['AttendanceRate', 'StudyHoursPerWeek', 'PreviousGrade', 'ExtracurricularActivities']
for col in numerical_cols:
    df[col] = cap_outliers(df[col])

print("Generating EDA plots...")
# 1. Distribution of Final Grade
plt.figure(figsize=(8, 5))
sns.histplot(df['FinalGrade'], kde=True, color='blue')
plt.title('Distribution of Final Grades')
plt.savefig('images/final_grade_distribution.png')
plt.close()

# 2. Correlation Matrix
plt.figure(figsize=(10, 8))
numeric_df = df.select_dtypes(include=[np.number])
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.savefig('images/correlation_matrix.png')
plt.close()

# 3. Study Hours vs Final Grade
plt.figure(figsize=(8, 5))
sns.scatterplot(x='StudyHoursPerWeek', y='FinalGrade', hue='ParentalSupport', data=df)
plt.title('Study Hours vs Final Grade')
plt.savefig('images/study_hours_vs_grade.png')
plt.close()

print("Preprocessing and Feature Engineering...")
X = df.drop(columns=['FinalGrade', 'Gender'])
y = df['FinalGrade']

categorical_cols = ['ParentalSupport']
# numerical_cols already defined

from sklearn.linear_model import LinearRegression

preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numerical_cols),
        ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_cols)
    ])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training Models...")
rf_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                              ('scaler', StandardScaler()),
                              ('model', RandomForestRegressor(n_estimators=100, random_state=42))])

xgb_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('scaler', StandardScaler()),
                               ('model', XGBRegressor(n_estimators=100, random_state=42))])

lr_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                              ('scaler', StandardScaler()),
                              ('model', LinearRegression())])

rf_pipeline.fit(X_train, y_train)
xgb_pipeline.fit(X_train, y_train)
lr_pipeline.fit(X_train, y_train)

print("Evaluating Models...")
report_lines = ["Model Evaluation Report (Regression)", "=" * 36, ""]

def evaluate_model(model, name):
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Calculate Adjusted R2
    n = X_test.shape[0]
    # Number of features after preprocessing
    p = preprocessor.transform(X_test).shape[1]
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
    
    result_text = f"--- {name} ---\nRMSE: {rmse:.2f}\nMAE:  {mae:.2f}\nR2:   {r2:.2f}\nAdj R2: {adj_r2:.2f}\n"
    print(result_text)
    report_lines.append(result_text)
    return r2

rf_r2 = evaluate_model(rf_pipeline, "Random Forest")
xgb_r2 = evaluate_model(xgb_pipeline, "XGBoost")
lr_r2 = evaluate_model(lr_pipeline, "Linear Regression")

best_score = max(rf_r2, xgb_r2, lr_r2)
if best_score == lr_r2:
    best_pipeline = lr_pipeline
    best_name = "Linear Regression"
elif best_score == rf_r2:
    best_pipeline = rf_pipeline
    best_name = "Random Forest"
else:
    best_pipeline = xgb_pipeline
    best_name = "XGBoost"

report_lines.append(f"Best Model Selected: {best_name}")

# Save report to outputs folder
with open('outputs/model_evaluation_report.txt', 'w') as f:
    f.write('\n'.join(report_lines))
print("Evaluation report saved to outputs/model_evaluation_report.txt")

print(f"Saving best model ({best_name})...")
joblib.dump(best_pipeline, 'models/best_model.pkl')

print("Generating Feature Importance Plot...")
best_model = best_pipeline.named_steps['model']
preprocessor = best_pipeline.named_steps['preprocessor']

# Get feature names after one-hot encoding
cat_features = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_cols)
all_features = numerical_cols + list(cat_features)

# Extract importances or coefficients based on the model type
if hasattr(best_model, 'feature_importances_'):
    importances = best_model.feature_importances_
else:
    # Use absolute value of coefficients for linear models
    importances = np.abs(best_model.coef_)

indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 6))
sns.barplot(x=importances[indices], y=np.array(all_features)[indices], palette="viridis", hue=np.array(all_features)[indices], legend=False)
plt.title(f'Feature Importances ({best_name})')
plt.xlabel('Relative Importance (Absolute Coefficients)' if not hasattr(best_model, 'feature_importances_') else 'Relative Importance')
plt.tight_layout()
plt.savefig('images/feature_importance.png')
plt.close()

print("Pipeline completed successfully!")
