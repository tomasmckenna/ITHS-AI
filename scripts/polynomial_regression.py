import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import OrdinalEncoder, PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load filter dataset
file_path = "/Users/tomas/Documents/GitHub/AImed/data/df.xlsx"
df = pd.read_excel(file_path)
df = df[df['Information'] == 'match']

# Define features
df_features = df[['Activities.ActivityCategory', 'Activities.doubleStaffing', 'gender', 
                  'ageSpan', 'SpanDistanceM', 'SpanCarStartTime']]
target = df['DurationMin']

# Target encoding Activities.ActivityCategory
activity_category_means = df.groupby('Activities.ActivityCategory')['DurationMin'].mean()
df_features['Activities.ActivityCategory'] = df_features['Activities.ActivityCategory'].map(activity_category_means)
df_features = df_features.drop(columns=['Activities.ActivityCategory'])

# One-hot
categorical_features = ['Activities.doubleStaffing', 'gender', 'ageSpan', 'SpanDistanceM', 'SpanCarStartTime']
df_features = pd.get_dummies(df_features, columns=categorical_features, drop_first=True)

# # Check for NaN
# if df_features.isnull().any().any():
#     print("Features contain NaN")

# if target.isnull().any():
#     print("Target contains NaN")

# # Scale features
# scaler = StandardScaler()
# features_scaled = scaler.fit_transform(features)

# polynomial 
poly = PolynomialFeatures(degree=2, include_bias=False)  # You can adjust the degree here
features_poly = poly.fit_transform(df_features)

# Split train 
#  test sets
x_train, x_test, y_train, y_test = train_test_split(features_poly, target, test_size=0.2, random_state=42)

# Fit polynomial regression
model = LinearRegression()
model.fit(x_train, y_train)

# Evaluate
y_pred = model.predict(x_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("------ Polynomial regression ------")
print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")

# Cross-validation
cv_scores = cross_val_score(model, x_train, y_train, cv=5, scoring='r2')
print(f"Cross-Validated R-squared Scores: {cv_scores}")
print(f"Mean R-squared: {cv_scores.mean()}")

# Output polynomialRegression-results.txt
output_file_path = ('/Users/tomas/Documents/GitHub/ITHS-AI-Project/data/polynomialRegression-results.txt')

results_file_path = os.path.join(os.path.dirname(output_file_path), "polynomialRegression-results.txt")

with open(results_file_path, "w") as results_file:
    results_file.write("------ Polynomial regression ------\n")
    results_file.write(f"Mean Squared Error: {mse:.4f}\n")
    results_file.write(f"R-squared: {r2:.4f}\n")
    results_file.write("\nCross-Validation Results:\n")
    results_file.write(f"Cross-Validated R-squared Scores: {', '.join(f'{score:.4f}' for score in cv_scores)}\n")
    results_file.write(f"Mean R-squared: {cv_scores.mean():.4f}\n")
print(f"polynomialRegression results saved to {results_file_path}")
print(f"Results saved to {results_file_path}")