import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import os

# Load and filter the dataset
file_path = "/Users/tomas/Documents/GitHub/ITHS-AI-Project/data/df.xlsx"
df = pd.read_excel(file_path)
df = df[df['Information'] == 'match']

# Define features and target
df_features = df[['Activities.ActivityCategory', 'Activities.doubleStaffing', 'gender', 
                  'ageSpan', 'SpanDistanceM', 'SpanCarStartTime']]

# Target encoding for Activities.ActivityCategory
activity_category_means = df.groupby('Activities.ActivityCategory')['DurationMin'].mean()
df_features['Activities.ActivityCategory'] = df_features['Activities.ActivityCategory'].map(activity_category_means)
df_features = df_features.drop(columns=['Activities.ActivityCategory'])

# One-hot encoding for other categorical features
categorical_features = ['Activities.doubleStaffing', 'gender', 'ageSpan', 'SpanDistanceM', 'SpanCarStartTime']
df_features = pd.get_dummies(df_features, columns=categorical_features, drop_first=True)

target = df['DurationMin']

# Check for NaN values
if df_features.isnull().any().any():
    print("Features contain NaN")
if target.isnull().any():
    print("Target contains NaN")

# Split into train and test sets
x_train, x_test, y_train, y_test = train_test_split(df_features, target, test_size=0.2, random_state=42)

# Fit the Random Forest Regressor model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(x_train, y_train)

# Evaluate the model
y_pred = model.predict(x_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("------ Random Forest regression ------")
print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")

# Cross-validation
cv_scores = cross_val_score(model, x_train, y_train, cv=5, scoring='r2')
print(f"Cross-Validated R-squared Scores: {cv_scores}")
print(f"Mean R-squared: {cv_scores.mean()}")

# Feature importance
feature_importances = pd.DataFrame({
    "Feature": df_features.columns,
    "Importance": model.feature_importances_
}).sort_values(by="Importance", ascending=False)

print("Feature Importances:")
print(feature_importances)

# Visualize feature importance
plt.figure(figsize=(10, 6))
plt.barh(feature_importances['Feature'], feature_importances['Importance'], color='skyblue')
plt.xlabel("Feature Importance")
plt.ylabel("Feature")
plt.title("Random Forest Feature Importances")
plt.gca().invert_yaxis()
plt.show()

output_dir ='/Users/tomas/Documents/GitHub/ITHS-AI-Project/data/'
results_file_path = os.path.join(output_dir, "RandomForest-results.txt")
with open(results_file_path, "w") as results_file:
    results_file.write("------ Random Forest Regression Results ------\n")
    results_file.write(f"Mean Squared Error: {mse:.4f}\n")
    results_file.write(f"R-squared: {r2:.4f}\n")
    results_file.write("\nCross-Validation Results:\n")
    cv_scores = cross_val_score(model, x_train, y_train, cv=5, scoring='r2')
    results_file.write(f"Cross-Validated R-squared Scores: {', '.join(f'{score:.4f}' for score in cv_scores)}\n")
    results_file.write(f"Mean R-squared: {cv_scores.mean():.4f}\n")

print(f"Regression results saved to {results_file_path}")
