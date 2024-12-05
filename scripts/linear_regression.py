import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import os

file_path = "/Users/tomas/Documents/GitHub/ITHS-AI-Project/data/df.xlsx"
df = pd.read_excel(file_path)
df = df[df['Information'] == 'match']

# Determine OS
# def get_output_file_path(config):
#     if os.name == 'nt':  # Windows
#         return config['file_paths']['windows']
#     elif os.uname().sysname == 'Darwin':  # macOS
#         return config['file_paths']['mac']
#     else:  # Assume Linux for other systems
#         return config['file_paths']['linux']

# Define features
df_features = df[['Activities.ActivityCategory', 'Activities.doubleStaffing', 'gender', 
               'ageSpan', 'SpanDistanceM', 'SpanCarStartTime']]

# Activities.ActivityCategory encoding
activity_category_means = df.groupby('Activities.ActivityCategory')['DurationMin'].mean()
df_features['Activities.ActivityCategory'] = df_features['Activities.ActivityCategory'].map(activity_category_means)
df_features = df_features.drop(columns=['Activities.ActivityCategory'])

# One-hot encoding 
categorical_features = ['Activities.doubleStaffing', 'gender', 'ageSpan', 'SpanDistanceM', 'SpanCarStartTime']
df_features = pd.get_dummies(df_features, columns=categorical_features, drop_first=True)

target = df['DurationMin']

## Visualise features
# sns.boxplot(data=df[['Activities.ActivityCategory', 'Activities.doubleStaffing', 'gender', 
#                      'ageSpan', 'SpanDistanceM', 'SpanCarStartTime']])
# plt.title("Boxplot of Features")
# plt.show() 

## Check if linear relationship exists between SpanDistanceM / DurationMin
# plt.scatter(df['SpanDistanceM'], target)
# plt.xlabel('SpanDistanceM')
# plt.ylabel('DurationMin')
# plt.show()

if df_features.isnull().any().any():
    print("Features contain NaN")

if target.isnull().any():
    print("Target contains NaN")

# # # Scale
# scaler = StandardScaler()
# features_scaled = scaler.fit_transform(features)

# Split train 
#  test sets
x_train, x_test, y_train, y_test = train_test_split(df_features, target, test_size=0.2, random_state=42)

# Fit LinearRegression
model = LinearRegression()
model.fit(x_train, y_train)

# Evaluate model
y_pred = model.predict(x_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("------ Linear regression ------")
print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")

# Crossvalidation
cv_scores = cross_val_score(model, x_train, y_train, cv=5, scoring='r2')
print(f"Cross-Validated R-squared Scores: {cv_scores}")
print(f"Mean R-squared: {cv_scores.mean()}")

# coefficients = pd.DataFrame({
#     "Feature": df.columns,
#     "Coefficient": model.coef_
# }).sort_values(by="Coefficient", ascending=False)
# print(coefficients)

# Output LinearRegression-results.txt
output_file_path = ('/Users/tomas/Documents/GitHub/ITHS-AI-Project/data/df.xlsx')

results_file_path = os.path.join(os.path.dirname(output_file_path), "LinearRegression-results.txt")

with open(results_file_path, "w") as results_file:
    results_file.write("------ Linear Regression Results ------\n")
    results_file.write(f"Mean Squared Error: {mse:.4f}\n")
    results_file.write(f"R-squared: {r2:.4f}\n")
    results_file.write("\nCross-Validation Results:\n")
    results_file.write(f"Cross-Validated R-squared Scores: {', '.join(f'{score:.4f}' for score in cv_scores)}\n")
    results_file.write(f"Mean R-squared: {cv_scores.mean():.4f}\n")
print(f"Linear regression results saved to {results_file_path}")
