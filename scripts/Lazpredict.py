import pandas as pd
from lazypredict.Supervised import LazyRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
import matplotlib.pyplot as plt

# load df
file_path = "/home/tomas/GitHub/ITHS-AI/data/df.xlsx"
df = pd.read_excel(file_path)

#filter"no match"
df_filtered = df[df['Information'] != 'no match']

# analyze `DurationMin` by `Activities.ActivityCategory`
duration_by_activity = df_filtered.groupby('Activities.ActivityCategory')['DurationMin'].agg(['mean', 'median', 'std', 'count'])
print("Duration Analysis by Activity:")
print(duration_by_activity)

# check double staffing
double_staffing_analysis = df_filtered.groupby('Activities.doubleStaffing')['DurationMin'].agg(['mean', 'median', 'std', 'count'])
print("\nDuration Analysis by Double Staffing:")
print(double_staffing_analysis)

# check gender
gender_analysis = df_filtered.groupby('gender')['DurationMin'].agg(['mean', 'median', 'std', 'count'])
print("\nDuration Analysis by Gender:")
print(gender_analysis)

# check agespan
age_span_analysis = df_filtered.groupby('ageSpan')['DurationMin'].agg(['mean', 'median', 'std', 'count'])
print("\nDuration Analysis by Age Span:")
print(age_span_analysis)

# Plot duration by activity
duration_by_activity.sort_values(by='mean', ascending=False)['mean'].plot(kind='bar', figsize=(12, 6))
plt.title("Average Duration by Activity Category")
plt.xlabel("Activity Category")
plt.ylabel("Average Duration (Min)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

#LazyPredict
# features and target
df_features = df_filtered[['Activities.ActivityCategory', 'Activities.doubleStaffing', 'gender',  'ageSpan', 'SpanDistanceM', 'SpanCarStartTime']]

# normalise numerical features
scaler = StandardScaler()
numeric_features = ['SpanDistanceM', 'SpanCarStartTime']  
df_features[numeric_features] = scaler.fit_transform(df_features[numeric_features])

# encode categorical features
activity_category_means = df_filtered.groupby('Activities.ActivityCategory')['DurationMin'].mean()# map Activities.ActivityCategory toits  mean
df_features['Activities.ActivityCategory'] = df_features['Activities.ActivityCategory'].map(activity_category_means)

# one-hot encode other categorical columns
categorical_features = ['Activities.doubleStaffing', 'gender', 'ageSpan']
df_features = pd.get_dummies(df_features, columns=categorical_features, drop_first=True)#drop one dummy

target = df_filtered['DurationMin']

# train/test
x_train, x_test, y_train, y_test = train_test_split(df_features, target, test_size=0.2, random_state=42)

# top 5 features
selector = SelectKBest(score_func=f_regression, k=5)
df_features_selected = selector.fit_transform(df_features, target)
selected_features = df_features.columns[selector.get_support()]
print("Selected Features:", selected_features)

# LazyPredict
print("\n LazyPredict")
regressor = LazyRegressor(verbose=0, ignore_warnings=True, custom_metric=None)
models, predictions = regressor.fit(x_train, x_test, y_train, y_test)
print(models)

output_file = "/home/tomas/GitHub/ITHS-AI/data/lazypredict_results_standard_scaled.xlsx" 
with pd.ExcelWriter(output_file) as writer:
    models.to_excel(writer, sheet_name="Model Comparison")
    predictions.to_excel(writer, sheet_name="Predictions")
