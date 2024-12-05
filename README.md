# AImed
 
Phase 1: Environment Setup and Data Exploration

 Setup Environment
Install Python and necessary libraries (pandas, scikit-learn, torch, etc.)
Configure VSCodium with Git, Python, and Jupyter notebook extensions.
Clone or initialize the Git repository.
 Data Exploration
Review and understand the raw data (FinishedOccurrences, patientsDemographics, carTrips, PatientLocation, FinishedVisits).
Inspect for missing values, outliers, data types, and distribution patterns in each field.
Document findings and structure the plan for data cleaning and feature engineering.
Phase 2: Data Processing and Feature Engineering

 Data Cleaning
Define data-cleaning functions in src/data_processing.py.
Address missing values, drop irrelevant or corrupted data, and ensure data consistency (e.g., correcting visit timestamps using carTrips data).
Validate cleaned data by visualizing distributions and identifying any remaining inconsistencies.
 Feature Engineering
Engineer features such as task_duration, time_of_day, day_of_week, travel_distance, etc., in src/feature_engineering.py.
Use one-hot encoding for categorical variables (activity type, caregiver ID, etc.).
Save the processed dataset in the data/processed/ directory for easier access during model training.
Phase 3: Model Development and Training

 Model Design and Implementation
Define a baseline model using PyTorch in src/model.py.
Implement RandomForestRegressor with PyTorch wrappers if necessary for compatibility with custom metrics and evaluation tools.
 Training and Evaluation
Split the data into training and test sets, and implement cross-validation.
Train the model with evaluation metrics like MAE and RMSE, logging metrics after each epoch.
Save the trained model to models/ using Joblib.
Phase 4: Model Evaluation and Hyperparameter Tuning

 Hyperparameter Tuning
Use grid search or random search to tune Random Forest parameters (e.g., number of trees, max depth).
Implement tuning in src/model.py and update training script to accept hyperparameters from config files.
 Analyze Feature Importance
Generate feature importance scores for interpretability.
Visualize and document key factors that influence task duration.
Phase 5: Documentation and Deployment

 Document Project
Update README.md with project overview, setup instructions, and usage notes.
Complete docstrings in code files and outline data structures in docs/.
 Final Model Export and Deployment Preparation
Save the final model with relevant metadata and deploy instructions in models/.
Test the model loading and inference with sample data in a dedicated notebook.
