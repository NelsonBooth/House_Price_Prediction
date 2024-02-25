import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import numpy as np

# Load the dataset
file_path = 'kc_housing_prices/kc_house_data.csv'
housing_data = pd.read_csv(file_path)

# Drop specified features from dataset
features_to_drop = ['id', 'date', 'sqft_living', 'sqft_lot', 'waterfront', 
                    'view', 'condition', 'grade', 'yr_renovated', 'zipcode', 'lat', 'long']
housing_data = housing_data.drop(features_to_drop, axis=1)

# Display basic info of dataset
dataset_info = {
    "First 5 Rows" : housing_data.head(),
    "Dataset Shape" : housing_data.shape,
    "Basic Stats" : housing_data.describe(),
    "Missing Values" : housing_data.isnull().sum()
}
#print(dataset_info)

#-------------------------------------------------------------------------------------------------------

# Preprocess the dataset
# Identify numerical and categorical columns
numerical_cols = housing_data.select_dtypes(include=['int64', 'float64']).columns.drop('price')
categorical_cols = housing_data.select_dtypes(include=['object']).columns

# Create transformers for numerical and categorical columns
numerical_transformer = StandardScaler() # used for normalizing to have zero mean and unit variance
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

# Bundle the transformers into preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ]
)

# Define features X and labels y
X = housing_data.drop('price', axis=1)
y = housing_data['price']

# Split dataset into train and test splits
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)

# Apply preprocessing to training data
X_train_preprocessed = preprocessor.fit_transform(X_train)
X_test_preprocessed = preprocessor.transform(X_test)

# Display preprocessed data shape
preprocessed_data_info = {
    "X_train_preprocessed" : X_train_preprocessed.shape,
    "X_test_preprocessed" : X_test_preprocessed.shape
}
print(preprocessed_data_info)

#-------------------------------------------------------------------------------------------------------

# Implement hyperparameter tuning
# Parameter grid to search
param_grid = {
    'n_estimators': [200, 300, 400],
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 3, 4],
    'min_samples_leaf': [4, 5, 6]
}

# Create the Random Forest Regressor model
rf_regressor = RandomForestRegressor(random_state=0)

# Start grid search model
grid_search = GridSearchCV(estimator=rf_regressor, param_grid=param_grid, cv=10, n_jobs=-1,
                           verbose=2, scoring='neg_mean_squared_error')

# Train the grid search on preprocessed training data
grid_search.fit(X_train_preprocessed, y_train)

# Display best parameters
best_params = grid_search.best_params_
print(f"Best Parameters: {best_params}")

# Train the model with best parameters
best_rf_regressor = RandomForestRegressor(**best_params, random_state=0)
best_rf_regressor.fit(X_train_preprocessed, y_train)

# Make predictions on test data
y_pred = best_rf_regressor.predict(X_test_preprocessed)

# Calculate performance metrics
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
print(f"RMSE: {rmse}")
print(f"R2 Score: {r2}")

# Display a plot of actual vs predicted values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--k')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs. Predicted Values')
plt.show()