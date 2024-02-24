import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load the dataset
file_path = 'housing_prices/Housing.csv'
housing_data = pd.read_csv(file_path)

# Display basic info of dataset
dataset_info = {
    "First 5 Rows" : housing_data.head(),
    "Dataset Shape" : housing_data.shape,
    "Basic Stats" : housing_data.describe(),
    "Missing Values" : housing_data.isnull().sum()
}
#print(dataset_info)

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

# Define features X and target y
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