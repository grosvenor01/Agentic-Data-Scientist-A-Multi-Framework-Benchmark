import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Load the dataset
file_path = r"C:\Users\BUYMORE\Desktop\Github Proj\ven\All\Data Science Agent\datasets\healthcare_messy_data.csv"
df = pd.read_csv(file_path)

# Drop irrelevant columns based on prior analysis
columns_to_drop = ['patient_id', 'record_number']
df.drop(columns=columns_to_drop, inplace=True, errors='ignore')

# Handle missing values for numerical columns
numeric_cols = df.select_dtypes(include=['number']).columns

df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

# Handle missing values for categorical columns
categorical_cols = df.select_dtypes(include=['object']).columns

df[categorical_cols] = df[categorical_cols].fillna(df[categorical_cols].mode().iloc[0])

# Verify missing values are handled
print("Remaining missing values:")
print(df.isnull().sum())

# Define preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ])

# Fit and transform the data
try:
    transformed_data = preprocessor.fit_transform(df)
except Exception as e:
    print("Error during transformation:", e)

# Get feature names for encoded categorical variables
try:
    encoded_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_cols)
except AttributeError:
    encoded_feature_names = []

# Combine feature names
feature_names = list(numeric_cols) + list(encoded_feature_names)

# Convert the transformed data to DataFrame
import numpy as np
try:
    df_processed = pd.DataFrame(transformed_data.toarray() if hasattr(transformed_data, "toarray") else transformed_data, columns=feature_names)
except Exception as e:
    print("Error converting transformed data to DataFrame:", e)

# Save the preprocessed dataset
output_path = r"C:\Users\BUYMORE\Desktop\Github Proj\ven\All\Data Science Agent\datasets\results\healthcare_cleaned_data.csv"
df_processed.to_csv(output_path, index=False)
print(f"Preprocessed data saved to {output_path}")