import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import os
import re

# Load the dataset
dataset_path = 'datasets\healthcare_messy_data.csv'
df = pd.read_csv(dataset_path)

# Create directory if not exists
output_dir = r'datasets\results'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"Created directory: {output_dir}")

print('Initial data shape:', df.shape)
print(df.info())

# Drop duplicate rows
n_duplicates = df.duplicated().sum()
df.drop_duplicates(inplace=True)
print('Number of duplicates dropped:', n_duplicates)

# Handling missing values
# Fill numerical columns with median, categorical with mode
for col in df.columns:
    if df[col].dtype in ['int64', 'float64']:
        median_value = df[col].median()
        df[col] = df[col].fillna(median_value)
        print(f"Filled missing numerical values in {col} with median: {median_value}")
    else:
        mode_value = df[col].mode()[0]
        df[col] = df[col].fillna(mode_value)
        print(f"Filled missing categorical values in {col} with mode: {mode_value}")

# Identify columns to split using regex pattern
score_pattern = re.compile(r'_(score|measurement)$', re.IGNORECASE)
for col in df.columns:
    if score_pattern.search(col):
        print(f"Identified column {col} for potential splitting")
        # Placeholder for actual splitting logic
        # For now, just note

# Encode categorical variables
label_encoders = {}
for col in df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le
    print(f"Encoded categorical column: {col}")

# Scale numerical features except target
target_col = 'Condition'
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
if target_col in numerical_cols:
    numerical_cols.remove(target_col)
scaler = StandardScaler()
try:
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    print('Scaled numerical features.')
except Exception as e:
    print(f"Error scaling numerical features: {e}")

# Save the preprocessed data
save_path = r'datasets\results\preprocessed_healthcare_data.csv'
df.to_csv(save_path, index=False)
print('Preprocessed dataset saved at:', save_path)

# Return the path to the saved dataset
save_path