import pandas as pd

# Load the dataset
file_path = 'C:\\Users\\BUYMORE\\Desktop\\Github Proj\\venv\\All\\crewai_version\\datasets\\healthcare_messy_data.csv'

try:
    data = pd.read_csv(file_path)
    print('Dataset loaded successfully.')
except Exception as e:
    print('Error loading dataset:', e)

# Inspect the dataset
print(data.head())
print(data.info())
print(data.describe())

# Drop identifier columns if detected
# Assuming 'id' is an identifier column
data.drop(columns=['id'], inplace=True, errors='ignore')
print('Dropped identifier columns if any.')

# Handle missing values
missing_percentage = data.isnull().mean() * 100
columns_to_drop = missing_percentage[missing_percentage > 50].index.tolist()
data.drop(columns=columns_to_drop, inplace=True)
print(f'Dropped columns with >50% missing values: {columns_to_drop}')

# Fill remaining missing values for categorical and numeric columns
for column in data.select_dtypes(include=['object']).columns:  # Categorical columns
    mode_val = data[column].mode()[0]
    data[column] = data[column].fillna(mode_val)
    print(f'Filled missing values for categorical column {column}.')
for column in data.select_dtypes(include=['number']).columns:
    median_val = data[column].median()
    data[column] = data[column].fillna(median_val)
    print(f'Filled missing values for numeric column {column}.')

# Standardize date features if any
# Assuming 'date' is a column that needs standardization
date_columns = [col for col in data.columns if 'date' in col.lower()]
for date_col in date_columns:
    data[date_col] = pd.to_datetime(data[date_col], errors='coerce').dt.date
print('Standardized date features.')

# Clean categorical text
cat_columns = data.select_dtypes(include=['object']).columns
for column in cat_columns:
    data[column] = data[column].astype(str).str.strip().str.lower()
print('Cleaned categorical text.')

# Remove duplicates
data.drop_duplicates(inplace=True)
print('Removed duplicates.')

# Encode categorical features
data = pd.get_dummies(data, drop_first=True)
print('Encoded categorical features.')

# Scale numeric features if needed
num_columns = data.select_dtypes(include=['number']).columns.tolist()
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
data[num_columns] = scaler.fit_transform(data[num_columns])
print('Scaled numeric features.')

# Save the cleaned dataset
output_path = 'C:\\Users\\BUYMORE\\Desktop\\Github Proj\\venv\\All\\crewai_version\\datasets\\results\\cleaned_healthcare_messy_data.csv'
data.to_csv(output_path, index=False)
print('Cleaned dataset saved at:', output_path)
