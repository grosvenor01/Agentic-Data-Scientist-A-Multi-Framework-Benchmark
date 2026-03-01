import pandas as pd

# Load dataset
file_path = 'datasets/results/cleaned_healthcare_data.csv'
df = pd.read_csv(file_path)

# Inspection
print('Shape:', df.shape)
print('Data types:', df.dtypes)
print('Missing values:', df.isnull().sum())

# Drop identifier columns if any
identifier_columns = [] # Update this based on actual identifiers
if identifier_columns:
    df.drop(columns=identifier_columns, inplace=True)

# Handle missing values
for column in df.columns:
    if df[column].isnull().sum() > 0:
        if df[column].dtype == 'object':
            df[column].fillna(df[column].mode()[0], inplace=True)  # fill with mode for categorical
        else:
            df[column].fillna(df[column].median(), inplace=True)  # fill with median for numeric

# Standardize date columns if any
for column in df.select_dtypes(include=['datetime64[ns]']).columns:
    df[column] = pd.to_datetime(df[column])  # ensure all date columns are in datetime format

# Extract useful date features
if 'date_column' in df.columns:
    df['year'] = df['date_column'].dt.year
    df['month'] = df['date_column'].dt.month
    del df['date_column']  # drop original date column

# Clean categorical text
for column in df.select_dtypes(include=['object']):
    df[column] = df[column].str.strip().str.lower()

# Remove duplicates
df.drop_duplicates(inplace=True)

# Encode categorical features
categorical_columns = df.select_dtypes(include=['object']).columns
for column in categorical_columns:
    df[column] = pd.factorize(df[column])[0]

# Check if 'Medication' column exists and encode it
if 'Medication' in df.columns:
    df['Medication'] = pd.factorize(df['Medication'])[0]

# Save cleaned dataset
output_path = 'datasets/results/cleaned_healthcare_data_encoded.csv'
df.to_csv(output_path, index=False)

# Output final saved path
print('SAVED_PATH:', output_path)