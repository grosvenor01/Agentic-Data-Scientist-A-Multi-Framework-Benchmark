import pandas as pd
import numpy as np
# Load dataset
file_path = r'C:\Users\BUYMORE\Desktop\Github Proj\ven\All\Data Science Agent\datasets\healthcare_messy_data.csv'
df = pd.read_csv(file_path)

# Initial info
print('Initial Data Info:')
print(df.info())

# Convert 'Age' to numerical, handle non-numeric
df['Age'] = pd.to_numeric(df['Age'], errors='coerce')

# Convert 'Blood Pressure' to numerical (assuming format 'systolic/diastolic')
def extract_systolic(bp):
    try:
        return int(bp.split('/')[0])
    except:
        return np.nan

if 'Blood Pressure' in df.columns:
    df['Systolic BP'] = df['Blood Pressure'].apply(extract_systolic)
    df.drop('Blood Pressure', axis=1, inplace=True)

# Remove duplicates
df.drop_duplicates(inplace=True)

# Drop unneeded columns
df.drop(['Patient Name', 'Email', 'Phone Number', 'Visit Date'], axis=1, errors='ignore', inplace=True)

# Handle missing data
# Numerical columns
for col in ['Age', 'Cholesterol', 'Systolic BP']:
    if col in df.columns:
        df[col].fillna(df[col].median(), inplace=True)
# Categorical columns
for col in ['Gender', 'Condition', 'Medication']:
    if col in df.columns:
        mode_value = df[col].mode()[0]
        df[col].fillna(mode_value, inplace=True)

# Final info
print('Cleaned Data Info:')
print(df.info())

# Save cleaned data
output_path = r"C:\Users\BUYMORE\Desktop\Github Proj\ven\All\Data Science Agent\datasets\results\cleaned_healthcare_data.csv"
df.to_csv(output_path, index=False)
print(f"Cleaned data saved to {output_path}")