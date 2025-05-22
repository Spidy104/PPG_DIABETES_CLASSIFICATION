import pandas as pd
import files


# Load Excel, skipping title row
try:
    metadata = pd.read_excel(files.DATASET_CSV_PATH, skiprows=1)
except Exception as e:
    print(f"Error loading Excel: {e}")
    raise

# Print column names to verify
print("Column names:")
print(list(metadata.columns))

# Verify "Diabetes" column
if 'Diabetes' not in metadata.columns:
    print("Error: 'Diabetes' column not found. Available columns:")
    print(list(metadata.columns))
    # Try case-insensitive match
    for col in metadata.columns:
        if col.lower() == 'diabetes':
            print(f"Found similar column: {col}")
            metadata.rename(columns={col: 'Diabetes'}, inplace=True)
            break
    else:
        raise ValueError("Please specify the correct column name for Diabetes")

# Create binary diabetes labels
metadata['diabetes_label'] = metadata['Diabetes'].apply(
    lambda x: 1 if pd.notna(x) and ('Diabetes' in str(x) or 'Type 2 Diabetes' in str(x)) else 0
)

# Verify label counts
total_subjects = len(metadata)
diabetic_count = metadata['diabetes_label'].sum()
print(f"\nTotal subjects: {total_subjects}")
print(f"Diabetic: {diabetic_count}")
print(f"Non-diabetic: {total_subjects - diabetic_count}")

# Select and rename metadata features
metadata_features = metadata[[
    'subject_ID',
    'Sex(M/F)',
    'Age(year)',
    'BMI(kg/m^2)',
    'Heart Rate(b/m)',
    'diabetes_label'
]].rename(columns={
    'Sex(M/F)': 'Sex',
    'Age(year)': 'Age',
    'BMI(kg/m^2)': 'BMI',
    'Heart Rate(b/m)': 'Heart Rate'
})

# Convert Sex to binary (M=1, F=0)
metadata_features['Sex'] = metadata_features['Sex'].map({'Male': 1, 'Female': 0})

# Handle missing values
metadata_features = metadata_features.fillna(0)

# Preview DataFrame
print("\nMetadata features preview:")
print(metadata_features.head())

# Save to CSV
metadata_features.to_csv('processed_metadata.csv', index=False)
print("\nSaved processed metadata to 'processed_metadata.csv'")