import json

# Create the final JSON output
output = {
    "report": "Data processing completed successfully. No missing values present in the final dataset.",
    "preprocessed_dataset_path": "datasets/results/cleaned_healthcare_messy_data_final_v2.csv"
}

# Convert to JSON
final_json_output = json.dumps(output)
final_json_output