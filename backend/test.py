import pandas as pd
import json

# Load your CSV file into a DataFrame
df = pd.read_csv('backend/data/GDSC_DATASET.csv')

# Count the unique values in the 'Cell Line Name' column
unique_cell_lines = df['CELL_LINE_NAME'].unique()

# Optionally, you can sort the list
unique_cell_lines_sorted = sorted(unique_cell_lines)

# Save the unique names to a JSON file (just the names, no counts)
with open('unique_cell_lines.json', 'w') as f:
    json.dump(unique_cell_lines_sorted, f)  # No need for .tolist()

# Print the result (for debugging purposes)
print(unique_cell_lines_sorted)
