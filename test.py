import csv

# Input and output file paths
input_file = 'SMSSpamCollection.csv'
output_file = 'messages_modified.csv'

# Open the input file for reading
with open(input_file, mode='r', encoding='utf-8') as infile:
    # Read all lines
    lines = infile.readlines()

# Prepare rows for the output
rows = []
for line in lines:
    # Strip any leading/trailing whitespace
    line = line.strip()
    # Split the line into label and message
    parts = line.split('\t', 1)
    if len(parts) == 2:
        label, message = parts
        # Replace 'ham' with '0' and 'spam' with '1'
        if label == 'ham':
            label = '0'
        elif label == 'spam':
            label = '1'
        # Append the modified row to the list, swapping message and label
        rows.append([message, label])

# Write the output file with modified labels
with open(output_file, mode='w', encoding='utf-8', newline='') as outfile:
    writer = csv.writer(outfile)
    # Write the header with swapped columns
    writer.writerow(['message', 'label'])
    # Write the transformed rows
    writer.writerows(rows)

print(f"File '{output_file}' has been created with message first and label second.")
