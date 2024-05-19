import csv
from collections import OrderedDict  # Import OrderedDict

# Read data from CSV file
def read_data_from_csv(file_path):
    data = {'text': [], 'off-platform': []}
    with open(file_path, 'r', newline='', encoding='utf-8') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header row if exists
        for row in reader:
            # Check if any field is not an empty string
            if any(field.strip() for field in row):
                data['text'].append(row[0])  # Assuming text is in the first column
                data['off-platform'].append(int(row[1]))  # Assuming off-platform is in the second column
    return data

# Remove duplicates from dataset
def remove_duplicates(data):
    # Combine text and off-platform into tuples
    combined_data = list(zip(data['text'], data['off-platform']))
    # Remove duplicates while preserving order
    unique_data = list(OrderedDict.fromkeys(combined_data))
    # Split back into text and off-platform
    unique_text, unique_off_platform = zip(*unique_data)
    # Convert back to dictionary
    unique_data_dict = {'text': list(unique_text), 'off-platform': list(unique_off_platform)}
    return unique_data_dict

# Write data to CSV file
def write_data_to_csv(data, file_path):
    with open(file_path, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['text', 'off-platform'])  # Write header row
        for text, off_platform in zip(data['text'], data['off-platform']):
            writer.writerow([text, off_platform])

# Example usage
input_csv_file = 'data.csv'
output_csv_file = 'collected_data.csv'

# Read data from CSV file
data = read_data_from_csv(input_csv_file)

# Remove duplicates
unique_data = remove_duplicates(data)

# Write unique data to CSV file
write_data_to_csv(unique_data, output_csv_file)
