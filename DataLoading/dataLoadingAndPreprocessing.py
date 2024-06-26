import pandas as pd
import json

# Function to load a JSON file into a DataFrame and add a 'category' column
def load_json_file(file_path, category):
    with open(file_path, 'r') as file:
        data = json.load(file)
        df = pd.DataFrame(data)
        df['category'] = category  # Add a category column # Actually DONT NEED THIS
    return df

# Paths to your JSON files
folder_path = './data/'

bug_path = 'Bug_tt.json'
feature_path = 'Feature_tt.json'
rating_path = 'Rating_tt.json'
user_exp_path = 'UserExperience_tt.json'

# Load each file with the appropriate category label
bug_df = load_json_file(folder_path + bug_path, 'Bug')
feature_df = load_json_file(folder_path + feature_path, 'Feature')
rating_df = load_json_file(folder_path + rating_path, 'Rating')
user_exp_df = load_json_file(folder_path + user_exp_path, 'UserExperience')

# Combine all the dataframes into one
combined_df = pd.concat([bug_df, feature_df, rating_df, user_exp_df], ignore_index=True)
# pd.set_option('display.max_columns', None)
# pd.set_option('display.expand_frame_repr', False)
# print(combined_df.head())
new_df = combined_df[['category','comment','stopwords_removal','rating','stemmed','lemmatized_comment','stopwords_removal_nltk','sentiScore_pos','stopwords_removal_lemmatization', 'label']]

# Drop duplicate comments, keep only the first occurrence
unique_df = new_df.drop_duplicates('comment', keep='first')

# Calculate the new size of DataFrame after dropping duplicates
new_size = unique_df.shape[0]
original_size = combined_df.shape[0]

# Calculate the reduction in size
reduction_percentage = ((original_size - new_size) / original_size) * 100

# Output the results
print(f"Original size of dataset: {original_size}")
print(f"New size of dataset after dropping duplicates: {new_size}")
print(f"Reduction in dataset size: {reduction_percentage:.2f}%")

unique_df.to_csv(folder_path + 'combined_reviews.csv', index=False)
initial_count = unique_df.shape[0]

# Drop rows where 'label' starts with "Not_"
filtered_df = unique_df[~unique_df['label'].str.startswith('Not_')]

# Count of rows after dropping
final_count = filtered_df.shape[0]

# Calculate the percentage of rows dropped
percentage_dropped = 100 * (initial_count - final_count) / initial_count
print(f"Percentage of rows dropped: {percentage_dropped:.2f}%")


# Check for NaN values in each column
nan_counts = unique_df.isnull().sum()
print(nan_counts)

# Save the filtered DataFrame to a new CSV file
new_csv_file_path = folder_path + 'filtered_combined_reviews.csv'

# Save the cleaned DataFrame to a new CSV file
filtered_df.to_csv(new_csv_file_path, index=False)