import pandas as pd
import json
from sklearn.model_selection import train_test_split

def json_to_csv(file_path, columns):
    """Load a JSON file, select necessary columns, and save to CSV."""
    with open(file_path, 'r') as file:
        data = json.load(file)
        df = pd.DataFrame(data)
        # Select the necessary columns
        df = df[columns]
    return df

# Specify the columns you want to extract
columns = ['comment', 'label', 'rating', 'sentiScore_pos', 'stopwords_removal', 'comment', 'stemmed', 'lemmatized_comment', 'stopwords_removal_nltk', 'stopwords_removal_lemmatization']

# Paths to your JSON files and CSV conversion
folder_path = './data/'
output_path = './data/binary/'
labels = ['Bug', 'Feature', 'Rating', 'UserExperience']
json_files = {label: f'{label}_tt.json' for label in labels}
csv_files = {label: f'{label}.csv' for label in labels}

# Process each JSON file
for label in labels:
    json_path = folder_path + json_files[label]
    csv_path = output_path + csv_files[label]
    df = json_to_csv(json_path, columns)
    df.to_csv(csv_path, index=False)
