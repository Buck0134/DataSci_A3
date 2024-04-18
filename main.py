from Model.textClassifier import TextClassifier
from textPreprocessor.testPreprocessor import TextPreprocessor
import pandas as pd

# Specify the path to your dataset
file_path = './data/filtered_combined_reviews.csv'

# file_path = './data/combined_reviews.csv'

data = pd.read_csv(file_path)
print(len(data))
labelsPre = data['label']
assert len(labelsPre) == len(data), "ORGINAL DF: Mismatch in number of samples between features and labels"

# Initialize and use the TextPreprocessor
preprocessor = TextPreprocessor(data)
features = preprocessor.preprocess(key = 'stopwords_removal_lemmatization', include_original = False, include_rating=False, include_sentiScore_pos= False, method= 'BERT')

# Assume 'label' is the column in your DataFrame that contains the target labels
labels = preprocessor.data['label']
assert len(features) == len(labels), "Mismatch in number of samples between features and labels"

print("Finished Translation")

# Initialize and train the TextClassifier
classifier = TextClassifier(features, labels, model_type='random_forest')
classifier.train()
classifier.evaluate()
