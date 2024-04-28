import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from Model.textClassifier import TextClassifier
from textPreprocessor.testPreprocessor import TextPreprocessor
import joblib

# Create a unified TextPreprocessor instance or function
class TextPreprocessor:
    def __init__(self, data):
        self.data = data
        self.vectorizer = TfidfVectorizer()  # Use a single vectorizer for all tasks

    def preprocess(self, key='text', method='tfidf'):
        if method == 'tfidf':
            # Fit the vectorizer only once and transform the data
            features = self.vectorizer.fit_transform(self.data[key])
            features_df = pd.DataFrame(features.toarray(), columns=self.vectorizer.get_feature_names_out())
            features_df.columns = features_df.columns.astype(str)  # Ensure column names are string type
            return features_df

# Combine datasets for fitting the vectorizer
combined_data = pd.read_csv('./data/combined_reviews.csv')
def drop_na_rows(df):
    """
    Drops all rows from the DataFrame that contain NaN values and reports the number of dropped rows.
    
    :param df: pandas DataFrame from which to drop NaN-containing rows.
    :return: A tuple containing the cleaned DataFrame and the number of rows dropped.
    """
    initial_row_count = len(df)
    cleaned_df = df.dropna()
    final_row_count = len(cleaned_df)
    print(f'{final_row_count - initial_row_count} rows have been dropped')
    return cleaned_df
combined_data = drop_na_rows(combined_data)
preprocessor = TextPreprocessor(combined_data)
combined_features = preprocessor.preprocess(key='stopwords_removal_lemmatization')

# Now split and apply to specific models
# You would use the already fitted `preprocessor.vectorizer` to transform training and testing datasets for each specific model without fitting it again

# Serialize the fitted vectorizer for consistent use in production
joblib.dump(preprocessor.vectorizer, 'unified_tfidf_vectorizer.joblib')
