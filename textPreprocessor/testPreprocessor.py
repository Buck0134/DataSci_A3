import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import pandas as pd
import spacy

class TextPreprocessor:
    def __init__(self, data):
        self.data = data
        self.tfidf_vectorizer = TfidfVectorizer()
        self.count_vectorizer = CountVectorizer()
        self.nlp = spacy.load('en_core_web_md')  # Load spaCy model
        self.data = self.drop_na_rows(self.data)

    def drop_na_rows(self, df):
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

    def preprocess(self, key, include_original=False, include_rating=False, include_sentiScore_pos=False, method='tfidf'):
        features = []

        if method == 'tfidf':
            # Encode comments using TF-IDF
            tfidf_features = self.tfidf_vectorizer.fit_transform(self.data[key])
            tfidf_df = pd.DataFrame(tfidf_features.toarray(), columns=self.tfidf_vectorizer.get_feature_names_out())
            features.append(tfidf_df)

        elif method == 'spacy':
            # Vectorize text using spaCy
            doc_vectors = [self.nlp(text).vector for text in self.data[key]]
            vector_df = pd.DataFrame(doc_vectors)
            features.append(vector_df)

            if include_original:
                original_vectors = [self.nlp(text).vector for text in self.data['comment']]
                original_df = pd.DataFrame(original_vectors)
                features.append(original_df)

        elif method == 'count':
            # Encode comments using Count Vectorization
            count_features = self.count_vectorizer.fit_transform(self.data[key])
            count_df = pd.DataFrame(count_features.toarray(), columns=self.count_vectorizer.get_feature_names_out())
            features.append(count_df)

        # Add optional features
        if include_rating:
            rating_df = self.data['rating'].reset_index(drop=True).to_frame('rating')
            features.append(rating_df)

        if include_sentiScore_pos:
            sentiScore_pos_df = self.data['sentiScore_pos'].reset_index(drop=True).to_frame('sentiScore_pos')
            features.append(sentiScore_pos_df)

        # Combine all selected features into a single DataFrame
        combined_features = pd.concat(features, axis=1) if len(features) > 1 else features[0]

        return combined_features
