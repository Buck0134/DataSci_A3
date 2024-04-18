import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from transformers import BertTokenizer, BertModel
import torch

class TextPreprocessor:
    def __init__(self, data):
        self.data = data
        self.tfidf_vectorizer = TfidfVectorizer()
        self.count_vectorizer = CountVectorizer()
        def drop_na_rows(df):
            """
            Drops all rows from the DataFrame that contain NaN values and reports the number of dropped rows.
            
            :param df: pandas DataFrame from which to drop NaN-containing rows.
            :return: A tuple containing the cleaned DataFrame and the number of rows dropped.
            """
            initial_row_count = len(df)
            cleaned_df = df.dropna()
            final_row_count = len(cleaned_df)
            rows_dropped = initial_row_count - final_row_count
            print(f'{rows_dropped} has been dropped')
            return cleaned_df
        self.data = drop_na_rows(self.data)

    def preprocess(self, key, include_original=False, include_rating=False, include_sentiScore_pos=False, method = 'tfidf'):
        features = []

        if method == 'tfidf':
            # Encode comments using TF-IDF
            tfidf_features = self.tfidf_vectorizer.fit_transform(self.data[key])
            tfidf_df = pd.DataFrame(tfidf_features.toarray(), columns=self.tfidf_vectorizer.get_feature_names_out())
            tfidf_df.reset_index(drop=True, inplace=True)
            features.append(tfidf_df)
        
            if include_original:
                # Original comments also encoded via TF-IDF if included
                original_features = self.tfidf_vectorizer.fit_transform(self.data['comment'])
                original_df = pd.DataFrame(original_features.toarray(), columns=self.tfidf_vectorizer.get_feature_names_out())
                original_df.reset_index(drop=True, inplace=True)
                features.append(original_df)

        # NOT APPLICABLE: Time/Resource Consuming
        elif method == 'BERT':
            # This line downloads and caches the tokenizer and model
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            model = BertModel.from_pretrained('bert-base-uncased')
            # Use BERT to "translate" the features     
            # Encode and transform text using BERT
            encoded_input = tokenizer(self.data[key].tolist(), padding=True, truncation=True, max_length=512, return_tensors='pt')
            with torch.no_grad():
                outputs = model(**encoded_input)
            
            # Use the pooler_output for classification tasks as it represents the entire sequence
            bert_features = outputs.pooler_output
            bert_features = bert_features.numpy()  # Convert to NumPy array for use in pandas DataFrame
            bert_df = pd.DataFrame(bert_features)
            bert_df.reset_index(drop=True, inplace=True)
            features.append(bert_df)

            if include_original:
                encoded_original = tokenizer(self.data['comment'].tolist(), padding=True, truncation=True, max_length=512, return_tensors='pt')
                with torch.no_grad():
                    outputs_original = model(**encoded_original)
                # Use the pooler_output for classification tasks as it represents the entire sequence
                bert_features_original = outputs_original.pooler_output
                bert_features_original = bert_features.numpy()  # Convert to NumPy array for use in pandas DataFrame
                bert_df_original = pd.DataFrame(bert_features_original)
                bert_df_original.reset_index(drop=True, inplace=True)
                features.append(bert_df_original)

        elif method == 'count':
            # Encode comments using Count Vectorization
            count_features = self.count_vectorizer.fit_transform(self.data[key])
            count_df = pd.DataFrame(count_features.toarray(), columns=self.count_vectorizer.get_feature_names_out())
            count_df.reset_index(drop=True, inplace=True)
            features.append(count_df)

            if include_original:
                # Original comments also encoded via Count Vectorization if included
                original_features = self.count_vectorizer.fit_transform(self.data['comment'])
                original_df = pd.DataFrame(original_features.toarray(), columns=self.count_vectorizer.get_feature_names_out())
                original_df.reset_index(drop=True, inplace=True)
                features.append(original_df)

        if include_rating:
            rating_df = self.data['rating'].reset_index(drop=True).to_frame('rating')
            features.append(rating_df)

        if include_sentiScore_pos:
            sentiScore_pos_df = self.data['sentiScore_pos'].reset_index(drop=True).to_frame('sentiScore_pos')
            features.append(sentiScore_pos_df)
        # Combine all selected features into a single DataFrame
        if len(features) > 1:
            combined_features = pd.concat(features, axis=1)
        else:
            combined_features = features[0]  # If there's only one DataFrame, just use it directly

        return combined_features
