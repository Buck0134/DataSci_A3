from Model.textClassifier import TextClassifier
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
import joblib
import time
import numpy as np
import spacy
# nlp = spacy.load('en_core_web_md')  # Load the medium model with vectors

def pipeline(model_type, textual_feature, confidence_threshold, include_rating, include_sentiScore_pos, shuffle_data, use_grid_search, experiement_Model = True):
    start_time = time.time()
    def preprocess_text_data(train_df, test_df, text_column, include_rating=False, include_sentiScore_pos=False, vectorizer_path='unified_tfidf_vectorizer.joblib'):
        # Load or fit the vectorizer
        vectorizer = joblib.load(vectorizer_path)
        # print(train_df.columns)
        # Vectorize the training and testing text data separately
        X_train = vectorizer.transform(train_df[text_column])
        X_test = vectorizer.transform(test_df[text_column])

        # Convert to DataFrame
        train_features = pd.DataFrame(X_train.toarray(), columns=vectorizer.get_feature_names_out())
        test_features = pd.DataFrame(X_test.toarray(), columns=vectorizer.get_feature_names_out())

        # Include additional features like rating and sentiment scores if specified
        if include_rating:
            train_features['rating'] = train_df['rating'].values
            test_features['rating'] = test_df['rating'].values

        if include_sentiScore_pos:
            train_features['sentiScore_pos'] = train_df['sentiScore_pos'].values
            test_features['sentiScore_pos'] = test_df['sentiScore_pos'].values

        return train_features, test_features, train_df['label'], test_df['label']
    # def preprocess_text_data(train_df, test_df, text_column, include_rating=False, include_sentiScore_pos=False):
    #         # Vectorize the training and testing text data separately using spaCy
    #     X_train = vectorize_embeddings(train_df[text_column])
    #     X_test = vectorize_embeddings(test_df[text_column])

    #     # Convert to DataFrame
    #     train_features = pd.DataFrame(X_train)
    #     test_features = pd.DataFrame(X_test)
    #     train_features.columns = train_features.columns.astype(str)
    #     test_features.columns = test_features.columns.astype(str)

    #     # Include additional features like rating and sentiment scores if specified
    #     if include_rating:
    #         train_features['rating'] = train_df['rating'].values
    #         test_features['rating'] = test_df['rating'].values

    #     if include_sentiScore_pos:
    #         train_features['sentiScore_pos'] = train_df['sentiScore_pos'].values
    #         test_features['sentiScore_pos'] = test_df['sentiScore_pos'].values

    #     return train_features, test_features, train_df['label'], test_df['label']

    # def vectorize_embeddings(data):
    #     def document_vector(doc):
    #         doc = nlp(doc)
    #         vectors = [token.vector for token in doc if not token.is_stop and not token.is_punct and token.has_vector]
    #         if vectors:
    #             return np.mean(vectors, axis=0)
    #         else:
    #             return np.zeros(nlp.vocab.vectors_length)  # Return a zero vector if doc has no tokens or all tokens are stopwords/punctuations
        
    #     return np.vstack(data.apply(document_vector))

    def drop_na_rows(df):
        """
        Drops all rows from the DataFrame that contain NaN values and reports the number of dropped rows.
        
        :param df: pandas DataFrame from which to drop NaN-containing rows.
        :return: A tuple containing the cleaned DataFrame and the number of rows dropped.
        """
        initial_row_count = len(df)
        cleaned_df = df.dropna()
        final_row_count = len(cleaned_df)
        # print(f'{final_row_count - initial_row_count} rows have been dropped')
        return cleaned_df
    # Delete Duplicated Review: or added a new label to be eitherAorB

    # Specify the path to your dataset
    MC_file_path = './data/filtered_combined_reviews.csv'

    # file_path = './data/combined_reviews.csv'

    MC_data = pd.read_csv(MC_file_path)
    MC_data = drop_na_rows(MC_data)
    if shuffle_data:
        MC_data = MC_data.sample(frac=1).reset_index(drop=True)
    # Separating 
    MCtrain_data, MCtest_data = train_test_split(MC_data, test_size=0.2, random_state=42)
    X_train_mc, X_test_mc, y_train_mc, y_test_mc = preprocess_text_data(MCtrain_data, MCtest_data, textual_feature, include_rating, include_sentiScore_pos)
    # print("MultiClass Finished Translation")

    # Initialize and train the TextClassifier
    MC_classifier = TextClassifier(X_train_mc, y_train_mc, X_test_mc, y_test_mc, use_grid_search= use_grid_search, model_type=model_type)
    MC_classifier.train()
    mc_performance = MC_classifier.evaluate(experiement_Model)


    # Now we process and split individual binary ones

    splits_binary = {}
    for label in ['Bug', 'Feature', 'Rating', 'UserExperience']:
        # print(f"Processing for {label}")
        binary_file_path = f'./data/binary/{label}.csv'
        binary_data = pd.read_csv(binary_file_path)
        def drop_na_rows(df):
            initial_row_count = len(df)
            cleaned_df = df.dropna()
            final_row_count = len(cleaned_df)
            rows_dropped = initial_row_count - final_row_count
            # print(f'{rows_dropped} has been dropped')
            return cleaned_df
        binary_data = drop_na_rows(binary_data)
        if shuffle_data:
            binary_data = binary_data.sample(frac=1).reset_index(drop=True)
        # Preprocess and split the binary dataset
        Binarytrain_data, Binarytest_data = train_test_split(binary_data, test_size=0.2, random_state=42)
        X_train_b, X_test_b, y_train_b, y_test_b = preprocess_text_data(Binarytrain_data, Binarytest_data, textual_feature, include_rating, include_sentiScore_pos)
        splits_binary[label] = (X_train_b, X_test_b, y_train_b, y_test_b)


    binary_classifiers = {}
    for label in ['Bug', 'Feature', 'Rating', 'UserExperience']:
        # print(f"Training for Binary Classifier {label}")
        X_train_b, X_test_b, y_train_b, y_test_b = splits_binary[label]
        classifier_b = TextClassifier(X_train_b, y_train_b, X_test_b, y_test_b, use_grid_search= use_grid_search, model_type='svm')
        classifier_b.train()
        binary_classifiers[label] = classifier_b

    def run_pipeline(X_test, multi_class_model, binary_classifiers, confidence_threshold=0.60):
        # Predict with the multi-class classifier
        probabilities = multi_class_model.predict_proba(X_test)
        predicted_classes = multi_class_model.predict(X_test)
        final_predictions = []
        for i, (probs, predicted_class) in enumerate(zip(probabilities, predicted_classes)):
            max_prob = max(probs)
            # print(max_prob)
            if max_prob < confidence_threshold:
                # print(predicted_class)
                # If the confidence of the multi-class prediction is below threshold,
                # check the binary classifier for the predicted class.
                binary_model = binary_classifiers[predicted_class]  # Dictionary of binary classifiers
                is_class = binary_model.predict(X_test[i:i+1])[0]
                if is_class == predicted_class:
                    # print(f"Verified for {predicted_class}")
                    final_predictions.append(predicted_class)
                else:
                    # If the binary classifier disagrees with the multi-class classifier,
                    # run the remaining binary classifiers and pick the one with the highest probability
                    other_probabilities = []
                    for label, classifier in binary_classifiers.items():
                        if label != predicted_class:  # Do not re-evaluate the already checked classifier
                            prob = classifier.predict_proba(X_test[i:i+1])[0][1]  # Assuming 1 is the class index
                            other_probabilities.append((prob, label))
                    # print(other_probabilities)
                    # Find the label with the maximum probability among the other classifiers
                    if other_probabilities:
                        max_prob_label = max(other_probabilities, key=lambda x: x[0])[1]
                        # print(f'usedTobe: {predicted_class} --> now: {max_prob_label}')
                        final_predictions.append(max_prob_label)
                    else:
                        final_predictions.append('Unclassified')  # Fallback if no other classifier is available
            else:
                final_predictions.append(predicted_class)
        
        return final_predictions

    final_predictions = run_pipeline(X_test_mc, MC_classifier, binary_classifiers, confidence_threshold)
    # print("Pipeline evaluation:")
    # print(classification_report(y_test_mc, final_predictions))
    binary_performance = classification_report(y_test_mc, final_predictions, output_dict=experiement_Model)
    # Record the time taken
    end_time = time.time()
    time_taken = end_time - start_time
    return mc_performance, binary_performance, time_taken


# model_type, confidence_threshold, include_rating, include_sentiScore_pos, shuffle_data, use_grid_search
mc_performance, binary_performance, time_taken = pipeline('random_forest', 'stopwords_removal_lemmatization', 0.7, True, False, False, False, False)
print("Pre pipeline performance from Multi-class model:\n", mc_performance)
print("Post pipeline performance going into Binary classifiers:\n", binary_performance)
print("Time taken: {:.2f} seconds".format(time_taken))