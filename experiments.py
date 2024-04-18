import itertools
import pandas as pd
import matplotlib.pyplot as plt
from Model.textClassifier import TextClassifier
from textPreprocessor.testPreprocessor import TextPreprocessor
from sklearn.metrics import classification_report, accuracy_score, f1_score
from tqdm import tqdm

# Possible Classifers Models 

models = ['random_forest', 'svm', 'naive_bayes','mlp']

# Possible settings
keys = ['stopwords_removal_nltk', 'stopwords_removal', 'stemmed', 'lemmatized_comment', 'stopwords_removal_lemmatization']
options = [False, True]

# Generate all possible combinations of meta options
meta_options = list(itertools.product(options, repeat=3))  # Three meta options: include_original, include_rating, include_sentiScore_pos

# Load and prepare data
file_path = './data/filtered_combined_reviews.csv'
data = pd.read_csv(file_path)
labelsPre = data['category']
print()
assert len(labelsPre) == len(data), "ORGINAL DF: Mismatch in number of samples between features and labels"
results = []

# Initialize TextPreprocessor
preprocessor = TextPreprocessor(data)

total = len(keys) * len(meta_options) * len(models)
with tqdm(total=total, desc="Processing configurations") as pbar:
    for model_name in models:
        print(f"Start Processing Config for {model_name}")
        results = []
        for key in keys:
            for original, rating, sentiScore in meta_options:
                labels = preprocessor.data['category']
                features = preprocessor.preprocess(key=key, include_original=original, include_rating=rating, include_sentiScore_pos=sentiScore)
                if len(features) != len(labels):
                    pbar.update(1)
                    print('Misconfigured')
                    continue  # Skip misconfigured setups

                # Train and evaluate classifier
                classifier = TextClassifier(features, labels, model_type= model_name)
                classifier.train()
                accuracy = accuracy_score(classifier.y_test, classifier.predict())
                f1 = f1_score(classifier.y_test, classifier.predict(), average='weighted')

                # Store results
                results.append({
                    'key': key,
                    'include_original': original,
                    'include_rating': rating,
                    'include_sentiScore_pos': sentiScore,
                    'accuracy': accuracy,
                    'f1_score': f1
                })
                pbar.update(1)  # Update progress bar after each configuration
        # Convert results to DataFrame and save
        results_df = pd.DataFrame(results)
        results_df.to_csv(f'./results/NOT/NOT_DROPPED_{model_name}_Result.csv', index=False)

print("All model configurations processed and saved.")
# Add this print just before converting results to DataFrame
# print("Results collected:", results)

# # Convert results to DataFrame
# results_df = pd.DataFrame(results)
# # print("DataFrame Head:", results_df.head())
# results_df.to_csv('./results/RF_Result', index=False)

# Plotting
# fig, ax = plt.subplots(figsize=(12, 8))
# for key, group in results_df.groupby('key'):
#     ax.plot(group['accuracy'], label=f"{key} Accuracy", marker='o')
#     ax.plot(group['f1_score'], label=f"{key} F1-Score", marker='x')

# ax.set_title('Classification Performance by Preprocessing Key and Meta Options')
# ax.set_xlabel('Configuration Index')
# ax.set_ylabel('Score')
# ax.legend()
# plt.xticks(ticks=range(len(results_df)), labels=[f"Config {i+1}" for i in range(len(results_df))], rotation=45)
# plt.tight_layout()
# plt.show()
