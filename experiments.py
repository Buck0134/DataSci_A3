import pandas as pd
from main import pipeline
import itertools
from tqdm import tqdm
import spacy

# Define the configurations
model_types = ['random_forest','naive_bayes', 'mlp']
textual_feature_selection = ['comment', 'lemmatized_comment', 'stopwords_removal_nltk', 'stopwords_removal_lemmatization']
confidence_thresholds = [0.5, 0.6, 0.7, 0.8]
include_ratings = [True, False]
include_sentiScore_pos = [True, False]
shuffle_datas = [True, False]
use_grid_searches = [False]

configurations = list(itertools.product(model_types, textual_feature_selection, confidence_thresholds, include_ratings, include_sentiScore_pos, shuffle_datas, use_grid_searches))
nlp = spacy.load('en_core_web_md')  # Load the medium model with vectors

config_dicts = [
    {   'nlp': nlp,
        'model_type': config[0],
        'textual_feature': config[1],  # New feature selection parameter
        'confidence_threshold': config[2],
        'include_rating': config[3],
        'include_sentiScore_pos': config[4],
        'shuffle_data': config[5],
        'use_grid_search': config[6]
    }
    for config in configurations
]
# Initialize an empty DataFrame to store results
results = []

# Loop through configurations and run the pipeline
for config in tqdm(config_dicts, desc="Running experiments", unit="config"):
    mc_perf, binary_perf, time_taken = pipeline(**config)
    result ={
        'Model Type': config['model_type'],
        'Textual Feature': config['textual_feature'],  # Log the feature selection used
        'Confidence Threshold': config['confidence_threshold'],
        'Include Rating': config['include_rating'],
        'Include SentiScore': config['include_sentiScore_pos'],
        'Shuffle Data': config['shuffle_data'],
        'Use Grid Search': config['use_grid_search'],
        'MC Accuracy': mc_perf['accuracy'],  # Assuming 'accuracy' is a key in the returned dict
        'Binary Accuracy': binary_perf['accuracy'],  # Same assumption as above
        'Time Taken': time_taken
    }
    results.append(result)

results_df = pd.DataFrame(results)

# Save the results to CSV for further analysis
results_df.to_csv('./results/TFIDF_experiment_results.csv', index=False)
print("Experiments completed and results saved.")