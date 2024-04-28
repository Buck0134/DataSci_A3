# import pandas as pd
# import matplotlib.pyplot as plt
# import numpy as np

# # Load your data into a DataFrame
# data = pd.read_csv('results/experiment_results.csv')

# # # Initialize an empty list to store the results
# # results = []

# # # Unique model types and shuffle data status
# # model_types = data['Model Type'].unique()
# # shuffle_data_statuses = [True, False]

# # # Manually calculate the means
# # for model_type in model_types:
# #     for shuffle in shuffle_data_statuses:
# #         filtered_data = data[(data['Model Type'] == model_type) & (data['Shuffle Data'] == shuffle)]
# #         pre_mean = filtered_data['Pre_Pipeline_F1'].sum() / len(filtered_data)
# #         post_mean = filtered_data['Post_Pipeline_F1'].sum() / len(filtered_data)
# #         results.append({
# #             'Model Type': model_type,
# #             'Shuffle Data': shuffle,
# #             'Pre_Pipeline_F1': pre_mean,
# #             'Post_Pipeline_F1': post_mean,
# #             'F1 Score Difference': post_mean - pre_mean
# #         })

# # # Convert results to DataFrame
# # average_f1_by_shuffling = pd.DataFrame(results)

# # # Plotting
# # fig, ax = plt.subplots(figsize=(12, 8))
# # x = np.arange(len(average_f1_by_shuffling))  # the label locations
# # width = 0.35  # the width of the bars

# # # Plot both pre and post F1 scores
# # rects1 = ax.bar(x - width/2, average_f1_by_shuffling['Pre_Pipeline_F1'], width, label='Pre Pipeline F1', color='skyblue')
# # rects2 = ax.bar(x + width/2, average_f1_by_shuffling['Post_Pipeline_F1'], width, label='Post Pipeline F1', color='orange')

# # # Add some text for labels, title and custom x-axis tick labels, etc.
# # ax.set_ylabel('F1 Scores')
# # ax.set_title('Pre and Post Pipeline F1 Scores by Model Type and Data Shuffling')
# # ax.set_xticks(x)
# # ax.set_xticklabels(average_f1_by_shuffling.apply(lambda x: f"{x['Model Type']} {'Shuffle' if x['Shuffle Data'] else 'No Shuffle'}", axis=1), rotation=45)
# # ax.legend()

# # # Adding text labels for differences on the chart
# # def add_difference_labels(rects1, rects2):
# #     for rect1, rect2 in zip(rects1, rects2):
# #         height1 = rect1.get_height()
# #         height2 = rect2.get_height()
# #         diff = height2 - height1
# #         ax.annotate(f'{diff:.2f}',
# #                     xy=(rect2.get_x() + rect2.get_width() / 2, height2),
# #                     xytext=(0, 3),  # 3 points vertical offset
# #                     textcoords="offset points",
# #                     ha='center', va='bottom')

# # add_difference_labels(rects1, rects2)

# # fig.tight_layout()
# # plt.show()


# # Calculating the maximum pre-pipeline F1 score for each model type
# max_pre_f1 = data.loc[data.groupby(['Model Type'])['Pre_Pipeline_F1'].idxmax()]

# # Calculating the average pre-pipeline F1 score for each model type
# avg_pre_f1 = data.groupby(['Model Type'])['Pre_Pipeline_F1'].mean().reset_index()
# avg_pre_f1.rename(columns={'Pre_Pipeline_F1': 'Average Pre_Pipeline_F1'}, inplace=True)

# # Merging the results for plotting
# max_pre_f1 = max_pre_f1.merge(avg_pre_f1, on=['Model Type'])

# # Label for configurations that achieved the max pre-pipeline F1
# max_pre_f1['config_label'] = max_pre_f1.apply(
#     lambda row: f"Including Rating?: {row['Include Rating']}, Include SentiScore?: {row['Include SentiScore']}", axis=1
# )

# # Plotting
# fig, ax = plt.subplots(figsize=(10, 6))
# x = np.arange(len(max_pre_f1))  # label locations
# width = 0.35  # width of the bars

# rects1 = ax.bar(x - width/2, max_pre_f1['Pre_Pipeline_F1'], width, label='Max Pre-Pipeline F1', color='skyblue')
# rects2 = ax.bar(x + width/2, max_pre_f1['Average Pre_Pipeline_F1'], width, label='Average Pre-Pipeline F1', color='orange')

# # Adding text and labels
# ax.set_ylabel('F1 Scores')
# ax.set_title('Max vs. Average Pre-Pipeline F1 Scores Across Models')
# ax.set_xticks(x)
# ax.set_xticklabels([f"{row['Model Type']}\n{row['config_label']}" for index, row in max_pre_f1.iterrows()], rotation=45, ha="right")
# ax.legend()

# ax.bar_label(rects1, padding=3)
# ax.bar_label(rects2, padding=3)

# fig.tight_layout()
# plt.show()

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Lets load the results CSV files
df1 = pd.read_csv('results/experiment_results_RF.csv')
df2 = pd.read_csv('results/WE_experiment_results_mlp.csv')
df3 = pd.read_csv('results/TFIDF_experiment_results.csv')

# Add a 'source' column to each DataFrame
df1['vectorization'] = 'word_embedding'
df2['vectorization'] = 'word_embedding'
df3['vectorization'] = 'TFIDF'

# Combine all DataFrames into a single DataFrame
combined_df = pd.concat([df1, df2, df3], ignore_index=True)

# Display the combined DataFrame
# print(combined_df)

def VectMethodModel(combined_df):
    # Convert to string to avoid any type issues
    combined_df['vectorization'] = combined_df['vectorization'].astype(str)
    combined_df['Model Type'] = combined_df['Model Type'].astype(str)

    # Group by 'vectorization' and 'Model Type' and calculate mean and max of 'Pre_Pipeline_F1'
    grouped = combined_df.groupby(['vectorization', 'Model Type'])['Pre_Pipeline_F1'].agg(['mean', 'max']).reset_index()

    # Create a new column combining 'vectorization' and 'Model Type' for detailed x-axis labels
    grouped['Vectorization_Model'] = grouped['vectorization'] + '_' + grouped['Model Type']

    # Ensure the new column is string type
    grouped['Vectorization_Model'] = grouped['Vectorization_Model'].astype(str)

    # Melt the DataFrame to make it suitable for seaborn's barplot, only include 'mean' and 'max'
    melted_df = pd.melt(grouped, id_vars=['Vectorization_Model'], value_vars=['mean', 'max'], var_name='Statistic', value_name='Pre_Pipeline_F1')

    # Ensure the 'Statistic' column is also string type
    melted_df['Statistic'] = melted_df['Statistic'].astype(str)

    # Create the plot
    plt.figure(figsize=(12, 8))
    sns.barplot(data=melted_df, x='Vectorization_Model', hue='Statistic', y='Pre_Pipeline_F1', ci=None)
    plt.title('Average and Maximum Pre_Pipeline F1 by Vectorization Method and Model Type')
    plt.xlabel('Vectorization Method and Model Type')
    plt.ylabel('Pre_Pipeline F1 Score')
    plt.xticks(rotation=45)  # Rotates labels to avoid overlap
    plt.legend(title='Statistic')

    # Show the plot
    plt.show()

def VectMethodModel_Rating(combined_df):
    # Convert to string to avoid any type issues
    combined_df['vectorization'] = combined_df['vectorization'].astype(str)
    combined_df['Model Type'] = combined_df['Model Type'].astype(str)
    combined_df['Include Rating'] = combined_df['Include Rating'].astype(str)  # Ensure this is also a string

    # Group by 'vectorization', 'Model Type', and 'Include Rating', then calculate mean and max of 'Pre_Pipeline_F1'
    grouped = combined_df.groupby(['vectorization', 'Model Type', 'Include Rating'])['Pre_Pipeline_F1'].agg(['mean', 'max']).reset_index()

    # Create a new column combining 'vectorization' and 'Model Type' for detailed x-axis labels
    grouped['Vectorization_Model'] = grouped['vectorization'] + '_' + grouped['Model Type']

    # Melt the DataFrame to make it suitable for seaborn's barplot, only include 'mean' and 'max'
    melted_df = pd.melt(grouped, id_vars=['Vectorization_Model', 'Include Rating'], value_vars=['mean', 'max'], var_name='Statistic', value_name='Pre_Pipeline_F1')

    # Create a new 'Hue' column combining 'Include Rating' and 'Statistic'
    melted_df['Hue'] = melted_df['Include Rating'] + ' ' + melted_df['Statistic']

    # Create the plot
    plt.figure(figsize=(14, 8))
    sns.barplot(data=melted_df, x='Vectorization_Model', hue='Hue', y='Pre_Pipeline_F1', ci=None)
    plt.title('Average and Maximum Pre_Pipeline F1 by Vectorization Method, Model Type, and Rating Inclusion')
    plt.xlabel('Vectorization Method and Model Type')
    plt.ylabel('Pre_Pipeline F1 Score')
    plt.xticks(rotation=45)  # Rotates labels to avoid overlap
    plt.legend(title='Rating and Statistic', bbox_to_anchor=(1.05, 1), loc='upper left')

    # Show the plot
    plt.show()

def VectMethodModel_SentiScore(combined_df):
    # Convert to string to avoid any type issues
    combined_df['vectorization'] = combined_df['vectorization'].astype(str)
    combined_df['Model Type'] = combined_df['Model Type'].astype(str)
    combined_df['Include SentiScore'] = combined_df['Include SentiScore'].astype(str)  # Ensure this is also a string

    # Group by 'vectorization', 'Model Type', and 'Include SentiScore', then calculate mean and max of 'Pre_Pipeline_F1'
    grouped = combined_df.groupby(['vectorization', 'Model Type', 'Include SentiScore'])['Pre_Pipeline_F1'].agg(['mean', 'max']).reset_index()

    # Create a new column combining 'vectorization' and 'Model Type' for detailed x-axis labels
    grouped['Vectorization_Model'] = grouped['vectorization'] + '_' + grouped['Model Type']

    # Melt the DataFrame to make it suitable for seaborn's barplot, only include 'mean' and 'max'
    melted_df = pd.melt(grouped, id_vars=['Vectorization_Model', 'Include SentiScore'], value_vars=['mean', 'max'], var_name='Statistic', value_name='Pre_Pipeline_F1')

    # Create a new 'Hue' column combining 'Include SentiScore' and 'Statistic'
    melted_df['Hue'] = melted_df['Include SentiScore'] + ' ' + melted_df['Statistic']

    # Create the plot
    plt.figure(figsize=(14, 8))
    sns.barplot(data=melted_df, x='Vectorization_Model', hue='Hue', y='Pre_Pipeline_F1', ci=None, palette='tab10')
    plt.title('Average and Maximum Pre_Pipeline F1 by Vectorization Method, Model Type, and Sentiment Score Inclusion')
    plt.xlabel('Vectorization Method and Model Type')
    plt.ylabel('Pre_Pipeline F1 Score')
    plt.xticks(rotation=45)  # Rotates labels to avoid overlap
    plt.legend(title='Include Sentiment Score and Statistic', bbox_to_anchor=(1.05, 1), loc='upper left')

    # Show the plot
    plt.show()

def VectMethodModel_Shuffle(combined_df):
    # Convert to string to avoid any type issues
    combined_df['vectorization'] = combined_df['vectorization'].astype(str)
    combined_df['Model Type'] = combined_df['Model Type'].astype(str)
    combined_df['Shuffle Data'] = combined_df['Shuffle Data'].astype(str)  # Ensure this is also a string

    # Group by 'vectorization', 'Model Type', and 'Shuffle Data', then calculate mean and max of 'Pre_Pipeline_F1'
    grouped = combined_df.groupby(['vectorization', 'Model Type', 'Shuffle Data'])['Pre_Pipeline_F1'].agg(['mean', 'max']).reset_index()

    # Create a new column combining 'vectorization' and 'Model Type' for detailed x-axis labels
    grouped['Vectorization_Model'] = grouped['vectorization'] + '_' + grouped['Model Type']

    # Melt the DataFrame to make it suitable for seaborn's barplot, only include 'mean' and 'max'
    melted_df = pd.melt(grouped, id_vars=['Vectorization_Model', 'Shuffle Data'], value_vars=['mean', 'max'], var_name='Statistic', value_name='Pre_Pipeline_F1')

    # Create a new 'Hue' column combining 'Shuffle Data' and 'Statistic'
    melted_df['Hue'] = melted_df['Shuffle Data'] + ' ' + melted_df['Statistic']

    # # Calculate overall mean and max to add to the plot
    # overall_mean = combined_df['Pre_Pipeline_F1'].mean()
    # overall_max = combined_df['Pre_Pipeline_F1'].max()
    # overall_summary = pd.DataFrame({
    #     'Vectorization_Model': ['Overall'],
    #     'Pre_Pipeline_F1': [overall_mean, overall_max],
    #     'Statistic': ['mean', 'max'],
    #     'Shuffle Data': ['All', 'All'],
    #     'Hue': ['All mean', 'All max']
    # })

    # # Append overall summary to the melted DataFrame
    # melted_df = pd.concat([melted_df, overall_summary], ignore_index=True)

    # Create the plot
    plt.figure(figsize=(14, 8))
    sns.barplot(data=melted_df, x='Vectorization_Model', hue='Hue', y='Pre_Pipeline_F1', ci=None, palette='tab10')
    plt.title('Average and Maximum Pre_Pipeline F1 by Vectorization Method, Model Type, and Data Shuffling')
    plt.xlabel('Vectorization Method and Model Type')
    plt.ylabel('Pre_Pipeline F1 Score')
    plt.xticks(rotation=45)  # Rotates labels to avoid overlap
    plt.legend(title='Shuffled Data and Statistic', bbox_to_anchor=(1.05, 1), loc='upper left')

    # Show the plot
    plt.show()

def VectMethodModel_PipelinePerformance(combined_df):
    # Convert to string to avoid any type issues
    combined_df['vectorization'] = combined_df['vectorization'].astype(str)
    combined_df['Model Type'] = combined_df['Model Type'].astype(str)

    # Group by 'vectorization' and 'Model Type', calculate mean of 'Pre_Pipeline_F1' and 'Post_Pipeline_F1'
    grouped = combined_df.groupby(['vectorization', 'Model Type']).agg({
        'Pre_Pipeline_F1': ['mean', 'max'],
        'Post_Pipeline_F1': ['mean', 'max']
    }).reset_index()

    # Flatten MultiIndex columns caused by aggregation
    grouped.columns = [' '.join(col).strip() for col in grouped.columns.values]

    # Create a new column combining 'vectorization' and 'Model Type' for x-axis labels
    grouped['Vectorization_Model'] = grouped['vectorization'] + '_' + grouped['Model Type']

    # Calculate overall mean and max for Pre and Post Pipeline F1
    overall_stats = combined_df.agg({
        'Pre_Pipeline_F1': ['mean', 'max'],
        'Post_Pipeline_F1': ['mean', 'max']
    })

    # Adding overall data to the dataframe
    overall_data = pd.DataFrame({
        'Vectorization_Model': ['Overall Mean', 'Overall Max'],
        'Pre_Pipeline_F1 mean': [overall_stats['Pre_Pipeline_F1']['mean'], overall_stats['Pre_Pipeline_F1']['max']],
        'Post_Pipeline_F1 mean': [overall_stats['Post_Pipeline_F1']['mean'], overall_stats['Post_Pipeline_F1']['max']],
        'Pre_Pipeline_F1 max': [overall_stats['Pre_Pipeline_F1']['mean'], overall_stats['Pre_Pipeline_F1']['max']],
        'Post_Pipeline_F1 max': [overall_stats['Post_Pipeline_F1']['mean'], overall_stats['Post_Pipeline_F1']['max']]
    })

    # Append overall summary to the grouped DataFrame
    grouped = pd.concat([grouped, overall_data], ignore_index=True)

    # Melt the DataFrame to make it suitable for seaborn's barplot
    melted_df = pd.melt(grouped, id_vars=['Vectorization_Model'], value_vars=['Pre_Pipeline_F1 mean', 'Post_Pipeline_F1 mean', 'Pre_Pipeline_F1 max', 'Post_Pipeline_F1 max'], var_name='Pipeline Stage', value_name='F1 Score')

    # Create the plot
    plt.figure(figsize=(20, 8))
    barplot = sns.barplot(data=melted_df, x='Vectorization_Model', hue='Pipeline Stage', y='F1 Score')

    # Annotations for improvements
    grouped['Improvement Mean'] = grouped['Post_Pipeline_F1 mean'] - grouped['Pre_Pipeline_F1 mean']
    grouped['Improvement Max'] = grouped['Post_Pipeline_F1 max'] - grouped['Pre_Pipeline_F1 max']
    for index, row in grouped.iterrows():
        plt.text(index - 0.2, row['Post_Pipeline_F1 mean'] + 0.02, f'M+{row["Improvement Mean"]:.2f}', color='black', ha='center')
        plt.text(index + 0.2, row['Post_Pipeline_F1 max'] + 0.02, f'Mx+{row["Improvement Max"]:.2f}', color='blue', ha='center')

    plt.title('Pre and Post Pipeline F1 Scores by Vectorization Method and Model Type with Improvements')
    plt.xlabel('Vectorization Method and Model Type')
    plt.ylabel('F1 Score')
    plt.xticks(rotation=90)
    plt.legend(title='Pipeline Stage')

    # Show the plot
    plt.show()

def TextualFeaturePipelinePerformance(combined_df):
    # Ensure that 'Textual Feature' column is in string form
    combined_df['Textual Feature'] = combined_df['Textual Feature'].astype(str)

    # Group by 'Textual Feature', calculate mean and max of 'Pre_Pipeline_F1' and 'Post_Pipeline_F1'
    grouped = combined_df.groupby('Textual Feature').agg({
        'Pre_Pipeline_F1': ['mean', 'max'],
        'Post_Pipeline_F1': ['mean', 'max']
    }).reset_index()

    # Flatten MultiIndex columns caused by aggregation
    grouped.columns = [' '.join(col).strip() for col in grouped.columns.values]

    # Calculate overall mean and max for Pre and Post Pipeline F1
    overall_stats = combined_df.agg({
        'Pre_Pipeline_F1': ['mean', 'max'],
        'Post_Pipeline_F1': ['mean', 'max']
    })

    # Adding overall data to the dataframe
    overall_data = pd.DataFrame({
        'Textual Feature': ['Overall Mean', 'Overall Max'],
        'Pre_Pipeline_F1 mean': [overall_stats['Pre_Pipeline_F1']['mean'], overall_stats['Pre_Pipeline_F1']['max']],
        'Post_Pipeline_F1 mean': [overall_stats['Post_Pipeline_F1']['mean'], overall_stats['Post_Pipeline_F1']['max']],
        'Pre_Pipeline_F1 max': [overall_stats['Pre_Pipeline_F1']['mean'], overall_stats['Pre_Pipeline_F1']['max']],
        'Post_Pipeline_F1 max': [overall_stats['Post_Pipeline_F1']['mean'], overall_stats['Post_Pipeline_F1']['max']]
    })

    # Append overall summary to the grouped DataFrame
    grouped = pd.concat([grouped, overall_data], ignore_index=True)

    # Melt the DataFrame to make it suitable for seaborn's barplot
    melted_df = pd.melt(grouped, id_vars=['Textual Feature'], value_vars=['Pre_Pipeline_F1 mean', 'Post_Pipeline_F1 mean', 'Pre_Pipeline_F1 max', 'Post_Pipeline_F1 max'], var_name='Pipeline Stage', value_name='F1 Score')

    # Create the plot
    plt.figure(figsize=(16, 8))
    barplot = sns.barplot(data=melted_df, x='Textual Feature', hue='Pipeline Stage', y='F1 Score', palette='coolwarm')

    # Annotations for improvements
    grouped['Improvement Mean'] = grouped['Post_Pipeline_F1 mean'] - grouped['Pre_Pipeline_F1 mean']
    grouped['Improvement Max'] = grouped['Post_Pipeline_F1 max'] - grouped['Pre_Pipeline_F1 max']
    for index, row in grouped.iterrows():
        plt.text(index - 0.2, row['Post_Pipeline_F1 mean'] + 0.02, f'M+{row["Improvement Mean"]:.2f}', color='black', ha='center')
        plt.text(index + 0.2, row['Post_Pipeline_F1 max'] + 0.02, f'Mx+{row["Improvement Max"]:.2f}', color='blue', ha='center')

    plt.title('Pre and Post Pipeline F1 Scores by Textual Features with Improvements')
    plt.xlabel('Textual Features')
    plt.ylabel('F1 Score')
    plt.xticks(rotation=45)
    plt.legend(title='Pipeline Stage')

    # Show the plot
    plt.show()

def PipelineChangeVisualization(combined_df):
    # Ensure all necessary columns are in string form for grouping
    combined_df['Textual Feature'] = combined_df['Textual Feature'].astype(str)
    combined_df['vectorization'] = combined_df['vectorization'].astype(str)
    combined_df['Model Type'] = combined_df['Model Type'].astype(str)

    # Create a new column combining 'Textual Feature', 'vectorization', and 'Model Type'
    combined_df['Feature_Vector_Model'] = combined_df['Textual Feature'] + '_' + combined_df['vectorization'] + '_' + combined_df['Model Type']

    # Calculate the change in F1 score from pre to post pipeline
    combined_df['F1 Score Change'] = combined_df['Post_Pipeline_F1'] - combined_df['Pre_Pipeline_F1']

    # Group by the new combined column to prepare for visualization
    grouped = combined_df.groupby('Feature_Vector_Model')['F1 Score Change'].mean().reset_index()

    # Sort the results to make the plot more readable
    grouped = grouped.sort_values(by='F1 Score Change', ascending=False)

    # Create the plot
    plt.figure(figsize=(20, 10))
    barplot = sns.barplot(data=grouped, x='Feature_Vector_Model', y='F1 Score Change', palette='vlag')

    plt.title('Change in Pipeline F1 Score by Textual Feature, Vectorization Method, and Model Type')
    plt.xlabel('Textual Feature, Vectorization Method, and Model Type Combination')
    plt.ylabel('Average F1 Score Change (Post - Pre)')
    plt.xticks(rotation=90)

    # Show the plot
    plt.show()

def PipelineChangeByConfidence(combined_df):
    # Ensure the 'Confidence Threshold' column is treated as a string for grouping purposes
    combined_df['Confidence Threshold'] = combined_df['Confidence Threshold'].astype(str)

    # Calculate the change in F1 score from pre to post pipeline
    combined_df['F1 Score Change'] = combined_df['Post_Pipeline_F1'] - combined_df['Pre_Pipeline_F1']

    # Group by 'Confidence Threshold' to prepare for visualization
    grouped = combined_df.groupby('Confidence Threshold')['F1 Score Change'].mean().reset_index()

    # Sort the results to make the plot more readable
    grouped = grouped.sort_values(by='F1 Score Change', ascending=False)

    # Create the plot
    plt.figure(figsize=(12, 6))
    barplot = sns.barplot(data=grouped, x='Confidence Threshold', y='F1 Score Change', palette='vlag')

    plt.title('Change in Pipeline F1 Score by Confidence Threshold')
    plt.xlabel('Confidence Threshold')
    plt.ylabel('Average F1 Score Change (Post - Pre)')
    plt.xticks(rotation=45)

    # Show the plot
    plt.show()

def TextualFeaturesPerformance(combined_df):
    # Ensure the 'Textual Feature' column is in string form
    combined_df['Textual Feature'] = combined_df['Textual Feature'].astype(str)

    # Group by 'Textual Feature', calculate mean and max of 'Pre_Pipeline_F1'
    grouped = combined_df.groupby('Textual Feature')['Pre_Pipeline_F1'].agg(['mean', 'max']).reset_index()

    # Melt the DataFrame to make it suitable for seaborn's barplot
    melted_df = pd.melt(grouped, id_vars=['Textual Feature'], value_vars=['mean', 'max'], var_name='Metric', value_name='F1 Score')

    # Create the plot
    plt.figure(figsize=(12, 6))
    barplot = sns.barplot(data=melted_df, x='Textual Feature', hue='Metric', y='F1 Score')

    plt.title('Mean and Maximum Pre Pipeline F1 Scores by Textual Features')
    plt.xlabel('Textual Features')
    plt.ylabel('F1 Score')
    plt.xticks(rotation=45)

    # Show the plot
    plt.show()

def TimeComparisonByVectModel(combined_df):
    # Ensure that 'vectorization' and 'Model Type' columns are in string form
    combined_df['vectorization'] = combined_df['vectorization'].astype(str)
    combined_df['Model Type'] = combined_df['Model Type'].astype(str)

    # Create a new column combining 'vectorization' and 'Model Type' for x-axis labels
    combined_df['Vectorization_Model'] = combined_df['vectorization'] + '_' + combined_df['Model Type']

    # Group by the new combined column, calculate mean time if multiple entries exist
    grouped = combined_df.groupby('Vectorization_Model')['Time Taken'].mean().reset_index()

    # Sort the results to make the plot more readable, especially if there are many categories
    grouped = grouped.sort_values(by='Time Taken', ascending=True)

    # Create the plot
    plt.figure(figsize=(14, 7))
    barplot = sns.barplot(data=grouped, x='Vectorization_Model', y='Time Taken', palette='viridis')

    plt.title('Computational Time Comparison by Vectorization Method and Model Type')
    plt.xlabel('Vectorization Method and Model Type')
    plt.ylabel('Mean Computational Time (seconds)')
    plt.xticks(rotation=45)

    # Show the plot
    plt.show()

def TimeComparisonByTextualFeature(combined_df):
    # Ensure that 'Textual Feature' column is in string form
    combined_df['Textual Feature'] = combined_df['Textual Feature'].astype(str)

    # Group by 'Textual Feature', calculate mean time if multiple entries exist
    grouped = combined_df.groupby('Textual Feature')['Time Taken'].mean().reset_index()

    # Sort the results to make the plot more readable, especially if there are many categories
    grouped = grouped.sort_values(by='Time Taken', ascending=True)

    # Create the plot
    plt.figure(figsize=(12, 6))
    barplot = sns.barplot(data=grouped, x='Textual Feature', y='Time Taken', palette='magma')

    plt.title('Computational Time Comparison by Textual Features')
    plt.xlabel('Textual Features')
    plt.ylabel('Mean Computational Time (seconds)')
    plt.xticks(rotation=45)

    # Show the plot
    plt.show()

# Plot 1: vectorization and model's average and max
VectMethodModel(combined_df)

# Plot 2: Including Rating?
VectMethodModel_Rating(combined_df)

# Plot 3: Including SentiScore?
VectMethodModel_SentiScore(combined_df)

# Plot 4: Shuffled Data or Not?
VectMethodModel_Shuffle(combined_df)

# Plots 5: Pre/Post Pipeline Performance
PipelineChangeByConfidence(combined_df)
PipelineChangeVisualization(combined_df)
VectMethodModel_PipelinePerformance(combined_df)
TextualFeaturePipelinePerformance(combined_df)

# Plot 6: Textual Processing Techniques Comparsion
TextualFeaturesPerformance(combined_df)

# Plot 7: Computational Time Spent Comparsion
TimeComparisonByVectModel(combined_df)

# Plot 8: Computational Time Spent Comparsion
TimeComparisonByTextualFeature(combined_df)