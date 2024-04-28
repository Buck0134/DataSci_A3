# Data Science for Software Engineers, Assignment 3, Comments Classification

## Overview

The goal of assignment 3 is to explore, implement, and compare different natural language processing techniques. Please use the following steps to replicate the findings in the system. 

## Getting Started

### Prerequisites

Before you begin, ensure you have Python 3 installed on your system.

### Installation

1. **Set up the virtual ENV**

    Begin by setting up the dependencies with python virtual env. This step will install a virtual environment to the code base and download all necessary dependencies. 

    ```sh
    sh start.sh
    ```

2. **Activate the virtual environment**

    Activate the virtual environment to isolate your package dependencies:

    ```sh
    source myenv/bin/activate
    ```

### Usage

To replicate findings reported: run the following command. 
1. **Run Training Pipeline with a Signle Instance**

    The default signle pipeline instance is setted up with Multiclass model 'random_forest' with textual feature'stopwords_removal_lemmatization' processed by TF-IDF and four seconding binary poly kernel SVM models, initaied when the confidence level of the MC result is lower than 0.7. 

    ```sh
    python3 main.py
    ```
2. **Run 2 experiments with a total of 640 different configurations**

   The default set up will first run 320 sets of congigurations based on TF-IFD transfomer. To experiment with words embedding, rerun experiments.py after commenting out the preprocess_text_data with TFIDF function and uncommenting the preprocess_text_data with words embeddings in main.py

    ```sh
    python3 experiments.py
    ```
4. **Run Visualization of experiments result based on pre-ran experiments results**

   The default set up will run all 11 visualization, comment out any unwanted plotting as wishes. The data plotted is located under result folder.  

    ```sh
    python3 dataVisualization.py
    ```
