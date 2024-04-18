from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier

class TextClassifier:
    def __init__(self, features, labels, model_type, test_size=0.2, random_state=42):
        """
        Initialize the TextClassifier with the preprocessed features and labels.
        
        :param features: Preprocessed features from TextPreprocessor.
        :param labels: Labels for the dataset.
        :param test_size: Proportion of the dataset to include in the test split.
        :param random_state: A seed used by the random number generator for splitting the data.
        """
        self.features = features
        self.labels = labels
        self.test_size = test_size
        self.random_state = random_state
        self.model = self.select_model(model_type, random_state)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.features, self.labels, test_size=self.test_size, random_state=self.random_state
        )
    
    def select_model(self, model_type, random_state):
        models = {
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=random_state),
            'naive_bayes': MultinomialNB(),
            'mlp': MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, random_state=random_state)
        }
        param_grid = {
            'random_forest': {'n_estimators': [100, 200, 300, 400, 500], 'max_features': ['auto', 'sqrt', 'log2']},
            'naive_bayes': {'alpha': [1.0, 0.5, 0.1]},
            'mlp': {'hidden_layer_sizes': [(100,), (50, 50)], 'activation': ['tanh', 'relu']}
        }
        if model_type in models:
            grid_search = GridSearchCV(models[model_type], param_grid[model_type], cv=5, scoring='accuracy')
            return grid_search
        else:
            raise ValueError(f"Unsupported model type {model_type}. Available types: {', '.join(models.keys())}.")

    def train(self):
        """
        Train the model using the training data.
        """
        self.model.fit(self.X_train, self.y_train)
        print("Best parameters found:", self.model.best_params_)
        print("Best cross-validation score: {:.4f}".format(self.model.best_score_))

    def predict(self):
        """
        Predict labels for the test data using the trained model.
        """
        return self.model.predict(self.X_test)

    def evaluate(self):
        """
        Evaluate the model on the test set and print F1 score and accuracy.
        """
        predictions = self.predict()
        print("Classification Report:")
        print(classification_report(self.y_test, predictions))
        print(f"Accuracy: {accuracy_score(self.y_test, predictions):.4f}")
        print(f"F1 Score (Weighted): {f1_score(self.y_test, predictions, average='weighted'):.4f}")

