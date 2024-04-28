from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

class TextClassifier:
    def __init__(self, X_train, y_train, X_test, y_test, model_type, use_grid_search=False, random_state=42):
        """
        Initialize the TextClassifier with the training and testing data, and the model type.
        
        :param X_train: Training data features.
        :param y_train: Training data labels.
        :param X_test: Testing data features.
        :param y_test: Testing data labels.
        :param model_type: Type of the model to train ('random_forest', 'naive_bayes', 'mlp').
        :param random_state: A seed used by the random number generator.
        """
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.use_grid_search = use_grid_search
        self.model = self.select_model(model_type, random_state)
    
    def select_model(self, model_type, random_state):
        models = {
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=random_state),
            'naive_bayes': MultinomialNB(),
            'mlp': MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, random_state=random_state),
            'svm': SVC(kernel='poly', probability=True, random_state=random_state) 
        }
        param_grid = {
            'random_forest': {'n_estimators': [100, 200, 300, 400, 500], 'max_features': ['auto', 'sqrt', 'log2']},
            'naive_bayes': {'alpha': [1.0, 0.5, 0.1]},
            'mlp': {'hidden_layer_sizes': [(100,), (50, 50)], 'activation': ['tanh', 'relu']},
            'svm': {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf', 'poly']}  # SVM parameters
        }
        if model_type not in models:
            raise ValueError(f"Unsupported model type {model_type}. Available types: {', '.join(models.keys())}")

        if self.use_grid_search:
            model = GridSearchCV(models[model_type], param_grid[model_type], cv=5, scoring='accuracy')
        else:
            model = models[model_type]

        return model

    def train(self):
        """
        Train the model using the training data.
        """
        self.model.fit(self.X_train, self.y_train)
        if self.use_grid_search:
            pass
            # print("Best parameters found:", self.model.best_params_)
            # print("Best cross-validation score: {:.4f}".format(self.model.best_score_))
        else:
            pass
            # print("Model training completed without grid search.")

    def predict(self, X=None):
        if X is None:
            X = self.X_test
        return self.model.predict(X)

    def evaluate(self, experiement_Model = False):
        """
        Evaluate the model on the test set and print F1 score and accuracy.
        """
        predictions = self.predict()
        # print("Classification Report:")
        # print(classification_report(self.y_test, predictions))
        # print(f"Accuracy: {accuracy_score(self.y_test, predictions):.4f}")
        # print(f"F1 Score (Weighted): {f1_score(self.y_test, predictions, average='weighted'):.4f}")
        return classification_report(self.y_test, predictions, output_dict=experiement_Model)

    def predict_proba(self, X):
        """
        Predict class probabilities for the given data using the trained model.
        
        :param X: Data for which to predict class probabilities.
        """
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        else:
            raise NotImplementedError("This model does not support probability predictions.")
