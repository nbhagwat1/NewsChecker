import joblib
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from backend.model.train import load_datasets, unpack_dataset, print_stats

def main():
    """
    Prints important information about the trained machine learning model's performance
    when it predicts the truthfulness of an article.

    This method loads the trained logistic regression model as well as the training, 
    validation, and testing datasets. The method takes the training, validation, and 
    testing datasets and splits each one into its embeddings and their respective 
    binary truthfulness labels (0 = fake news, 1 = real news). The method then prints 
    information about the number of real news articles and fake news articles in each 
    of the three datasets. It also checks to make sure that the embeddings in each of 
    the three datasets contain valid values and have the same shape as each other. 
    After that, the method has the trained model make predictions on the embeddings
    in the training, validation, and testing datasets, and it prints the model's
    accuracy when making predictions on the embeddings in each of the three datasets.
    Finally, the method prints information about important metrics, such as precision,
    recall, F1-score, and ROC-AUC, that provide valuable data about the trained model's 
    performance when making predictions.

    Args:
        None
    
    Returns:
        None
    """

    # Preparing the model and the three datasets for evaluation

    logistic_model = joblib.load("logistic_model.pkl")

    train_dataset, validate_dataset, test_dataset = load_datasets()

    X_train_dataset, y_train_dataset = unpack_dataset(train_dataset)
    X_validate_dataset, y_validate_dataset = unpack_dataset(validate_dataset)
    X_test_dataset, y_test_dataset = unpack_dataset(test_dataset)

    X_train_dataset = X_train_dataset.astype(np.float32)
    X_validate_dataset = X_validate_dataset.astype(np.float32)
    X_test_dataset = X_test_dataset.astype(np.float32)

    # Printing statistics about each of the three datasets

    print_stats(X_train_dataset, X_validate_dataset, X_test_dataset, y_train_dataset, y_validate_dataset, y_test_dataset)

    # Printing metrics of the model

    y_train_prediction = logistic_model.predict(X_train_dataset)
    train_score = accuracy_score(y_train_prediction, y_train_dataset)
    print(f"Train dataset accuracy: {train_score}")

    y_validate_prediction = logistic_model.predict(X_validate_dataset)
    validate_score = accuracy_score(y_validate_prediction, y_validate_dataset)
    print(f"Validate dataset accuracy: {validate_score}")

    print(f"Order of classes: {logistic_model.classes_}")

    y_test_prediction = logistic_model.predict(X_test_dataset)
    test_score = accuracy_score(y_test_prediction, y_test_dataset)
    print(f"Test dataset accuracy: {test_score}")

    real_probability_dataset = logistic_model.predict_proba(X_test_dataset)[:, 1]
    print("Real probability dataset: ")
    print(real_probability_dataset)

    print("Classification report: ")
    print(classification_report(y_test_dataset, y_test_prediction))
    print("")

    auc = roc_auc_score(y_test_dataset, real_probability_dataset)
    print("ROC-AUC:")
    print(auc)
    print("")

if __name__ == "__main__":
    main()