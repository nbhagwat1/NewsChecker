import joblib
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from backend.model.train import load_datasets, unpack_dataset, print_stats

def main():
    """
    Evaluates and reports the performance of the trained machine learning model.

    This function loads the trained logistic regression model and the
    processed training, validation, and testing datasets. It evaluates the
    model's predictions on each dataset and prints performance statistics,
    including dataset information, accuracy scores, classification metrics,
    and ROC-AUC.

    The evaluation metrics provide insight into how well the model performs
    when classifying articles as fake news or real news.

    Args:
        None
    
    Returns:
        None
    """

    # Preparing the model and the three datasets for evaluation

    logistic_model = joblib.load("models/logistic_model_v2.pkl")

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

    final_score = roc_auc_score(y_test_dataset, real_probability_dataset)
    print("ROC-AUC:")
    print(final_score)
    print("")

if __name__ == "__main__":
    main()