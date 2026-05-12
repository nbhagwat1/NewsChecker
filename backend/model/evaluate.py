import joblib
import numpy as np
from sklearn.metrics import accuracy_score
from backend.model.train import load_datasets, unpack_dataset, print_stats

def main():
    logistic_model = joblib.load("logistic_model.pkl")

    train_dataset, validate_dataset, test_dataset = load_datasets()

    X_train_dataset, y_train_dataset = unpack_dataset(train_dataset)
    X_validate_dataset, y_validate_dataset = unpack_dataset(validate_dataset)
    X_test_dataset, y_test_dataset = unpack_dataset(test_dataset)

    X_train_dataset = X_train_dataset.astype(np.float32)
    X_validate_dataset = X_validate_dataset.astype(np.float32)
    X_test_dataset = X_test_dataset.astype(np.float32)

    # print_stats(X_train_dataset, X_validate_dataset, X_test_dataset, y_train_dataset, y_validate_dataset, y_test_dataset)

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
    print(f"Real probability dataset: {real_probability_dataset}")

if __name__ == "__main__":
    main()