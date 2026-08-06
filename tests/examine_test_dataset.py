import joblib
from backend.model.train import load_datasets, unpack_dataset
import numpy as np

def main():
    """
    Compares the distribution of predicted and actual class labels.

    This function loads the trained logistic regression model and the
    testing dataset, generates predictions for the testing dataset, and
    prints the number of articles predicted for each binary truthfulness
    label alongside the number of articles that actually belong to each
    label.

    The function also identifies the most frequently predicted class and
    reports the proportion of testing samples assigned to that class.
    These statistics help identify prediction bias or class imbalance in
    the model's output.

    Args:
        None

    Returns:
        None
    """

    logistic_model = joblib.load("models/logistic_model_v2.pkl")

    _, _, test_dataset = load_datasets()
    X_test_dataset, y_test_dataset = unpack_dataset(test_dataset)

    y_test_prediction = logistic_model.predict(X_test_dataset)

    unique_labels, label_count = np.unique(y_test_prediction, return_counts=True)

    print("Labels predicted by the model: ")
    for label, count in zip(unique_labels, label_count):
        print(f"Class {label}: {count} predictions")
    print("")

    unique_dataset_labels, dataset_label_count = np.unique(y_test_dataset, return_counts=True)
    
    print("Labels in the actual dataset: ")
    for label, count in zip(unique_dataset_labels, dataset_label_count):
        print(f"Class {label}: {count} samples")
    print("")

    most_common_label = np.bincount(y_test_prediction).argmax()
    most_common_label_count = np.sum(y_test_prediction == most_common_label)

    print(f"Most predicted class: {most_common_label}")
    print(f"Percentage: {most_common_label_count / len(y_test_prediction)}")

if __name__ == "__main__":
    main()