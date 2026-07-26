import joblib
from backend.model.train import load_datasets, unpack_dataset
import numpy as np

def main():
    """
    Prints information related to how many times the trained machine
    learning model predicts a certain binary truthfulness label for
    all the embeddings in the testing dataset in comparison to how
    many times that particular binary truthfulness label is present
    in the testing dataset.

    This method loads the trained logistic regression model and the
    testing dataset, splits the testing dataset into the embeddings
    and their actual binary truthfulness labels (0 = fake news, 1 = 
    real news), has the trained model predict the binary truthfulness 
    label of each embedding, and compares the total number of times 
    that each binary truthfulness label was predicted by the trained 
    model to the total number of times that each binary truthfulness 
    label is actually present in the testing dataset.

    Args:
        None
    
    Returns:
        None
    """

    logistic_model = joblib.load("logistic_model.pkl")

    train_dataset, validate_dataset, test_dataset = load_datasets()
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