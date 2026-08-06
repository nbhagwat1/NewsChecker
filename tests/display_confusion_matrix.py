from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import joblib
from backend.model.train import load_datasets, unpack_dataset

def main():
    """
    Displays the confusion matrix for the trained logistic regression model.

    This function loads the trained logistic regression model and the
    testing dataset, generates predictions for the testing dataset, and
    displays a confusion matrix comparing the model's predicted binary
    truthfulness labels with the ground-truth labels.

    The confusion matrix provides a visual summary of the model's
    classification performance by showing the numbers of true positives,
    true negatives, false positives, and false negatives.

    Args:
        None

    Returns:
        None
    """

    logistic_model = joblib.load("models/logistic_model_v2.pkl")

    _, _, test_dataset = load_datasets()
    X_test_dataset, y_test_dataset = unpack_dataset(test_dataset)

    y_test_prediction = logistic_model.predict(X_test_dataset)

    final_matrix = confusion_matrix(y_test_dataset, y_test_prediction)
    display = ConfusionMatrixDisplay(confusion_matrix=final_matrix)
    display.plot()
    plt.show()

if __name__ == "__main__":
    main()