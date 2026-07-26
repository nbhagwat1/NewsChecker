from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import joblib
from backend.model.train import load_datasets, unpack_dataset

def main():
    """
    Displays the confusion matrix that represents the trained machine
    learning model's performance when predicting the binary truthfulness
    label for the embeddings in the testing dataset.

    This method loads the trained logistic regression model and the
    testing dataset, has the model predict the binary truthfulness label
    (0 = fake news, 1 = real news) for each embedding in the testing
    dataset, and displays the confusion matrix that describes the trained
    model's performance when it predicts the binary truthfulness label of
    each embedding in the testing dataset and how that predicted label
    compares to the actual binary truthfulness label of each embedding
    in the testing dataset.

    Args:
        None
    
    Returns:
        None
    """

    logistic_model = joblib.load("logistic_model.pkl")

    train_dataset, validate_dataset, test_dataset = load_datasets()
    X_test_dataset, y_test_dataset = unpack_dataset(test_dataset)

    y_test_prediction = logistic_model.predict(X_test_dataset)

    cm = confusion_matrix(y_test_dataset, y_test_prediction)
    display = ConfusionMatrixDisplay(confusion_matrix=cm)
    display.plot()
    plt.show()

if __name__ == "__main__":
    main()