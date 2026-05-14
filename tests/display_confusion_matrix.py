from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import joblib
from backend.model.train import load_datasets, unpack_dataset

def main():
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