import joblib
import os
import numpy as np
from sklearn.linear_model import LogisticRegression

def load_datasets():
    """
    Loads the processed training, validation, and testing datasets.

    This function loads the saved datasets from their respective .npy files
    and returns them for use during model training, validation, and
    evaluation.

    Args:
        None

    Returns:
        list[tuple[np.ndarray, int]]: The training dataset containing article
            embeddings and their corresponding binary truthfulness labels.
        list[tuple[np.ndarray, int]]: The validation dataset containing article
            embeddings and their corresponding binary truthfulness labels.
        list[tuple[np.ndarray, int]]: The testing dataset containing article
            embeddings and their corresponding binary truthfulness labels.

    """

    # Create the path to the clean data directory relative to the project
    # location so the saved datasets can be located consistently regardless
    # of where the script is run from.
    DATA_LOCATION = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "clean")

    # Create file paths for each saved dataset.
    train_file = os.path.join(DATA_LOCATION, "training_dataset.npy")
    validate_file = os.path.join(DATA_LOCATION, "validating_dataset.npy")
    test_file = os.path.join(DATA_LOCATION, "testing_dataset.npy")

    # Load the saved datasets from NumPy files.
    train_dataset = np.load(train_file, allow_pickle=True)
    validate_dataset = np.load(validate_file, allow_pickle=True)
    test_dataset = np.load(test_file, allow_pickle=True)

    return train_dataset, validate_dataset, test_dataset

def unpack_dataset(dataset):
    """
    Separates article embeddings and labels for model training.

    This function accepts a dataset containing article embeddings paired with
    their corresponding binary truthfulness labels. It separates the
    embeddings into a feature matrix and the labels into a target array so
    they can be used independently by machine learning models.

    Args:
        dataset (list[tuple[np.ndarray, int]]): A list of tuples where each
            tuple contains an article embedding and its corresponding binary
            truthfulness label.

    Returns:
        np.ndarray: A 2D array containing the article embeddings, where each
            row represents one article's features.
        np.ndarray: A 1D array containing the binary truthfulness labels for
            each article.
    """

    X_list = []
    y_list = []

    for sample in dataset:
        average_embedding = sample[0]
        label = sample[1]

        X_list.append(average_embedding)
        y_list.append(label)
    
    # Stack the article embedding vectors into a 2D NumPy array where each
    # row represents one article's features for model input.
    X_dataset = np.stack(X_list)

    # Convert the article labels into a NumPy array for model training.
    y_dataset = np.array(y_list)

    return X_dataset, y_dataset

def main():
    """
    Trains and saves the logistic regression model.

    This function loads the processed training dataset, separates the article
    embeddings from their corresponding binary truthfulness labels, and trains
    a logistic regression model using the training data. The trained model is
    saved to a file so it can be loaded later for predictions without needing
    to be retrained.

    The model uses balanced class weights to account for differences in the
    number of real and fake news examples in the training dataset.

    Args:
        None

    Returns:
        None
    """

    train_dataset, validate_dataset, test_dataset = load_datasets()

    X_train_dataset, y_train_dataset = unpack_dataset(train_dataset)

    # Convert the feature data to float32 format to ensure compatibility
    # with the machine learning model while reducing memory usage.
    X_train_dataset = X_train_dataset.astype(np.float32)

    # Allow the model enough training iterations to find a stable solution.
    # Balance class weights because the dataset contains more real than fake articles.
    logistic_model = LogisticRegression(max_iter=1000, class_weight='balanced')

    logistic_model.fit(X_train_dataset, y_train_dataset)

    # Save the trained model so it can be loaded later for predictions
    # without retraining.
    joblib.dump(logistic_model, "logistic_model_v2.pkl")

def print_stats(X_train_dataset, X_validate_dataset, X_test_dataset, y_train_dataset, y_validate_dataset, y_test_dataset):
    """
    Prints summary statistics for the training, validation, and testing datasets.

    This function reports statistics for the processed datasets, including
    the number of real and fake news articles in each dataset, dataset
    dimensions, whether all article embeddings have the expected shape, and
    whether any embeddings contain invalid numerical values such as NaN or
    infinity.

    These statistics help verify that the datasets were prepared correctly
    before training or evaluating the machine learning model.

    Args:
        X_train_dataset (np.ndarray): Feature matrix containing the training
            article embeddings.
        X_validate_dataset (np.ndarray): Feature matrix containing the
            validation article embeddings.
        X_test_dataset (np.ndarray): Feature matrix containing the testing
            article embeddings.
        y_train_dataset (np.ndarray): Binary truthfulness labels for the
            training dataset.
        y_validate_dataset (np.ndarray): Binary truthfulness labels for the
            validation dataset.
        y_test_dataset (np.ndarray): Binary truthfulness labels for the
            testing dataset.

    Returns:
        None
    """

    print(f"TOTAL ARTICLES: {len(X_train_dataset) + len(X_validate_dataset) + len(X_test_dataset)}")

    # Print the shape of each dataset to verify the expected feature and label dimensions.
    print("X_train:", X_train_dataset.shape, "y_train:", y_train_dataset.shape)
    print("X_val:", X_validate_dataset.shape, "y_val:", y_validate_dataset.shape)
    print("X_test:", X_test_dataset.shape, "y_test:", y_test_dataset.shape)
    print("-"*30)

    # Print the number of fake and real articles in each dataset split to verify
    # that both classes are represented.
    
    print(f"Train - total:", len(y_train_dataset))
    print(f"Train - fake (0):", (y_train_dataset==0).sum())
    print(f"Train - real (1):", (y_train_dataset==1).sum())
    print("-"*30)

    print(f"Validate - total:", len(y_validate_dataset))
    print(f"Validate - fake (0):", (y_validate_dataset==0).sum())
    print(f"Validate - real (1):", (y_validate_dataset==1).sum())
    print("-"*30)

    print(f"Test - total:", len(y_test_dataset))
    print(f"Test - fake (0):", (y_test_dataset==0).sum())
    print(f"Test - real (1):", (y_test_dataset==1).sum())
    print("-"*30)

    # Verify that every article embedding has the expected 768-dimensional
    # shape produced by the embedding model before model training.

    a = all(x.shape == (768,) for x in X_train_dataset)
    b = all(x.shape == (768,) for x in X_validate_dataset)
    c = all(x.shape == (768,) for x in X_test_dataset)
    print(f"Consistent shape in train: {a}")
    print(f"Consistent shape in validate: {b}")
    print(f"Consistent shape in test: {c}")
    print("-"*30)

    # Check for invalid numerical values in the embeddings before model training.

    X_train = np.stack(X_train_dataset)
    d = np.isnan(X_train).any()
    e = np.isinf(X_train).any()
    print(f"NaN in train: {d}")
    print(f"Inf in train: {e}")

    X_validate = np.stack(X_validate_dataset)
    f = np.isnan(X_validate).any()
    g = np.isinf(X_validate).any()
    print(f"NaN in validate: {f}")
    print(f"Inf in validate: {g}")

    X_test = np.stack(X_test_dataset)
    h = np.isnan(X_test).any()
    j = np.isinf(X_test).any()
    print(f"NaN in test: {h}")
    print(f"Inf in test: {j}")
    print("-"*30)

if __name__ == "__main__":
    main()