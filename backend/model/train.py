import joblib
import os
import numpy as np
from sklearn.linear_model import LogisticRegression

def load_datasets():
    """
    Loads and returns the training, validation, and testing datasets.

    This method loads the training, validation, and testing datasets from their
    respective .npy files and returns them.

    Args:
        None
    
    Returns:
        list[tuple[np.ndarray, int]]: The training dataset.
        list[tuple[np.ndarray, int]]: The validation dataset.
        list[tuple[np.ndarray, int]]: The testing dataset.
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
    Splits a dataset into two lists, with one list containing all of the
    singular embeddings and the other list containing all of the binary
    truthfulness labels.

    This method accepts a list of tuples, where each tuple contains an
    embedding and its respective binary truthfulness label. The method
    puts all of the embeddings in one list and all of the binary
    truthfulness labels in another list. In the end, the method returns
    both of these lists. (Why is this splitting necessary?)

    Args:
        dataset (list[tuple[np.ndarray, int]]): A list of tuples, where each tuple contains an
        embedding and its respective binary truthfulness label.
    
    Returns:
        np.ndstack: A list of all of the singular embeddings in the inputted dataset.
        np.ndarray: A list of all of the binary truthfulness labels in the inputted dataset.
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
    Trains a logistic regression model with the training dataset and
    saves the trained model.

    This method loads the training, validation, and testing datasets. Then,
    it takes the training dataset and splits it into two lists: a list containing
    the dataset's singular embeddings and a list containing the embeddings'
    respective binary truthfulness labels. After that, the method creates a new
    logistic regression model and trains it using the training dataset, with the
    model's input features being the dataset's singular embeddings and the model's
    target labels being the embeddings' respective binary truthfulness labels.
    Once the model has been trained, it is saved to a file so that it can be loaded
    and used later without needing to be retrained.

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
    Prints valuable statistics about the training dataset, the validation dataset, and the testing dataset.

    This method prints information about the training dataset, the validation dataset, and the testing dataset, printing
    information about the number of real news articles in each dataset, the number of fake news articles in each dataset,
    whether or not the singular embeddings in each dataset have the same shape, and whether or not the singular embeddings
    in each dataset contain valid numerical values. (Check this last part)

    Args:
        X_train_dataset (list[np.ndarray]): The list of all singular embeddings in the training dataset.
        X_validate_dataset (list[np.ndarray]): The list of all singular embeddings in the validation dataset.
        X_test_dataset (list[np.ndarray]): The list of all singular embeddings in the testing dataset.
        y_train_dataset (list[int]): The list of all binary truthfulness labels in the training dataset.
        y_validate_dataset (list[int]): The list of all binary truthfulness labels in the validation dataset.
        y_test_dataset (list[int]): The list of all binary truthfulness labels in the testing dataset.
    
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