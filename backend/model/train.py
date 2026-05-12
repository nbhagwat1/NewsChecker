import joblib
import os
import numpy as np
from sklearn.linear_model import LogisticRegression

def load_datasets():
    DATA_LOCATION = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "clean")

    train_file = os.path.join(DATA_LOCATION, "training_dataset.npy")
    validate_file = os.path.join(DATA_LOCATION, "validating_dataset.npy")
    test_file = os.path.join(DATA_LOCATION, "testing_dataset.npy")

    train_dataset = np.load(train_file, allow_pickle=True)
    validate_dataset = np.load(validate_file, allow_pickle=True)
    test_dataset = np.load(test_file, allow_pickle=True)

    return train_dataset, validate_dataset, test_dataset

def unpack_dataset(dataset):
    X_list = []
    y_list = []

    for sample in dataset:
        average_embedding = sample[0]
        label = sample[1]

        X_list.append(average_embedding)
        y_list.append(label)
    
    X_dataset = np.stack(X_list)
    y_dataset = np.array(y_list)

    return X_dataset, y_dataset

def main():
    train_dataset, validate_dataset, test_dataset = load_datasets()

    X_train_dataset, y_train_dataset = unpack_dataset(train_dataset)
    X_train_dataset = X_train_dataset.astype(np.float32)

    logistic_model = LogisticRegression(max_iter=1000)
    logistic_model.fit(X_train_dataset, y_train_dataset)

    joblib.dump(logistic_model, "logistic_model.pkl")

def print_stats(X_train_dataset, X_validate_dataset, X_test_dataset, y_train_dataset, y_validate_dataset, y_test_dataset):
    print(f"TOTAL ARTICLES: {len(X_train_dataset) + len(X_validate_dataset) + len(X_test_dataset)}")

    print("X_train:", X_train_dataset.shape, "y_train:", y_train_dataset.shape)
    print("X_val:", X_validate_dataset.shape, "y_val:", y_validate_dataset.shape)
    print("X_test:", X_test_dataset.shape, "y_test:", y_test_dataset.shape)

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

    a = all(x.shape == (768,) for x in X_train_dataset)
    b = all(x.shape == (768,) for x in X_validate_dataset)
    c = all(x.shape == (768,) for x in X_test_dataset)
    print(f"Consistent shape in train: {a}")
    print(f"Consistent shape in validate: {b}")
    print(f"Consistent shape in test: {c}")

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

if __name__ == "__main__":
    main()