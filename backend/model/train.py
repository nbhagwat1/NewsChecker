import os
import numpy as np

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
    X_validate_dataset, y_validate_dataset = unpack_dataset(validate_dataset)
    X_test_dataset, y_test_dataset = unpack_dataset(test_dataset)

    X_train_dataset = X_train_dataset.astype(np.float32)
    X_validate_dataset = X_validate_dataset.astype(np.float32)
    X_test_dataset = X_test_dataset.astype(np.float32)

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

    for i in range(5):
        print(y_train_dataset[i], X_train_dataset[i][:5])
    
    print(np.isnan(X_train_dataset).any())
    print(np.isnan(X_validate_dataset).any())
    print(np.isnan(X_test_dataset).any())

if __name__ == "__main__":
    main()