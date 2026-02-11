import numpy as np
import os

def main():
    DATA_LOCATION = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "clean")

    train_file = os.path.join(DATA_LOCATION, "training_dataset.npy")
    validate_file = os.path.join(DATA_LOCATION, "validating_dataset.npy")
    test_file = os.path.join(DATA_LOCATION, "testing_dataset.npy")

    train_dataset = np.load(train_file, allow_pickle=True)
    validate_dataset = np.load(validate_file, allow_pickle=True)
    test_dataset = np.load(test_file, allow_pickle=True)

    print(f"{type(train_dataset)}")
    print(f"{type(validate_dataset)}")
    print(f"{type(test_dataset)}")

    print(f"{train_dataset.shape}")
    print(f"{validate_dataset.shape}")
    print(f"{test_dataset.shape}")

    print(f"{train_dataset[0]}")
    print(f"{validate_dataset[0]}")
    print(f"{test_dataset[0]}")

if __name__ == "__main__":
    main()