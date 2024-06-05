import pickle
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def get_train_test_split(file_path: Path) -> tuple:
    df = pd.read_csv(file_path.as_posix(), sep=" ", header=None, skiprows=7)
    x = df.iloc[:, :-2]
    y = df.iloc[:, -2]

    X_train, X_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
    )
    return (
        X_train.to_numpy(),
        X_test.to_numpy(),
        y_train.to_numpy(),
        y_test.to_numpy(),
    )


def save_genome(genome, file_path: Path) -> None:
    with open(file_path, "wb") as file:
        pickle.dump(genome, file)


def load_genome(file_path: Path) -> None:
    with open(file_path, "rb") as file:
        return pickle.load(file)
