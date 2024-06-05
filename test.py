from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, mean_squared_error

from neat.population import Population
from utils import get_train_test_split, load_genome

DATA_PATH = "Data/cancer/cancer1.dt"
GENOME_PKL = r"RESULTS/10_1_30_10/2024-06-05 12:33:46.821811+00:00/0.38_0.62_2.28.pkl"
INPUT_SIZE = 10
OUTPUT_SIZE = 1
POPULATION_SIZE = 30


def main(data_path: str) -> None:
    data_path = Path(data_path)
    _, X_test, _, y_test = get_train_test_split(data_path)

    neat = Population(
        input_size=INPUT_SIZE,
        output_size=OUTPUT_SIZE,
        population_size=POPULATION_SIZE,
    )
    best_genome = load_genome(Path(GENOME_PKL))
    predictions = np.array(
        [0 if neat.feed_forward(best_genome, x) < 0 else 1 for x in X_test]
    )
    mse = mean_squared_error(predictions, y_test)
    score = accuracy_score(predictions, y_test)
    print(
        f"MSE: {mse:.2f}, "
        f"Acccuracy: {score:.2f}, "
        f"Best Fitness: {best_genome.fitness:.2f}"
    )


if __name__ == "__main__":
    main(DATA_PATH)
