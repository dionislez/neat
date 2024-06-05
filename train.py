from datetime import UTC, datetime
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from sklearn.metrics import accuracy_score, mean_squared_error

from neat.population import Population
from utils import get_train_test_split, save_genome

DATA_PATH = "Data/cancer/cancer1.dt"
SAVE_PATH = "RESULTS"
INPUT_SIZE = 2
OUTPUT_SIZE = 1
POPULATION_SIZE = 100
GENERATIONS = 30


def save_topology(save_path: Path, *, generation: int, G, edge_labels) -> None:
    pictures_dir = save_path / "topologies"
    if not pictures_dir.exists():
        pictures_dir.mkdir()

    fig, ax = plt.subplots(figsize=(12, 8))
    pos = nx.spring_layout(G)
    nx.draw(
        G,
        pos,
        with_labels=True,
        node_size=500,
        node_color="lightblue",
        font_size=5,
        font_weight="bold",
        arrowsize=5,
        ax=ax,
    )
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=ax)
    ax.set_title(f"Generation {generation + 1}")
    plt.savefig(pictures_dir / f"generation_{generation + 1}.png")
    plt.close(fig)


def save_convergence_plots(save_path: Path, mse_list: list, score_list: list) -> None:
    fig, ax1 = plt.subplots(figsize=(12, 8))

    ax2 = ax1.twinx()
    ax1.plot(range(1, len(mse_list) + 1), mse_list, "g-")
    ax2.plot(range(1, len(score_list) + 1), score_list, "b-")

    ax1.set_xlabel("Generation")
    ax1.set_ylabel("MSE", color="g")
    ax2.set_ylabel("Accuracy", color="b")

    plt.title("Convergence Plots")
    plt.savefig(save_path / "convergence_plots.png")
    plt.close(fig)


def save_convergence_plot(
    save_path: Path, data_list: list, title: str, ylabel: str, filename: str
) -> None:
    plt.figure(figsize=(12, 8))
    plt.plot(range(1, len(data_list) + 1), data_list, marker="o")
    plt.title(title)
    plt.xlabel("Generation")
    plt.ylabel(ylabel)
    plt.grid()
    plt.savefig(save_path / filename)
    plt.close()


def main(
    data_path: str,
    save_path: str,
    *,
    input_size: int,
    output_size: int,
    population_size: int,
    generations: int,
) -> None:
    if not (data_path := Path(data_path)).exists():
        return

    if not (save_path := Path(save_path)).exists():
        save_path.mkdir()

    model_path = (
        save_path / f"{input_size}_{output_size}_{population_size}_{generations}"
    )
    if not model_path.exists():
        model_path.mkdir()

    current_dir = model_path / f"{datetime.now(UTC)}"
    if not current_dir.exists():
        current_dir.mkdir()

    X_train, X_test, y_train, y_test = get_train_test_split(data_path)
    neat = Population(
        input_size=input_size,
        output_size=output_size,
        population_size=population_size,
    )
    mse, score = 0, 0
    mse_list, score_list = [], []
    for generation in range(generations):
        neat.run_generation(X_train, y_train)

        best_genome = max(neat.genomes, key=lambda g: g.fitness)
        predictions = np.array(
            [0 if neat.feed_forward(best_genome, x) < 0 else 1 for x in X_test]
        )

        mse = mean_squared_error(predictions, y_test)
        mse_list.append(mse)
        score = accuracy_score(predictions, y_test)
        score_list.append(score)

        G, edge_labels = best_genome.get_topology()
        save_topology(current_dir, generation=generation, G=G, edge_labels=edge_labels)

        print(
            f"[{generation + 1}]: MSE: {mse:.2f}, "
            f"Acccuracy: {score:.2f}, "
            f"Best Fitness: {best_genome.fitness:.2f}"
        )

    save_convergence_plots(current_dir, mse_list, score_list)
    save_convergence_plot(
        current_dir,
        mse_list,
        "MSE Convergence",
        "MSE",
        "mse_convergence.png",
    )
    save_convergence_plot(
        current_dir,
        score_list,
        "Accuracy Convergence",
        "Accuracy",
        "accuracy_convergence.png",
    )

    best_genome = max(neat.genomes, key=lambda g: g.fitness)
    genome_path = current_dir / f"{mse:.2f}_{score:.2f}_{best_genome.fitness:.2f}.pkl"
    save_genome(best_genome, genome_path)


if __name__ == "__main__":
    main(
        DATA_PATH,
        SAVE_PATH,
        input_size=INPUT_SIZE,
        output_size=OUTPUT_SIZE,
        population_size=POPULATION_SIZE,
        generations=GENERATIONS,
    )
