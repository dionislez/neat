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
    ax.set_title(f"Generation {generation + 1} / {GENERATIONS}")
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
    save_path: Path,
    data_list: list,
    data2_list,
    title: str,
    ylabel: str,
    labels: list[str],
    filename: str,
) -> None:
    _, ax = plt.subplots(figsize=(20, 15))
    line1 = plt.plot(range(1, len(data_list) + 1), data_list, marker="o", label=labels[0])
    line2 = plt.plot(
        range(1, len(data2_list) + 1), data2_list, marker="o", label=labels[1]
    )

    lines = line1 + line2
    labels = [line.get_label() for line in lines]
    ax.legend(lines, labels, loc=0)

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
    mse_list_train, score_list_train = [], []
    mse_list_test, score_list_test = [], []
    for generation in range(generations):
        neat.run_generation(X_train, y_train)

        best_genome = max(neat.genomes, key=lambda g: g.fitness)
        predictions_train = np.array(
            [0 if neat.feed_forward(best_genome, x) < 0 else 1 for x in X_train]
        )
        predictions_test = np.array(
            [0 if neat.feed_forward(best_genome, x) < 0 else 1 for x in X_test]
        )

        mse_list_train.append(mean_squared_error(predictions_train, y_train))
        mse_list_test.append(mean_squared_error(predictions_test, y_test))
        score_list_train.append(accuracy_score(predictions_train, y_train))
        score_list_test.append(accuracy_score(predictions_test, y_test))

        G, edge_labels = best_genome.get_topology()
        save_topology(current_dir, generation=generation, G=G, edge_labels=edge_labels)

        title = (
            f"\n[{generation + 1} | {GENERATIONS}]\n"
            f"[MSE] {mse_list_test[-1]:.2f} (test), {mse_list_train[-1]:.2f} (train)\n"
            f"[Acccuracy] {score_list_test[-1]:.2f} (test), {score_list_train[-1]:.2f} (train)\n"
            f"[Nodes] {best_genome.nodes}, {best_genome.nodes & neat.node_innovations} (added)\n"
            f"[Best Fitness] {best_genome.fitness}"
        )

        save_convergence_plots(current_dir, mse_list_test, score_list_test)
        save_convergence_plot(
            current_dir,
            mse_list_test,
            mse_list_train,
            f"MSE Convergence{title}",
            "MSE",
            ["mse test", "mse train"],
            "mse_convergence.png",
        )
        save_convergence_plot(
            current_dir,
            score_list_test,
            score_list_train,
            f"Accuracy Convergence{title}",
            "Accuracy",
            ["score test", "score train"],
            "accuracy_convergence.png",
        )

        genome_path = (
            current_dir
            / f"{mse_list_test[-1]:.2f}_{score_list_test[-1]:.2f}_{best_genome.fitness:.2f}.pkl"
        )
        save_genome(best_genome, genome_path)

        print(title)


if __name__ == "__main__":
    main(
        DATA_PATH,
        SAVE_PATH,
        input_size=INPUT_SIZE,
        output_size=OUTPUT_SIZE,
        population_size=POPULATION_SIZE,
        generations=GENERATIONS,
    )
