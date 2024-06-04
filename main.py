import time  # noqa: F401
from typing import Any

import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.animation import FuncAnimation

from neat.population import Population


def update(generation: int, population: Population, ax: Any):
    population.run_generation()
    best_genome = max(population.genomes, key=lambda g: g.fitness)
    ax.clear()
    G, edge_labels = best_genome.get_topology()
    pos = nx.spring_layout(G)  # positions for all nodes
    nx.draw(
        G,
        pos,
        with_labels=True,
        node_size=500,
        node_color="lightblue",
        font_size=10,
        font_weight="bold",
        arrowsize=20,
        ax=ax,
    )
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=ax)
    ax.set_title(f"Generation {generation+1}, Best Fitness: {best_genome.fitness:.2f}")
    time.sleep(0.2)


def main(*, population_size: int = 100, generations: int = 1000):
    population = Population(size=population_size)
    fig, ax = plt.subplots(figsize=(12, 8))
    _ = FuncAnimation(
        fig,
        update,
        frames=generations,
        fargs=(population, ax),
        repeat=False,
    )
    plt.show()


if __name__ == "__main__":
    main(population_size=100, generations=1000)
