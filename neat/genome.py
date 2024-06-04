import networkx as nx

from neat.gene import Gene


class Genome:
    """Класс объявления генотипа."""

    def __init__(self):
        self.genes: list[Gene] = []
        self.nodes: set[int] = set()
        self.fitness: int = 0

    def add_gene(self, gene: Gene):
        self.genes.append(gene)
        self.nodes.add(gene.in_node)
        self.nodes.add(gene.out_node)

    def get_topology(self):
        G = nx.DiGraph()
        edge_labels = {}
        for gene in self.genes:
            if gene.enabled:
                G.add_edge(gene.in_node, gene.out_node, weight=gene.weight)
                edge_labels[(gene.in_node, gene.out_node)] = f"{gene.weight:.2f}"
        return G, edge_labels
