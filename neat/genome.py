import networkx as nx

from neat.gene import Gene


class Genome:
    """Класс объявления генотипа."""

    def __init__(self) -> None:
        self.genes: list[Gene] = []  # гены генотипа
        self.nodes: set[int] = set()  # уникальные узлы
        self.fitness: int = 0  # совместимость

    def add_gene(self, gene: Gene) -> None:
        self.genes.append(gene)
        self.nodes.add(gene.in_node)
        self.nodes.add(gene.out_node)

    def get_topology(self) -> None:
        # получение топологии генотипа
        G = nx.DiGraph()
        edge_labels = {}
        for gene in self.genes:
            if gene.enabled:
                G.add_edge(gene.in_node, gene.out_node, weight=gene.weight)
                edge_labels[(gene.in_node, gene.out_node)] = f"{gene.weight:.2f}"
        return G, edge_labels
