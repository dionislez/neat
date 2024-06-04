class Species:
    """Класс создания новых видов."""

    def __init__(self, representative):
        self.representative = representative  # генотип вида
        self.genomes = []
        self.best_fitness = 0
        self.stagnation = 0

    def add_genome(self, genome):
        self.genomes.append(genome)
