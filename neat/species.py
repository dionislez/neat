from neat.genome import Genome


class Species:
    """Класс создания новых видов."""

    def __init__(self, representative: Genome) -> None:
        self.representative = representative  # представитель вида
        self.genomes = []  # генотипы вида

    def add_genome(self, genome: Genome) -> None:
        self.genomes.append(genome)
