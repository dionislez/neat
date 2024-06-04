import random

from neat.gene import Gene
from neat.genome import Genome
from neat.species import Species


class Population:
    """Класс популяции."""

    def __init__(self, size):
        self.size = size  # размер (количество) генотипов
        self.genomes = []  # генотипы
        self.species = []  # виды
        self.innovation_number = 0
        self.node_innovations = set()  # инновационные узлы
        self.initialize_population()  # инициализация геномов с генами

    def initialize_population(self):
        input_nodes = [0, 1]  # входные узлы
        output_nodes = [2]  # выходные узлы

        for _ in range(self.size):
            # инициализация генома
            genome = Genome()
            for in_node in input_nodes:
                for out_node in output_nodes:
                    # инициализация гена
                    initial_gene = Gene(
                        in_node=in_node,
                        out_node=out_node,
                        weight=random.uniform(-1, 1),
                        enabled=True,
                        innovation=self.innovation_number,
                    )
                    self.innovation_number += 1  # добавление инновационного номера
                    genome.add_gene(initial_gene)  # добавление гена генотипу
            self.genomes.append(genome)

    def evaluate_fitness(self):
        for genome in self.genomes:
            genome.fitness = self.calculate_fitness(genome)

    def calculate_fitness(self, genome):
        # ?
        return random.random()

    def speciate(self):
        for species in self.species:
            species.genomes = []
        for genome in self.genomes:
            for species in self.species:
                # если виды одинаковые, то добавляем генотип
                if self.is_same_species(genome, species.representative):
                    species.add_genome(genome)
                    break
            else:
                # иначе создаем новый вид
                new_species = Species(genome)
                self.species.append(new_species)
                new_species.add_genome(genome)

    def is_same_species(
        self, genome1, genome2, compatibility_threshold=3.0, c1=1.0, c2=1.0, c3=0.4
    ):
        # Коэффициенты https://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf
        # страница 111

        excess_genes = 0  # количество избыточных генов
        disjoint_genes = 0  # количество непересекающихся генов
        matching_genes = 0  # количество одинаковых генов
        weight_diff_sum = 0  # весовая разница генов (по модулю)

        innovations1 = {
            gene.innovation for gene in genome1.genes
        }  # множество инновационных номеров 1 генотипа
        innovations2 = {
            gene.innovation for gene in genome2.genes
        }  # множество инновационных номеров 2 генотипа

        all_innovations = innovations1.union(
            innovations2
        )  # общее множество инновационных номеров

        for innovation in all_innovations:
            gene1 = next(
                (gene for gene in genome1.genes if gene.innovation == innovation), None
            )
            gene2 = next(
                (gene for gene in genome2.genes if gene.innovation == innovation), None
            )

            if gene1 and gene2:
                # если 2 генотипа имеют одинаковые гены (одинаковое инновационное число)
                # то прибавляется 1 к matching_genes, считаем весовую разность по модулю
                matching_genes += 1
                weight_diff_sum += abs(gene1.weight - gene2.weight)
            elif gene1 or gene2:
                # иначе если какой-то ген есть, то прибавляется 1 к matching_genes
                # считаем весовую разность по модулю
                if innovation > max(max(innovations1), max(innovations2)):
                    excess_genes += 1
                    continue
                disjoint_genes += 1

        N = max(len(genome1.genes), len(genome2.genes))
        if N < 20:
            N = 1  # Если генотип маленький, то присваиваем единицу

        # рассчитываем расстояние совместимости генотипов (проверка деления на 0)
        compatibility = (
            (c1 * excess_genes / N)
            + (c2 * disjoint_genes / N)
            + (c3 * weight_diff_sum / matching_genes if matching_genes > 0 else 0)
        )
        return compatibility < compatibility_threshold

    def reproduce(self):
        new_genomes = []
        for species in self.species:
            if not species.genomes:
                continue
            species.genomes.sort(key=lambda g: g.fitness, reverse=True)
            new_genomes.append(species.genomes[0])  # Elitism: keep the best genome
            for _ in range(len(species.genomes) - 1):
                parent1 = random.choice(species.genomes)
                parent2 = random.choice(species.genomes)
                child = self.crossover(parent1, parent2)
                self.mutate(child)
                new_genomes.append(child)
        self.genomes = new_genomes

    def crossover(self, parent1, parent2):
        child = Genome()
        parent1_genes = {gene.innovation: gene for gene in parent1.genes}
        parent2_genes = {gene.innovation: gene for gene in parent2.genes}

        for innovation in parent1_genes:
            gene1 = parent1_genes[innovation]
            gene2 = parent2_genes.get(innovation)

            if gene2 and random.random() > 0.5 and gene2.enabled:
                child.add_gene(
                    Gene(
                        gene2.in_node,
                        gene2.out_node,
                        gene2.weight,
                        gene2.enabled,
                        gene2.innovation,
                    )
                )
                continue

            child.add_gene(
                Gene(
                    gene1.in_node,
                    gene1.out_node,
                    gene1.weight,
                    gene1.enabled,
                    gene1.innovation,
                )
            )

        return child

    def mutate(self, genome):
        # вероятность изменения существующего гена
        mutation_rate = 0.8
        # вероятность добавления нового узла путем разрыва существующей связи
        add_node_rate = 0.03
        # вероятность добавления новой связи между двумя существующими узлами
        add_link_rate = 0.05

        if random.random() < mutation_rate:
            for gene in genome.genes:
                if random.random() < mutation_rate:
                    gene.weight += random.uniform(-0.5, 0.5)
                    gene.weight = max(min(gene.weight, 1), -1)

        if random.random() < add_node_rate:
            self.add_node_mutation(genome)

        if random.random() < add_link_rate:
            self.add_link_mutation(genome)

    def add_node_mutation(self, genome):
        if not genome.genes:
            return

        gene_to_split = random.choice(genome.genes)
        gene_to_split.enabled = False

        new_node = max(genome.nodes) + 1
        self.node_innovations.add(new_node)
        genome.nodes.add(new_node)

        gene1 = Gene(
            gene_to_split.in_node,
            new_node,
            weight=1.0,
            enabled=True,
            innovation=self.innovation_number,
        )
        self.innovation_number += 1
        genome.add_gene(gene1)

        gene2 = Gene(
            new_node,
            gene_to_split.out_node,
            weight=gene_to_split.weight,
            enabled=True,
            innovation=self.innovation_number,
        )
        self.innovation_number += 1
        genome.add_gene(gene2)

    def add_link_mutation(self, genome):
        if len(genome.nodes) < 2:
            return

        node1 = random.choice(list(genome.nodes))
        node2 = random.choice(list(genome.nodes))
        if node1 == node2:
            return

        new_gene = Gene(
            in_node=node1,
            out_node=node2,
            weight=random.uniform(-1, 1),
            enabled=True,
            innovation=self.innovation_number,
        )
        self.innovation_number += 1
        genome.add_gene(new_gene)

    def run_generation(self):
        self.evaluate_fitness()
        self.speciate()
        self.reproduce()
