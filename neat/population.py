import random

import numpy as np

from neat.gene import Gene
from neat.genome import Genome
from neat.species import Species


class Population:
    """Класс популяции."""

    def __init__(self, input_size, output_size, population_size) -> None:
        self.input_size: int = input_size  # количество входных узлов
        self.output_size: int = output_size  # количество выходных узлов
        self.population_size = (
            population_size  # размер популяции (количество генотипов)
        )
        self.genomes = []  # генотипы
        self.species = []  # виды
        self.innovation_number = 0  # иновационный номер
        self.compatibility_threshold = 3.0  # Пороговое значение совместимости
        self.node_innovations = set()  # новые узлы после мутации
        self.initialize_population()  # инициализация генотипов и геномов по размеру

    def initialize_population(self) -> None:
        for _ in range(self.population_size):
            genome = Genome()  # генотип
            for in_node in range(self.input_size):
                for out_node in range(self.output_size):
                    initial_gene = Gene(
                        in_node=in_node,  # номер входного узла
                        out_node=self.input_size + out_node,  # номер выходного узла
                        weight=random.uniform(-1, 1),
                        enabled=True,
                        innovation=self.innovation_number,
                    )
                    self.innovation_number += 1  # инкремент инновационного номера
                    genome.add_gene(initial_gene)  # добавление гена к генотипу
            self.genomes.append(genome)

    def evaluate_fitness(self, X_train: np.array, y_train: np.array) -> None:
        # функция для расчета приспособленности генотипов
        for genome in self.genomes:
            genome.fitness = self.calculate_fitness(genome, X_train, y_train)

    def calculate_fitness(
        self, genome: Genome, X_train: np.array, y_train: np.array
    ) -> float | int:
        # функция расчета приспособленности
        predictions = np.array([self.feed_forward(genome, x) for x in X_train])
        mse = np.mean((predictions - y_train) ** 2)  # среднеквадратичная ошибка
        mae = np.mean(np.abs(predictions - y_train))  # средняя абсолютная ошибка
        adjust_fitness = 1 / (
            mae + 1e-6
        )  # обратное значение mse / mae для оценки приспособленности
        # 1e-6 - предотвращение деления на 0
        # для увеличения значения приспособленности
        summed_sh = sum(self.sharing_function(genome, other) for other in self.genomes)
        summed_sh = 1 if summed_sh == 0 else summed_sh  # предотвращение деления на 0
        fitness = adjust_fitness / summed_sh
        # препятствие росту более успешных видов для увеличения разнообразия
        # когда успешный вид заходит в тупик,
        # то его место может занять ранее менее успешный
        return fitness

    def sharing_function(self, genome1: Genome, genome2: Genome) -> float:
        # Функция схожести между двумя генотипами
        delta = self.compatibility_distance(genome1, genome2)
        if delta < self.compatibility_threshold:
            return 1
        return 0

    def feed_forward(self, genome: Genome, inputs: np.array) -> float | int:
        hidden_values = {}
        for gene in genome.genes:
            if not gene.enabled:
                continue

            in_node, out_node, weight = gene.in_node, gene.out_node, gene.weight
            if in_node < len(inputs):
                # если узел является входным (in_node)
                hidden_values[out_node] = (
                    hidden_values.get(out_node, 0) + inputs[in_node] * weight
                )
            else:
                # скрытый узел (in_node)
                hidden_values[out_node] = (
                    hidden_values.get(out_node, 0)
                    + hidden_values.get(in_node, 0) * weight
                )
        output = sum(
            hidden_values.get(node, 0)
            for node in range(self.input_size, self.input_size + self.output_size)
        )  # пороговая функция активации
        return output

    def speciate(self) -> None:
        # обнуление видов после каждого поколения для образования новых (эпохи)
        for species in self.species:
            species.genomes = []
        for genome in self.genomes:
            for species in self.species:
                # если виды одинаковые, то добавляем генотип
                if (
                    self.compatibility_distance(genome, species.representative)
                    < self.compatibility_threshold
                ):
                    species.add_genome(genome)
                    break
            else:
                # образование нового вида, если генотипы не одинаковы
                new_species = Species(genome)
                self.species.append(new_species)
                new_species.add_genome(genome)

    def compatibility_distance(
        self,
        genome1: Genome,
        genome2: Genome,
        c1: int | float = 1.0,
        c2: int | float = 1.0,
        c3: int | float = 0.4,
    ) -> bool:
        # Коэффициенты https://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf
        # страница 111

        excess_genes = 0  # количество избыточных узлов
        disjoint_genes = 0  # количество непересекающихся узлов
        matching_genes = 0  # количество одинаковых узлов
        weight_diff_sum = 0  # весовая разница узлов (по модулю)

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
            )  # поиск совпадения инновационного номера в 1 генотипе, иначе None
            gene2 = next(
                (gene for gene in genome2.genes if gene.innovation == innovation), None
            )  # поиск совпадения инновационного номера во 2 генотипе, иначе None

            if gene1 and gene2:
                # если одинаковые инновационные числа у двух генотипов
                # то прибавляется 1 к matching_genes, считаем весовую разность по модулю
                matching_genes += 1
                weight_diff_sum += abs(gene1.weight - gene2.weight)
            elif gene1 or gene2:
                # если у кого-то из генотипов нашелся инновационный номер, то
                if innovation > max(max(innovations1), max(innovations2)):
                    # проверка на избыточный узел по поиску максимального инновационного номера
                    # из 2 генотипов
                    excess_genes += 1
                    continue
                # иначе узлы не пересекаются
                disjoint_genes += 1

        N = max(
            len(genome1.genes), len(genome2.genes)
        )  # максимальное количество генов из 2 генотипов
        if N < 20:
            N = 1  # если генотип маленький

        # рассчитываем расстояние совместимости генотипов (проверка деления на 0)
        compatibility = (
            (c1 * excess_genes / N)
            + (c2 * disjoint_genes / N)
            + (c3 * weight_diff_sum / matching_genes if matching_genes > 0 else 0)
        )
        return compatibility

    def reproduce(self):
        # функция образования нового генотипа (воспроизведение)
        new_genomes = []
        for species in self.species:
            if not species.genomes:
                continue

            # добавление наиболее приспособленных генотипов
            species.genomes.sort(key=lambda g: g.fitness, reverse=True)
            new_genomes.append(species.genomes[0])

            for _ in range(len(species.genomes)):
                # выбор любых 2 родителей
                parent1 = random.choice(species.genomes)
                parent2 = random.choice(species.genomes)
                # скрещивание
                child = self.crossover(parent1, parent2)
                # мутация
                self.mutate(child)
                # образование нового генотипа
                new_genomes.append(child)
        # новые генотипы
        self.genomes = new_genomes

    def crossover(self, parent1, parent2):
        # скрещивание (выбор одного из 2 родителей для унаследования генов)
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

    def mutate(self, genome: Genome) -> None:
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

    def add_node_mutation(self, genome: Genome) -> None:
        if not genome.genes:
            return

        # случайный ген для разрыва связи
        gene_to_split = random.choice(genome.genes)
        gene_to_split.enabled = False

        # новый узел
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

    def add_link_mutation(self, genome: Genome) -> None:
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

    def run_generation(self, X_train: np.array, y_train: np.array) -> None:
        self.evaluate_fitness(X_train, y_train)
        self.speciate()
        self.reproduce()
