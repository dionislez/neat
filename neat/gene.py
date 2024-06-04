class Gene:
    """Класс генов."""

    def __init__(self, in_node, out_node, weight, enabled, innovation):
        self.in_node: int = in_node  # входной узел
        self.out_node: int = out_node  # выходной узел
        self.weight: int | float = weight  # вес соединения
        self.enabled: bool = enabled  # есть ли соединение между узлами
        self.innovation: int = innovation  # инновационный номер для поиска соответствующих генов при скрещивании

    def __repr__(self):
        return (
            f"Gene(in:{self.in_node}, "
            f"out:{self.out_node}, "
            f"weight:{self.weight:.2f}, "
            f"enabled:{self.enabled}, "
            f"innovation:{self.innovation})"
        )
