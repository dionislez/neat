class Gene:
    """Класс генов."""

    def __init__(
        self,
        in_node: int,
        out_node: int,
        weight: float | int,
        enabled: bool,
        innovation: int,
    ) -> None:
        self.in_node: int = in_node  # входной нейрон
        self.out_node: int = out_node  # выходной нейрон
        self.weight: float | int = weight  # вес соединения
        self.enabled: bool = enabled  # есть ли соединение между нейронами
        self.innovation: int = innovation  # инновационный номер для поиска соответствующих генов при скрещивании

    def __repr__(self) -> str:
        return (
            f"Gene(in:{self.in_node}, "
            f"out:{self.out_node}, "
            f"weight:{self.weight:.2f}, "
            f"enabled:{self.enabled}, "
            f"innovation:{self.innovation})"
        )
