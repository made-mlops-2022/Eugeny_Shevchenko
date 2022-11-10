from dataclasses import dataclass, field


@dataclass()
class TrainingParams:
    model_type: str = field(default="KNeighborsClassifier")
    random_state: int = field(default=255)
    n_neighbors: int = 10
    weights: str = "distance"
    metric: str = "minkiwsky"
    scaler: str = "StandartScaler"
    n_estimators: int = 100
    criterion: str = "entropy"
