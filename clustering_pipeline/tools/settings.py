from pathlib import Path
from pydantic.dataclasses import dataclass
from typing import Tuple, List

@dataclass
class Settings:
    data_path: Path = Path("data/sample.csv")
    output_dir: Path = Path("outputs/")
    k_range: Tuple[int, int] = (2, 10)
    random_state: int = 42
    scoring_metrics: List[str] = ("silhouette", "davies_bouldin", "calinski_harabasz")

settings = Settings()
