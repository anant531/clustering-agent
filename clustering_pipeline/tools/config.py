# clustering_pipeline/tools/config.py
from pathlib import Path
from typing import List, Literal, Optional, Tuple
import os
import yaml
from pydantic import BaseModel, Field, validator

from clustering_pipeline.tools.logger import get_logger

logger = get_logger(__name__)


class DataConfig(BaseModel):
    min_samples: int = 50
    handle_missing: Literal["drop", "impute"] = "drop"
    scaling_method: Literal["standard", "minmax", "none"] = "standard"


class ClusteringConfig(BaseModel):
    algorithms: List[Literal["kmeans", "agglo", "hdbscan"]] = ["kmeans"]
    k_range: Tuple[int, int] = (2, 10)
    n_jobs: int = -1
    random_state: int = 42


class EvaluationConfig(BaseModel):
    metrics: List[Literal["silhouette", "davies_bouldin", "calinski_harabasz"]] = ["silhouette"]


class OutputConfig(BaseModel):
    save_models: bool = True
    export_format: List[Literal["csv", "json"]] = ["csv"]


class PipelineConfig(BaseModel):
    data: DataConfig = DataConfig()
    clustering: ClusteringConfig = ClusteringConfig()
    evaluation: EvaluationConfig = EvaluationConfig()
    output: OutputConfig = OutputConfig()


def load_config(yaml_path: Optional[Path] = None) -> PipelineConfig:
    """Load pipeline config from YAML file with pydantic validation.

    Args:
        yaml_path (Path, optional): Path to config.yaml

    Returns:
        PipelineConfig: Validated config object
    """
    try:
        if yaml_path and yaml_path.exists():
            with open(yaml_path, "r") as f:
                raw_cfg = yaml.safe_load(f)
            
            # Convert k_range list back to tuple if it exists
            if raw_cfg and 'clustering' in raw_cfg and 'k_range' in raw_cfg['clustering']:
                if isinstance(raw_cfg['clustering']['k_range'], list):
                    raw_cfg['clustering']['k_range'] = tuple(raw_cfg['clustering']['k_range'])
            
            logger.info(f"Loaded config from: {yaml_path}")
        else:
            logger.warning("No config path provided. Using defaults.")
            raw_cfg = {}

        return PipelineConfig(**raw_cfg)

    except Exception as e:
        logger.error(f"Failed to load or validate config: {e}")
        raise

def save_config(config: PipelineConfig, yaml_path: Path) -> None:
    """Save configuration to YAML file."""
    try:
        yaml_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert config to dict and handle tuples
        config_dict = config.dict()
        
        # Convert tuples to lists for YAML compatibility
        if 'clustering' in config_dict and 'k_range' in config_dict['clustering']:
            config_dict['clustering']['k_range'] = list(config_dict['clustering']['k_range'])
        
        with open(yaml_path, "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)
        
        print(f"💾 Saved config to: {yaml_path}")
        
    except Exception as e:
        print(f"❌ Failed to save config: {e}")