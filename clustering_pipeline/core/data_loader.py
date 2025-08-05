# clustering_pipeline/core/data_loader.py
from typing import List, Tuple, Dict, Any, Optional, Union
from pathlib import Path
import pandas as pd
import numpy as np
from pydantic import BaseModel, Field, validator
from dataclasses import dataclass
from enum import Enum

from clustering_pipeline.tools.logger import get_logger
from clustering_pipeline.exceptions.errors import (
    DataLoadError,
    ColumnNotFoundError,
    MissingValuesError,
    InvalidColumnTypeError,
    InsufficientRowsError,
    DuplicateRowsError
)

logger = get_logger(__name__)


class MissingValueStrategy(str, Enum):
    """Strategies for handling missing values."""
    DROP = "drop"
    IMPUTE_MEAN = "impute_mean"
    IMPUTE_MEDIAN = "impute_median"
    IMPUTE_MODE = "impute_mode"
    RAISE_ERROR = "raise_error"


class DuplicateStrategy(str, Enum):
    """Strategies for handling duplicate rows."""
    DROP = "drop"
    KEEP_FIRST = "keep_first"
    KEEP_LAST = "keep_last"
    RAISE_ERROR = "raise_error"


@dataclass
class DataQualityReport:
    """Comprehensive data quality report."""
    original_shape: Tuple[int, int]
    final_shape: Tuple[int, int]
    columns_validated: List[str]
    missing_values_count: Dict[str, int]
    infinite_values_count: Dict[str, int]
    duplicate_rows_count: int
    data_types: Dict[str, str]
    quality_score: float
    warnings: List[str]
    actions_taken: List[str]


class DataValidationConfig(BaseModel):
    """Configuration for data validation and processing."""
    min_rows: int = Field(default=9, ge=1, description="Minimum required rows")
    max_missing_ratio: float = Field(default=0.1, ge=0.0, le=1.0, description="Max allowed missing ratio per column")
    missing_strategy: MissingValueStrategy = Field(default=MissingValueStrategy.DROP)
    duplicate_strategy: DuplicateStrategy = Field(default=DuplicateStrategy.RAISE_ERROR)
    allow_infinite: bool = Field(default=False, description="Allow infinite values")
    strict_types: bool = Field(default=True, description="Enforce strict numeric types")
    detect_outliers: bool = Field(default=False, description="Detect and report outliers")
    outlier_threshold: float = Field(default=3.0, ge=1.0, description="Z-score threshold for outliers")
    
    @validator('missing_strategy')
    def validate_missing_strategy(cls, v):
        return MissingValueStrategy(v)
    
    @validator('duplicate_strategy')
    def validate_duplicate_strategy(cls, v):
        return DuplicateStrategy(v)


class CSVInputSchema(BaseModel):
    """Schema for CSV input validation."""
    file_path: Path
    required_columns: List[str]
    config: Optional[DataValidationConfig] = Field(default_factory=DataValidationConfig)
    
    @validator('file_path')
    def validate_file_exists(cls, v):
        if not v.exists():
            raise ValueError(f"File not found: {v}")
        if not v.suffix.lower() in ['.csv', '.txt']:
            raise ValueError(f"Unsupported file type: {v.suffix}")
        return v
    
    @validator('required_columns')
    def validate_columns_not_empty(cls, v):
        if not v:
            raise ValueError("At least one required column must be specified")
        return v


class DataValidator:
    """Enhanced data validator with configurable strategies."""
    
    def __init__(self, config: DataValidationConfig):
        self.config = config
        self.warnings: List[str] = []
        self.actions_taken: List[str] = []
    
    def validate_and_clean(self, df: pd.DataFrame, required_columns: List[str]) -> Tuple[pd.DataFrame, DataQualityReport]:
        """
        Comprehensive data validation and cleaning.
        
        Args:
            df: Input DataFrame
            required_columns: List of required column names
            
        Returns:
            Tuple of (cleaned DataFrame, quality report)
        """
        original_shape = df.shape
        df_clean = df.copy()
        
        # Step 1: Validate columns exist
        self._validate_columns(df_clean, required_columns)
        
        # Step 2: Validate and convert data types
        df_clean = self._validate_and_convert_types(df_clean, required_columns)
        
        # Step 3: Handle missing values
        df_clean, missing_stats = self._handle_missing_values(df_clean, required_columns)
        
        # Step 4: Handle infinite values
        df_clean, infinite_stats = self._handle_infinite_values(df_clean, required_columns)
        
        # Step 5: Handle duplicates
        df_clean, duplicate_count = self._handle_duplicates(df_clean)
        
        # Step 6: Check minimum rows requirement
        self._check_minimum_rows(df_clean)
        
        # Step 7: Optional outlier detection
        if self.config.detect_outliers:
            outlier_stats = self._detect_outliers(df_clean, required_columns)
        else:
            outlier_stats = {}
        
        # Generate quality report
        quality_report = self._generate_quality_report(
            original_shape=original_shape,
            final_shape=df_clean.shape,
            columns_validated=required_columns,
            missing_stats=missing_stats,
            infinite_stats=infinite_stats,
            duplicate_count=duplicate_count,
            df_clean=df_clean
        )
        
        logger.info(f"✅ Data validation complete. Quality score: {quality_report.quality_score:.2f}")
        return df_clean, quality_report
    
    def _validate_columns(self, df: pd.DataFrame, required: List[str]) -> None:
        """Validate that all required columns exist."""
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise ColumnNotFoundError(f"Missing required columns: {missing}")
        logger.info(f"✅ All required columns present: {required}")
    
    def _validate_and_convert_types(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """Validate and convert column types to numeric."""
        df_converted = df.copy()
        
        for col in columns:
            if not pd.api.types.is_numeric_dtype(df_converted[col]):
                if self.config.strict_types:
                    raise InvalidColumnTypeError(f"Column '{col}' must be numeric, got {df_converted[col].dtype}")
                else:
                    # Attempt conversion
                    try:
                        df_converted[col] = pd.to_numeric(df_converted[col], errors='coerce')
                        self.warnings.append(f"Converted column '{col}' to numeric")
                        self.actions_taken.append(f"Type conversion: {col}")
                    except Exception as e:
                        raise InvalidColumnTypeError(f"Cannot convert column '{col}' to numeric: {e}")
        
        logger.info("✅ All required columns are numeric")
        return df_converted
    
    def _handle_missing_values(self, df: pd.DataFrame, columns: List[str]) -> Tuple[pd.DataFrame, Dict[str, int]]:
        """Handle missing values according to strategy."""
        df_clean = df.copy()
        missing_stats = {}
        
        for col in columns:
            missing_count = df_clean[col].isnull().sum()
            missing_ratio = missing_count / len(df_clean)
            missing_stats[col] = missing_count
            
            if missing_count > 0:
                if missing_ratio > self.config.max_missing_ratio:
                    self.warnings.append(f"Column '{col}' has {missing_ratio:.2%} missing values")
                
                if self.config.missing_strategy == MissingValueStrategy.RAISE_ERROR:
                    raise MissingValuesError(f"Missing values found in column '{col}': {missing_count}")
                
                elif self.config.missing_strategy == MissingValueStrategy.DROP:
                    df_clean = df_clean.dropna(subset=[col])
                    self.actions_taken.append(f"Dropped {missing_count} rows with missing values in '{col}'")
                
                elif self.config.missing_strategy == MissingValueStrategy.IMPUTE_MEAN:
                    df_clean[col].fillna(df_clean[col].mean(), inplace=True)
                    self.actions_taken.append(f"Imputed {missing_count} missing values in '{col}' with mean")
                
                elif self.config.missing_strategy == MissingValueStrategy.IMPUTE_MEDIAN:
                    df_clean[col].fillna(df_clean[col].median(), inplace=True)
                    self.actions_taken.append(f"Imputed {missing_count} missing values in '{col}' with median")
                
                elif self.config.missing_strategy == MissingValueStrategy.IMPUTE_MODE:
                    mode_value = df_clean[col].mode().iloc[0] if not df_clean[col].mode().empty else 0
                    df_clean[col].fillna(mode_value, inplace=True)
                    self.actions_taken.append(f"Imputed {missing_count} missing values in '{col}' with mode")
        
        logger.info("✅ Missing values handled")
        return df_clean, missing_stats
    
    def _handle_infinite_values(self, df: pd.DataFrame, columns: List[str]) -> Tuple[pd.DataFrame, Dict[str, int]]:
        """Handle infinite values."""
        df_clean = df.copy()
        infinite_stats = {}
        
        for col in columns:
            inf_count = np.isinf(df_clean[col]).sum()
            infinite_stats[col] = inf_count
            
            if inf_count > 0:
                if not self.config.allow_infinite:
                    # Replace infinities with NaN, then handle as missing values
                    df_clean[col] = df_clean[col].replace([np.inf, -np.inf], np.nan)
                    
                    if self.config.missing_strategy == MissingValueStrategy.DROP:
                        df_clean = df_clean.dropna(subset=[col])
                    else:
                        # Use max/min finite values as replacement
                        finite_values = df_clean[col][np.isfinite(df_clean[col])]
                        if len(finite_values) > 0:
                            max_val = finite_values.max()
                            min_val = finite_values.min()
                            df_clean[col] = df_clean[col].fillna(max_val if df_clean[col].isnull().sum() < inf_count else min_val)
                    
                    self.actions_taken.append(f"Handled {inf_count} infinite values in '{col}'")
                else:
                    self.warnings.append(f"Column '{col}' contains {inf_count} infinite values (allowed)")
        
        logger.info("✅ Infinite values handled")
        return df_clean, infinite_stats
    
    def _handle_duplicates(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
        """Handle duplicate rows."""
        duplicate_count = df.duplicated().sum()
        df_clean = df.copy()
        
        if duplicate_count > 0:
            if self.config.duplicate_strategy == DuplicateStrategy.RAISE_ERROR:
                raise DuplicateRowsError(f"Found {duplicate_count} duplicate rows")
            
            elif self.config.duplicate_strategy == DuplicateStrategy.DROP:
                df_clean = df_clean.drop_duplicates(ignore_index=True)
                self.actions_taken.append(f"Removed {duplicate_count} duplicate rows")
            
            elif self.config.duplicate_strategy == DuplicateStrategy.KEEP_FIRST:
                df_clean = df_clean.drop_duplicates(keep='first', ignore_index=True)
                self.actions_taken.append(f"Kept first occurrence of {duplicate_count} duplicate rows")
            
            elif self.config.duplicate_strategy == DuplicateStrategy.KEEP_LAST:
                df_clean = df_clean.drop_duplicates(keep='last', ignore_index=True)
                self.actions_taken.append(f"Kept last occurrence of {duplicate_count} duplicate rows")
        
        logger.info(f"✅ Duplicates handled: {duplicate_count} found")
        return df_clean, duplicate_count
    
    def _check_minimum_rows(self, df: pd.DataFrame) -> None:
        """Check minimum rows requirement."""
        if len(df) < self.config.min_rows:
            raise InsufficientRowsError(
                f"Dataset has insufficient rows: {len(df)} < {self.config.min_rows}"
            )
        logger.info(f"✅ Row count sufficient: {len(df)} rows")
    
    def _detect_outliers(self, df: pd.DataFrame, columns: List[str]) -> Dict[str, int]:
        """Detect outliers using Z-score method."""
        outlier_stats = {}
        
        for col in columns:
            z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
            outliers = (z_scores > self.config.outlier_threshold).sum()
            outlier_stats[col] = outliers
            
            if outliers > 0:
                self.warnings.append(f"Column '{col}' has {outliers} potential outliers")
        
        return outlier_stats
    
    def _generate_quality_report(self, **kwargs) -> DataQualityReport:
        """Generate comprehensive data quality report."""
        # Calculate quality score (0-1)
        total_issues = (
            sum(kwargs['missing_stats'].values()) +
            sum(kwargs['infinite_stats'].values()) +
            kwargs['duplicate_count'] +
            len(self.warnings)
        )
        
        total_cells = kwargs['original_shape'][0] * len(kwargs['columns_validated'])
        quality_score = max(0.0, 1.0 - (total_issues / max(total_cells, 1)))
        
        return DataQualityReport(
            original_shape=kwargs['original_shape'],
            final_shape=kwargs['final_shape'],
            columns_validated=kwargs['columns_validated'],
            missing_values_count=kwargs['missing_stats'],
            infinite_values_count=kwargs['infinite_stats'],
            duplicate_rows_count=kwargs['duplicate_count'],
            data_types={col: str(kwargs['df_clean'][col].dtype) for col in kwargs['columns_validated']},
            quality_score=quality_score,
            warnings=self.warnings.copy(),
            actions_taken=self.actions_taken.copy()
        )


def load_data(
    file_path: Union[str, Path], 
    required_columns: List[str],
    config: Optional[DataValidationConfig] = None
) -> Tuple[pd.DataFrame, DataQualityReport]:
    """
    Enhanced data loading and validation with comprehensive quality reporting.

    Args:
        file_path: Path to input CSV file
        required_columns: List of required column names
        config: Optional validation configuration

    Returns:
        Tuple of (validated DataFrame, quality report)

    Raises:
        DataLoadError: If validation fails
        ColumnNotFoundError: If required columns are missing
        InsufficientRowsError: If insufficient rows after cleaning
        Various other specific errors based on configuration
    """
    try:
        # Convert string path to Path object
        if isinstance(file_path, str):
            file_path = Path(file_path)
        
        # Use default config if none provided
        if config is None:
            config = DataValidationConfig()
        
        # Validate inputs
        schema = CSVInputSchema(
            file_path=file_path, 
            required_columns=required_columns,
            config=config
        )
        
        # Load CSV file
        logger.info(f"📂 Loading data from {file_path}")
        df = pd.read_csv(schema.file_path)
        logger.info(f"✅ Loaded data with shape {df.shape}")
        
        # Validate and clean data
        validator = DataValidator(config)
        df_clean, quality_report = validator.validate_and_clean(df, required_columns)
        
        logger.info(f"🎯 Final dataset shape: {df_clean.shape}")
        logger.info(f"📊 Data quality score: {quality_report.quality_score:.2f}")
        
        return df_clean, quality_report

    except Exception as e:
        logger.error(f"❌ Data loading failed: {e}")
        if isinstance(e, (DataLoadError, ColumnNotFoundError, InsufficientRowsError, 
                         InvalidColumnTypeError, MissingValuesError, DuplicateRowsError)):
            raise
        else:
            raise DataLoadError(f"Unexpected error loading data: {e}") from e


# Backward compatibility function
def load_and_validate_csv(file_path: Union[str, Path], feature_columns: List[str]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Backward compatibility wrapper."""
    df, quality_report = load_data(file_path, feature_columns)
    
    # Convert quality report to simple metadata dict for compatibility
    metadata = {
        "num_rows": quality_report.final_shape[0],
        "num_columns": quality_report.final_shape[1],
        "columns": quality_report.columns_validated,
        "quality_score": quality_report.quality_score,
        "warnings": quality_report.warnings
    }
    
    return df, metadata