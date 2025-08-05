# clustering_pipeline/exceptions/errors.py - COMPLETE FILE
class PipelineError(Exception):
    """Base exception for pipeline errors."""
    pass

class DataLoadError(PipelineError):
    """Raised when data loading fails."""
    pass

class ColumnNotFoundError(DataLoadError):
    """Raised when required columns are missing."""
    pass

class MissingValuesError(DataLoadError):
    """Raised when missing values are found and not allowed."""
    pass

class InvalidColumnTypeError(DataLoadError):
    """Raised when column types are invalid."""
    pass

class InsufficientRowsError(DataLoadError):
    """Raised when dataset has too few rows."""
    pass

class DuplicateRowsError(DataLoadError):
    """Raised when duplicate rows are found and not allowed."""
    pass

class ClusteringError(PipelineError):
    """Raised when clustering computation fails."""
    pass

class EvaluationError(PipelineError):
    """Raised when metric evaluation fails."""
    pass