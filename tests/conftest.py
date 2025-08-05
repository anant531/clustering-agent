# tests/fixtures/conftest.py
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import os

# Import your actual data loader components
from clustering_pipeline.core.data_loader import (
    DataValidationConfig,
    MissingValueStrategy,
    DuplicateStrategy
)


@pytest.fixture
def sample_dataframe():
    """Create a sample DataFrame for testing."""
    return pd.DataFrame({
        'feature_1': [1.0, 2.0, 3.0, 4.0, 5.0],
        'feature_2': [2.0, 3.0, 4.0, 5.0, 6.0],
        'feature_3': [3.0, 4.0, 5.0, 6.0, 7.0],
        'target': ['A', 'B', 'A', 'B', 'C']
    })


@pytest.fixture
def temp_data_dir(tmp_path):
    """Create a temporary directory for test data files."""
    data_dir = tmp_path / "test_data"
    data_dir.mkdir()
    return data_dir


@pytest.fixture
def sample_csv_content():
    """Sample CSV content for testing."""
    return """feature_1,feature_2,feature_3,target
1.0,2.0,3.0,A
2.0,3.0,4.0,B
3.0,4.0,5.0,A
4.0,5.0,6.0,B
5.0,6.0,7.0,C"""


@pytest.fixture
def basic_config():
    """Basic validation configuration for testing."""
    return DataValidationConfig(
        missing_strategy=MissingValueStrategy.DROP,
        duplicate_strategy=DuplicateStrategy.DROP,
        min_rows=5,
        max_missing_ratio=0.1
    )


@pytest.fixture
def strict_config():
    """Strict validation configuration for testing."""
    return DataValidationConfig(
        missing_strategy=MissingValueStrategy.RAISE_ERROR,
        duplicate_strategy=DuplicateStrategy.RAISE_ERROR,
        min_rows=10,
        max_missing_ratio=0.0,
        strict_types=True
    )


@pytest.fixture
def lenient_config():
    """Lenient validation configuration for testing."""
    return DataValidationConfig(
        missing_strategy=MissingValueStrategy.IMPUTE_MEAN,
        duplicate_strategy=DuplicateStrategy.KEEP_FIRST,
        min_rows=1,
        max_missing_ratio=0.5,
        strict_types=False,
        allow_infinite=True
    )


@pytest.fixture
def complex_csv_content():
    """More complex CSV content with various data types."""
    return """id,feature_1,feature_2,category,value,date,flag
1,1.5,2.3,Type_A,100.0,2023-01-01,true
2,2.1,3.7,Type_B,150.5,2023-01-02,false
3,1.8,2.9,Type_A,200.0,2023-01-03,true
4,3.2,4.1,Type_C,175.3,2023-01-04,false
5,2.5,3.5,Type_B,225.8,2023-01-05,true"""


@pytest.fixture
def missing_data_csv_content():
    """CSV content with missing values for testing."""
    return """feature_1,feature_2,feature_3,target
1.0,2.0,3.0,A
2.0,,4.0,B
,4.0,5.0,A
4.0,5.0,,B
5.0,6.0,7.0,"""


@pytest.fixture
def duplicate_data_csv_content():
    """CSV content with duplicate rows for testing."""
    return """feature_1,feature_2,feature_3,target
1.0,2.0,3.0,A
1.0,2.0,3.0,A
2.0,3.0,4.0,B
2.0,3.0,4.0,B
3.0,4.0,5.0,C"""


@pytest.fixture(scope="session")
def large_dataset():
    """Create a large dataset for performance testing."""
    np.random.seed(42)  # For reproducible tests
    n_rows = 1000  # Reduced size for faster testing
    return pd.DataFrame({
        'feature_1': np.random.randn(n_rows),
        'feature_2': np.random.randn(n_rows),
        'feature_3': np.random.randn(n_rows),
        'feature_4': np.random.randn(n_rows),
        'category': np.random.choice(['A', 'B', 'C', 'D'], n_rows),
        'target': np.random.choice([0, 1], n_rows)
    })


def create_temp_file(tmp_path, filename, content, encoding='utf-8'):
    """Helper function to create temporary files with content."""
    file_path = tmp_path / filename
    if isinstance(content, str):
        file_path.write_text(content, encoding=encoding)
    else:
        # Assume it's a DataFrame
        if filename.endswith('.csv'):
            content.to_csv(file_path, index=False)
        elif filename.endswith('.json'):
            content.to_json(file_path, orient='records')
        elif filename.endswith('.xlsx'):
            content.to_excel(file_path, index=False)
    return file_path


# Configure pytest markers
def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )
    config.addinivalue_line(
        "markers", "performance: marks tests as performance tests"
    )


# Add collection hook to show test progress
def pytest_collection_modifyitems(config, items):
    """Add markers to tests based on their names or paths."""
    for item in items:
        # Mark slow tests
        if "large" in item.name.lower() or "performance" in item.name.lower():
            item.add_marker(pytest.mark.slow)
        
        # Mark integration tests
        if "integration" in str(item.fspath) or "end_to_end" in item.name.lower():
            item.add_marker(pytest.mark.integration)
        
        # Mark unit tests (default for tests in unit directory)
        if "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)


# Custom assertion helpers
def assert_dataframe_equal(df1, df2, check_dtype=True, check_names=True):
    """Custom assertion for DataFrame equality with better error messages."""
    try:
        pd.testing.assert_frame_equal(df1, df2, check_dtype=check_dtype, check_names=check_names)
    except AssertionError as e:
        # Add more context to the error message
        msg = f"""
DataFrames are not equal:
Shape df1: {df1.shape}, Shape df2: {df2.shape}
Columns df1: {list(df1.columns)}
Columns df2: {list(df2.columns)}
Original error: {str(e)}
"""
        raise AssertionError(msg) from e


def assert_no_missing_values(df, columns=None):
    """Assert that DataFrame has no missing values in specified columns."""
    if columns is None:
        columns = df.columns
    
    missing_info = df[columns].isnull().sum()
    missing_cols = missing_info[missing_info > 0]
    
    if not missing_cols.empty:
        raise AssertionError(f"Found missing values in columns: {missing_cols.to_dict()}")


def assert_no_duplicates(df):
    """Assert that DataFrame has no duplicate rows."""
    n_duplicates = df.duplicated().sum()
    if n_duplicates > 0:
        raise AssertionError(f"Found {n_duplicates} duplicate rows")