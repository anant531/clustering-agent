#tests/test_data_loader.py
import pytest
import pandas as pd
import numpy as np
from pathlib import Path

# Import only what actually exists in your data loader
from clustering_pipeline.core.data_loader import (
    load_data,
    DataValidationConfig,
    MissingValueStrategy,
    DuplicateStrategy
)

# Import your custom exceptions
from clustering_pipeline.exceptions.errors import (
    DataLoadError,
    ColumnNotFoundError,
    MissingValuesError,
    InvalidColumnTypeError,
    InsufficientRowsError,
    DuplicateRowsError
)


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
def sample_file(tmp_path, sample_csv_content):
    """Create a temporary CSV file with sample data."""
    file_path = tmp_path / "sample.csv"
    file_path.write_text(sample_csv_content)
    return file_path


@pytest.fixture
def required_columns():
    """Standard required columns for testing."""
    return ['feature_1', 'feature_2']


class TestDataLoaderBasic:
    """Basic test cases for the data loader functionality."""

    def test_load_valid_data(self, sample_file, required_columns):
        """Test loading valid data with required columns."""
        df, quality_report = load_data(sample_file, required_columns)
        
        # Verify DataFrame is returned
        assert isinstance(df, pd.DataFrame)
        assert not df.empty
        
        # Verify required columns are present
        for col in required_columns:
            assert col in df.columns
        
        # Verify quality report has expected attributes
        assert hasattr(quality_report, 'final_shape')
        assert hasattr(quality_report, 'quality_score')
        assert quality_report.final_shape == df.shape

    def test_load_data_with_default_config(self, sample_file, required_columns):
        """Test loading data with default configuration."""
        config = DataValidationConfig()
        df, quality_report = load_data(sample_file, required_columns, config=config)
        
        assert isinstance(df, pd.DataFrame)
        assert not df.empty
        
        # Verify required columns are present
        for col in required_columns:
            assert col in df.columns

    def test_missing_file_raises_error(self, tmp_path, required_columns):
        """Test that missing file raises appropriate error."""
        fake_file = tmp_path / "missing.csv"
        
        with pytest.raises((DataLoadError, ValueError, FileNotFoundError)):
            load_data(fake_file, required_columns)

    def test_missing_required_column(self, sample_file):
        """Test that missing required column raises error."""
        with pytest.raises(ColumnNotFoundError):
            load_data(sample_file, ["nonexistent_column"])

    def test_string_path_conversion(self, sample_file, required_columns):
        """Test that string paths are properly converted to Path objects."""
        df, quality_report = load_data(str(sample_file), required_columns)
        
        assert isinstance(df, pd.DataFrame)
        assert not df.empty


class TestMissingValueHandling:
    """Test cases for missing value handling strategies."""

    def test_missing_values_drop_strategy(self, tmp_path, required_columns):
        """Test loading data with missing values using drop strategy."""
        csv_content = """feature_1,feature_2,feature_3,target
1.0,2.0,3.0,A
2.0,,4.0,B
,4.0,5.0,A
4.0,5.0,6.0,B"""
        
        file_path = tmp_path / "missing_values.csv"
        file_path.write_text(csv_content)
        
        config = DataValidationConfig(missing_strategy=MissingValueStrategy.DROP)
        df, quality_report = load_data(file_path, required_columns, config=config)
        
        # Should have fewer rows due to dropping missing values
        assert len(df) < 4
        assert not df[required_columns].isnull().any().any()

    def test_missing_values_impute_mean(self, tmp_path, required_columns):
        """Test loading data with missing values using mean imputation."""
        csv_content = """feature_1,feature_2,feature_3
1.0,2.0,3.0
2.0,,4.0
,4.0,5.0
4.0,5.0,6.0"""
        
        file_path = tmp_path / "missing_values.csv"
        file_path.write_text(csv_content)
        
        config = DataValidationConfig(missing_strategy=MissingValueStrategy.IMPUTE_MEAN)
        df, quality_report = load_data(file_path, required_columns, config=config)
        
        # Should have same number of rows, no missing values in required columns
        assert len(df) == 4
        assert not df[required_columns].isnull().any().any()

    def test_missing_values_raise_error(self, tmp_path, required_columns):
        """Test that missing values raise error when configured to do so."""
        csv_content = """feature_1,feature_2,feature_3
1.0,2.0,3.0
2.0,,4.0
3.0,4.0,5.0"""
        
        file_path = tmp_path / "missing_values.csv"
        file_path.write_text(csv_content)
        
        config = DataValidationConfig(missing_strategy=MissingValueStrategy.RAISE_ERROR)
        
        with pytest.raises(MissingValuesError):
            load_data(file_path, required_columns, config=config)


class TestDuplicateHandling:
    """Test cases for duplicate row handling strategies."""

    def test_duplicates_drop_strategy(self, tmp_path, required_columns):
        """Test loading data with duplicate rows using drop strategy."""
        csv_content = """feature_1,feature_2,feature_3,target
1.0,2.0,3.0,A
1.0,2.0,3.0,A
2.0,3.0,4.0,B
2.0,3.0,4.0,B"""
        
        file_path = tmp_path / "duplicates.csv"
        file_path.write_text(csv_content)
        
        config = DataValidationConfig(duplicate_strategy=DuplicateStrategy.DROP)
        df, quality_report = load_data(file_path, required_columns, config=config)
        
        # Should have fewer rows due to removing duplicates
        assert len(df) == 2
        assert not df.duplicated().any()

    def test_duplicates_raise_error(self, tmp_path, required_columns):
        """Test that duplicate rows raise error when configured to do so."""
        csv_content = """feature_1,feature_2,feature_3
1.0,2.0,3.0
1.0,2.0,3.0
2.0,3.0,4.0"""
        
        file_path = tmp_path / "duplicates.csv"
        file_path.write_text(csv_content)
        
        config = DataValidationConfig(duplicate_strategy=DuplicateStrategy.RAISE_ERROR)
        
        with pytest.raises(DuplicateRowsError):
            load_data(file_path, required_columns, config=config)


class TestDataValidation:
    """Test cases for data validation functionality."""

    def test_insufficient_rows_error(self, tmp_path, required_columns):
        """Test that insufficient rows raises error."""
        csv_content = """feature_1,feature_2,feature_3
1.0,2.0,3.0
2.0,3.0,4.0"""
        
        file_path = tmp_path / "small_dataset.csv"
        file_path.write_text(csv_content)
        
        config = DataValidationConfig(min_rows=100)
        
        with pytest.raises(InsufficientRowsError):
            load_data(file_path, required_columns, config=config)

    def test_invalid_column_type_strict_mode(self, tmp_path):
        """Test that non-numeric columns raise error in strict mode."""
        csv_content = """feature_1,feature_2,feature_3
1.0,hello,3.0
2.0,world,4.0
3.0,test,5.0"""
        
        file_path = tmp_path / "invalid_types.csv"
        file_path.write_text(csv_content)
        
        config = DataValidationConfig(strict_types=True)
        
        with pytest.raises(InvalidColumnTypeError):
            load_data(file_path, ['feature_1', 'feature_2'], config=config)

    def test_invalid_column_type_conversion_mode(self, tmp_path):
        """Test that non-numeric columns are converted when not in strict mode."""
        csv_content = """feature_1,feature_2,feature_3
1.0,2,3.0
2.0,3,4.0
3.0,4,5.0"""
        
        file_path = tmp_path / "convertible_types.csv"
        file_path.write_text(csv_content)
        
        config = DataValidationConfig(strict_types=False)
        df, quality_report = load_data(file_path, ['feature_1', 'feature_2'], config=config)
        
        # Should convert successfully
        assert pd.api.types.is_numeric_dtype(df['feature_2'])


class TestDataValidationConfig:
    """Test cases for DataValidationConfig."""

    def test_default_config_creation(self):
        """Test creating config with default values."""
        config = DataValidationConfig()
        
        assert config.min_rows == 50
        assert config.max_missing_ratio == 0.1
        assert config.missing_strategy == MissingValueStrategy.DROP
        assert config.duplicate_strategy == DuplicateStrategy.RAISE_ERROR

    def test_custom_config_creation(self):
        """Test creating config with custom values."""
        config = DataValidationConfig(
            min_rows=100,
            max_missing_ratio=0.05,
            missing_strategy=MissingValueStrategy.IMPUTE_MEAN,
            duplicate_strategy=DuplicateStrategy.DROP
        )
        
        assert config.min_rows == 100
        assert config.max_missing_ratio == 0.05
        assert config.missing_strategy == MissingValueStrategy.IMPUTE_MEAN
        assert config.duplicate_strategy == DuplicateStrategy.DROP


class TestQualityReport:
    """Test cases for quality reporting."""

    def test_quality_report_structure(self, sample_file, required_columns):
        """Test that quality report contains expected information."""
        df, quality_report = load_data(sample_file, required_columns)
        
        # Check quality report attributes
        assert hasattr(quality_report, 'original_shape')
        assert hasattr(quality_report, 'final_shape')
        assert hasattr(quality_report, 'columns_validated')
        assert hasattr(quality_report, 'missing_values_count')
        assert hasattr(quality_report, 'duplicate_rows_count')
        assert hasattr(quality_report, 'quality_score')
        assert hasattr(quality_report, 'warnings')
        assert hasattr(quality_report, 'actions_taken')
        
        # Verify quality report values
        assert quality_report.final_shape == df.shape
        assert quality_report.columns_validated == required_columns
        assert 0.0 <= quality_report.quality_score <= 1.0

    def test_high_quality_data_score(self, sample_file, required_columns):
        """Test that clean data gets a high quality score."""
        df, quality_report = load_data(sample_file, required_columns)
        
        # Clean data should have a high quality score
        assert quality_report.quality_score > 0.9


class TestEdgeCases:
    """Test edge cases and special scenarios."""

    def test_empty_file_handling(self, tmp_path, required_columns):
        """Test handling of empty files."""
        empty_file = tmp_path / "empty.csv"
        empty_file.write_text("")
        
        with pytest.raises((DataLoadError, pd.errors.EmptyDataError)):
            load_data(empty_file, required_columns)

    def test_header_only_file(self, tmp_path, required_columns):
        """Test handling of files with only headers."""
        header_only = tmp_path / "header_only.csv"
        header_only.write_text("feature_1,feature_2,feature_3")
        
        config = DataValidationConfig(min_rows=1)
        with pytest.raises(InsufficientRowsError):
            load_data(header_only, required_columns, config=config)

    def test_special_characters_in_data(self, tmp_path, required_columns):
        """Test handling of special characters in data."""
        special_content = '''feature_1,feature_2,description
1.0,2.0,"Contains, comma"
2.0,3.0,"Contains ""quotes"""
3.0,4.0,Contains;semicolon'''
        
        file_path = tmp_path / "special_chars.csv"
        file_path.write_text(special_content)
        
        df, quality_report = load_data(file_path, required_columns)
        assert len(df) == 3
        assert 'Contains, comma' in df['description'].values

    def test_unicode_handling(self, tmp_path, required_columns):
        """Test handling of Unicode characters in data."""
        unicode_content = """feature_1,feature_2,name
1.0,2.0,José
2.0,3.0,François
3.0,4.0,李小明"""
        
        file_path = tmp_path / "unicode_test.csv"
        file_path.write_text(unicode_content, encoding='utf-8')
        
        df, quality_report = load_data(file_path, required_columns)
        assert len(df) == 3
        assert 'José' in df['name'].values


@pytest.mark.parametrize("missing_strategy", [
    MissingValueStrategy.DROP,
    MissingValueStrategy.IMPUTE_MEAN,
    MissingValueStrategy.IMPUTE_MEDIAN,
    MissingValueStrategy.IMPUTE_MODE
])
def test_all_missing_strategies(tmp_path, missing_strategy):
    """Test all missing value strategies work."""
    csv_content = """feature_1,feature_2,feature_3
1.0,2.0,3.0
2.0,,4.0
,4.0,5.0
4.0,5.0,6.0"""
    
    file_path = tmp_path / f"missing_{missing_strategy.value}.csv"
    file_path.write_text(csv_content)
    
    config = DataValidationConfig(missing_strategy=missing_strategy)
    df, quality_report = load_data(file_path, ['feature_1', 'feature_2'], config=config)
    
    # All strategies should produce valid results
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0


@pytest.mark.parametrize("duplicate_strategy", [
    DuplicateStrategy.DROP,
    DuplicateStrategy.KEEP_FIRST,
    DuplicateStrategy.KEEP_LAST
])
def test_all_duplicate_strategies(tmp_path, duplicate_strategy):
    """Test all duplicate handling strategies work."""
    csv_content = """feature_1,feature_2,feature_3
1.0,2.0,3.0
1.0,2.0,3.0
2.0,3.0,4.0
2.0,3.0,4.0"""
    
    file_path = tmp_path / f"duplicates_{duplicate_strategy.value}.csv"
    file_path.write_text(csv_content)
    
    config = DataValidationConfig(duplicate_strategy=duplicate_strategy)
    df, quality_report = load_data(file_path, ['feature_1', 'feature_2'], config=config)
    
    # All strategies should remove duplicates
    assert len(df) == 2
    assert not df.duplicated().any()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])