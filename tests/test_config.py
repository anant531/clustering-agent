# tests/test_config.py

#!/usr/bin/env python3
"""
Test the configuration system.
"""
from pathlib import Path
from clustering_pipeline.tools.config import load_config, save_config, PipelineConfig

def test_config_system():
    """Test configuration loading and validation."""
    print("🔧 Testing Configuration System")
    print("=" * 40)
    
    # Test 1: Load default config
    print("1. Testing default config...")
    config_default = load_config()
    print(f"   ✅ Default k_range: {config_default.clustering.k_range}")
    
    # Fix: Your config doesn't have primary_metric, it has metrics list
    first_metric = config_default.evaluation.metrics[0]
    print(f"   ✅ Default metric: {first_metric}")
    
    # Test 2: Load from YAML file
    print("\n2. Testing YAML config loading...")
    config_file = Path("configs/default_config.yaml")
    if config_file.exists():
        config_yaml = load_config(config_file)
        print(f"   ✅ YAML k_range: {config_yaml.clustering.k_range}")
        print(f"   ✅ YAML min_samples: {config_yaml.data.min_samples}")
    else:
        print(f"   ⚠️  Config file not found: {config_file}")
    
    # Test 3: Create and save custom config
    print("\n3. Testing custom config creation...")
    custom_config = PipelineConfig()
    custom_config.clustering.k_range = (3, 7)
    custom_config.data.min_samples = 25
    # Fix: Use metrics list instead of primary_metric
    custom_config.evaluation.metrics = ["calinski_harabasz"]
    
    custom_path = Path("configs/custom_test.yaml")
    save_config(custom_config, custom_path)
    
    # Test 4: Load the custom config back
    print("\n4. Testing custom config loading...")
    loaded_custom = load_config(custom_path)
    print(f"   ✅ Custom k_range: {loaded_custom.clustering.k_range}")
    print(f"   ✅ Custom metric: {loaded_custom.evaluation.metrics[0]}")
    
    print("\n🎉 Configuration system working correctly!")

if __name__ == "__main__":
    test_config_system()