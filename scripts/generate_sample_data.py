# scripts/generate_sample_data.py

#!/usr/bin/env python3
"""
Generate various sample datasets for testing the clustering pipeline.
This creates realistic datasets with different characteristics.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.datasets import make_blobs, make_circles

def generate_customer_data(n_samples=400, output_file="customer_segments.csv"):
    """Generate realistic customer segmentation dataset."""
    np.random.seed(42)
    
    # Define customer segments with realistic patterns
    segments = {
        'Premium': {'income': (80000, 150000), 'spending': (8000, 20000), 'age': (35, 55)},
        'Young_Professional': {'income': (45000, 80000), 'spending': (3000, 8000), 'age': (25, 40)},
        'Family': {'income': (50000, 90000), 'spending': (4000, 12000), 'age': (30, 50)},
        'Senior': {'income': (30000, 60000), 'spending': (2000, 6000), 'age': (55, 75)}
    }
    
    data = []
    true_labels = []
    
    samples_per_segment = n_samples // len(segments)
    
    for i, (segment_name, ranges) in enumerate(segments.items()):
        for _ in range(samples_per_segment):
            # Generate correlated features
            income = np.random.uniform(*ranges['income'])
            age = np.random.uniform(*ranges['age'])
            
            # Spending correlated with income but with variation
            base_spending = income * np.random.uniform(0.08, 0.25)
            spending = np.clip(base_spending, *ranges['spending'])
            
            customer = {
                'annual_income': round(income, 2),
                'annual_spending': round(spending, 2),
                'age': int(age),
                'spending_ratio': round(spending / income, 3),
                'segment_true': segment_name
            }
            
            data.append(customer)
            true_labels.append(i)
    
    df = pd.DataFrame(data)
    df['cluster_true'] = true_labels
    
    # Add some realistic noise
    noise_cols = ['annual_income', 'annual_spending']
    for col in noise_cols:
        noise = np.random.normal(0, df[col].std() * 0.05, len(df))
        df[col] += noise
        df[col] = df[col].round(2)
    
    # Recalculate spending ratio after noise
    df['spending_ratio'] = (df['annual_spending'] / df['annual_income']).round(3)
    
    # Save
    output_path = Path("data") / output_file
    output_path.parent.mkdir(exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print(f"✅ Generated {output_file}")
    print(f"   Shape: {df.shape}")
    print(f"   Segments: {list(segments.keys())}")
    print(f"   Features: ['annual_income', 'annual_spending', 'age', 'spending_ratio']")
    
    return df

def generate_iris_like_data(n_samples=300, output_file="iris_like.csv"):
    """Generate Iris-like dataset with clear clusters."""
    np.random.seed(42)
    
    # Create 3 species with different characteristics
    species_params = {
        'Species_A': {'sepal_length': (4.5, 6.0), 'sepal_width': (2.5, 4.0), 
                      'petal_length': (1.0, 2.0), 'petal_width': (0.1, 0.5)},
        'Species_B': {'sepal_length': (5.5, 7.0), 'sepal_width': (2.0, 3.5),
                      'petal_length': (3.0, 5.0), 'petal_width': (1.0, 1.8)},
        'Species_C': {'sepal_length': (6.0, 8.0), 'sepal_width': (2.5, 4.0),
                      'petal_length': (4.5, 7.0), 'petal_width': (1.5, 2.5)}
    }
    
    data = []
    samples_per_species = n_samples // 3
    
    for i, (species, params) in enumerate(species_params.items()):
        for _ in range(samples_per_species):
            sample = {
                'sepal_length': round(np.random.uniform(*params['sepal_length']), 2),
                'sepal_width': round(np.random.uniform(*params['sepal_width']), 2),
                'petal_length': round(np.random.uniform(*params['petal_length']), 2),
                'petal_width': round(np.random.uniform(*params['petal_width']), 2),
                'species_true': species,
                'cluster_true': i
            }
            data.append(sample)
    
    df = pd.DataFrame(data)
    
    # Save
    output_path = Path("data") / output_file
    output_path.parent.mkdir(exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print(f"✅ Generated {output_file}")
    print(f"   Shape: {df.shape}")
    print(f"   Features: ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']")
    
    return df

def generate_problematic_data(output_file="problematic_data.csv"):
    """Generate dataset with common data quality issues."""
    np.random.seed(42)
    
    # Small dataset with issues
    data = {
        'feature_1': [1.5, 2.3, None, 4.1, 5.2, None, 3.8],  # Missing values
        'feature_2': [10, 20, 30, 999999, 50, 60, 70],       # Outlier
        'feature_3': [1, 1, 1, 1, 1, 1, 1],                  # No variance
        'feature_4': ['a', 'b', 'c', 'd', 'e', 'f', 'g'],    # Non-numeric
        'feature_5': [np.inf, 2.5, 3.1, -np.inf, 4.2, 5.5, 6.1]  # Infinite values
    }
    
    df = pd.DataFrame(data)
    
    # Add duplicate rows
    df = pd.concat([df, df.iloc[[0, 1]]], ignore_index=True)
    
    # Save
    output_path = Path("data") / output_file
    output_path.parent.mkdir(exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print(f"✅ Generated {output_file} (for testing error handling)")
    print(f"   Shape: {df.shape}")
    print(f"   Issues: Missing values, outliers, no variance, non-numeric, infinite values, duplicates")
    
    return df

def main():
    """Generate all sample datasets."""
    print("🎯 Generating Sample Datasets")
    print("=" * 50)
    
    # Generate different types of datasets
    generate_customer_data(n_samples=500)
    generate_iris_like_data(n_samples=300)
    generate_problematic_data()
    
    print("\n" + "=" * 50)
    print("🎉 Sample data generation complete!")
    print("\n📁 Generated files in 'data/' folder:")
    print("  • customer_segments.csv - Realistic customer data")
    print("  • iris_like.csv - Clear cluster patterns")  
    print("  • problematic_data.csv - For testing error handling")
    
    print("\n🧪 Test the datasets:")
    print("  python test_basic_pipeline.py")

if __name__ == "__main__":
    main()