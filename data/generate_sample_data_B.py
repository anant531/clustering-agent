# clustering_pipeline/generate_sample_data.py

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

# Set random seed for reproducibility
np.random.seed(42)

def generate_customers_data(n_customers=500):
    """Generate realistic customer segmentation dataset."""
    
    # Define customer segments
    segments = {
        'young_spenders': {  # 30% - Young customers with high spending
            'size': int(n_customers * 0.3),
            'age_range': (22, 35),
            'income_range': (25000, 45000),
            'spending_score_range': (70, 95),
            'loyalty_years_range': (0.5, 3),
            'preferred_categories': ['Fashion', 'Electronics', 'Entertainment']
        },
        'established_savers': {  # 25% - Middle-aged customers with moderate spending
            'size': int(n_customers * 0.25),
            'age_range': (35, 50),
            'income_range': (45000, 80000),
            'spending_score_range': (40, 65),
            'loyalty_years_range': (2, 8),
            'preferred_categories': ['Home', 'Groceries', 'Health']
        },
        'premium_customers': {  # 20% - High-income customers with high spending
            'size': int(n_customers * 0.2),
            'age_range': (30, 55),
            'income_range': (70000, 150000),
            'spending_score_range': (75, 100),
            'loyalty_years_range': (3, 12),
            'preferred_categories': ['Luxury', 'Travel', 'Premium']
        },
        'budget_conscious': {  # 15% - Price-sensitive customers
            'size': int(n_customers * 0.15),
            'age_range': (25, 65),
            'income_range': (20000, 50000),
            'spending_score_range': (15, 45),
            'loyalty_years_range': (1, 6),
            'preferred_categories': ['Discount', 'Groceries', 'Essentials']
        },
        'seniors': {  # 10% - Older customers with moderate spending
            'size': int(n_customers * 0.1),
            'age_range': (55, 75),
            'income_range': (30000, 65000),
            'spending_score_range': (35, 60),
            'loyalty_years_range': (5, 20),
            'preferred_categories': ['Health', 'Home', 'Groceries']
        }
    }
    
    customers_data = []
    customer_id = 1
    
    cities = ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix', 
              'Philadelphia', 'San Antonio', 'San Diego', 'Dallas', 'San Jose']
    genders = ['Male', 'Female', 'Other']
    
    for segment_name, segment_info in segments.items():
        for _ in range(segment_info['size']):
            # Generate customer attributes with some realistic variation
            age = np.random.normal(
                np.mean(segment_info['age_range']), 
                (segment_info['age_range'][1] - segment_info['age_range'][0]) / 6
            )
            age = max(18, min(80, int(age)))  # Bound between 18-80
            
            income = np.random.normal(
                np.mean(segment_info['income_range']),
                (segment_info['income_range'][1] - segment_info['income_range'][0]) / 6
            )
            income = max(15000, int(income))  # Minimum wage bound
            
            spending_score = np.random.normal(
                np.mean(segment_info['spending_score_range']),
                (segment_info['spending_score_range'][1] - segment_info['spending_score_range'][0]) / 6
            )
            spending_score = max(1, min(100, int(spending_score)))  # Bound 1-100
            
            loyalty_years = np.random.normal(
                np.mean(segment_info['loyalty_years_range']),
                (segment_info['loyalty_years_range'][1] - segment_info['loyalty_years_range'][0]) / 6
            )
            loyalty_years = max(0.1, round(loyalty_years, 1))  # Minimum 0.1 years
            
            # Calculate derived metrics
            total_purchases = int(np.random.poisson(loyalty_years * spending_score / 10))
            total_purchases = max(1, total_purchases)
            
            average_order_value = income * (spending_score / 100) / max(1, total_purchases * 0.1)
            average_order_value = round(np.random.normal(average_order_value, average_order_value * 0.3), 2)
            average_order_value = max(10, average_order_value)
            
            last_purchase_days = int(np.random.exponential(30))
            last_purchase_days = min(365, last_purchase_days)
            
            loyalty_points = int(total_purchases * average_order_value * 0.02)  # 2% back as points
            
            customers_data.append({
                'customer_id': f'CUST_{customer_id:05d}',
                'age': age,
                'income': income,
                'spending_score': spending_score,
                'membership_years': loyalty_years,
                'gender': np.random.choice(genders),
                'city': np.random.choice(cities),
                'preferred_category': np.random.choice(segment_info['preferred_categories']),
                'total_purchases': total_purchases,
                'last_purchase_days': last_purchase_days,
                'average_order_value': average_order_value,
                'loyalty_points': loyalty_points,
                'segment_true': segment_name  # Ground truth for validation
            })
            customer_id += 1
    
    return pd.DataFrame(customers_data)

def generate_sales_data(customers_df, n_sales=1000):
    """Generate realistic sales transaction data."""
    
    products = {
        'Fashion': ['T-Shirt', 'Jeans', 'Dress', 'Shoes', 'Jacket', 'Accessories'],
        'Electronics': ['Smartphone', 'Laptop', 'Headphones', 'Tablet', 'Camera', 'Gaming'],
        'Entertainment': ['Books', 'Music', 'Movies', 'Games', 'Concerts', 'Streaming'],
        'Home': ['Furniture', 'Decor', 'Kitchen', 'Bedding', 'Appliances', 'Garden'],
        'Groceries': ['Food', 'Beverages', 'Snacks', 'Organic', 'Frozen', 'Fresh'],
        'Health': ['Supplements', 'Skincare', 'Fitness', 'Medical', 'Wellness', 'Personal'],
        'Luxury': ['Jewelry', 'Watches', 'Designer', 'Premium', 'Exclusive', 'Collections'],
        'Travel': ['Hotels', 'Flights', 'Packages', 'Insurance', 'Activities', 'Car Rental'],
        'Discount': ['Clearance', 'Bulk', 'Generic', 'Seasonal', 'Overstock', 'Basic'],
        'Essentials': ['Household', 'Cleaning', 'Personal Care', 'Basic Clothing', 'Utilities', 'Transportation']
    }
    
    payment_methods = ['Credit Card', 'Debit Card', 'PayPal', 'Cash', 'Gift Card']
    channels = ['Online', 'In-Store', 'Mobile App', 'Phone']
    
    sales_data = []
    
    for _ in range(n_sales):
        # Select random customer
        customer = customers_df.sample(1).iloc[0]
        preferred_cat = customer['preferred_category']
        
        # Higher probability of purchasing from preferred category
        if np.random.random() < 0.7 and preferred_cat in products:
            category = preferred_cat
        else:
            category = np.random.choice(list(products.keys()))
        
        product_name = np.random.choice(products[category])
        
        # Price influenced by customer income and category
        base_price = {
            'Fashion': 50, 'Electronics': 300, 'Entertainment': 25,
            'Home': 150, 'Groceries': 15, 'Health': 40,
            'Luxury': 500, 'Travel': 400, 'Discount': 10, 'Essentials': 20
        }[category]
        
        # Adjust price based on customer segment
        price_multiplier = (customer['income'] / 50000) * (customer['spending_score'] / 50)
        price_multiplier = max(0.5, min(3.0, price_multiplier))
        
        purchase_amount = round(base_price * price_multiplier * np.random.uniform(0.7, 1.5), 2)
        quantity = int(np.random.poisson(2)) + 1
        
        # Discount probability higher for budget-conscious customers
        if customer['spending_score'] < 50:
            discount_prob = 0.4
        else:
            discount_prob = 0.2
            
        discount_applied = np.random.random() < discount_prob
        if discount_applied:
            discount_pct = np.random.uniform(0.1, 0.3)  # 10-30% discount
            purchase_amount *= (1 - discount_pct)
            purchase_amount = round(purchase_amount, 2)
        
        # Satisfaction correlated with spending score and price fairness
        base_satisfaction = 3.5  # Out of 5
        satisfaction_boost = (customer['spending_score'] / 100) * 1.5
        satisfaction_rating = round(base_satisfaction + satisfaction_boost + np.random.normal(0, 0.5), 1)
        satisfaction_rating = max(1.0, min(5.0, satisfaction_rating))
        
        # Repeat purchase probability
        repeat_purchase = satisfaction_rating > 3.5 and np.random.random() < 0.6
        
        # Purchase date within last year
        days_ago = int(np.random.exponential(60))  # More recent purchases more likely
        purchase_date = datetime.now() - timedelta(days=min(365, days_ago))
        
        sales_data.append({
            'sale_id': f'SALE_{len(sales_data)+1:06d}',
            'customer_id': customer['customer_id'],
            'product_id': f'PROD_{hash(product_name) % 10000:04d}',
            'product_name': product_name,
            'product_category': category,
            'purchase_amount': purchase_amount,
            'quantity': quantity,
            'purchase_date': purchase_date.strftime('%Y-%m-%d'),
            'discount_applied': discount_applied,
            'payment_method': np.random.choice(payment_methods),
            'channel': np.random.choice(channels),
            'satisfaction_rating': satisfaction_rating,
            'repeat_purchase': repeat_purchase
        })
    
    return pd.DataFrame(sales_data)

def generate_marketing_data(customers_df, n_campaigns=300):
    """Generate marketing campaign response data."""
    
    campaigns = [
        'Summer_Sale_2024', 'Back_to_School_2024', 'Holiday_Special_2024',
        'Spring_Collection_2024', 'Black_Friday_2024', 'New_Year_2024',
        'Valentine_Special_2024', 'Easter_Promotion_2024', 'Mother_Day_2024',
        'Father_Day_2024', 'Independence_Sale_2024', 'Flash_Sale_Weekly'
    ]
    
    channels = ['Email', 'SMS', 'Social Media', 'Display Ads', 'Search Ads', 'Direct Mail']
    
    marketing_data = []
    
    for _ in range(n_campaigns):
        customer = customers_df.sample(1).iloc[0]
        campaign = np.random.choice(campaigns)
        channel = np.random.choice(channels)
        
        # Response rate influenced by customer spending score and loyalty
        base_response_rate = 0.05  # 5% base response rate
        customer_factor = (customer['spending_score'] / 100) * (customer['membership_years'] / 5)
        response_rate = base_response_rate * (1 + customer_factor)
        response_rate = min(0.3, response_rate)  # Cap at 30%
        
        responded = np.random.random() < response_rate
        
        if responded:
            # Conversion rate for responders
            conversion_rate = np.random.beta(2, 5)  # Skewed towards lower conversion
            converted = np.random.random() < conversion_rate
            
            if converted:
                # Revenue from conversion
                avg_order = customer['average_order_value']
                campaign_revenue = avg_order * np.random.uniform(0.8, 2.0)  # -20% to +100%
            else:
                campaign_revenue = 0
                conversion_rate = 0
        else:
            conversion_rate = 0
            converted = False
            campaign_revenue = 0
        
        # Campaign costs (varies by channel)
        cost_per_contact = {
            'Email': 0.1, 'SMS': 0.05, 'Social Media': 2.0,
            'Display Ads': 1.5, 'Search Ads': 3.0, 'Direct Mail': 0.8
        }[channel]
        
        campaign_cost = cost_per_contact
        roi = ((campaign_revenue - campaign_cost) / campaign_cost * 100) if campaign_cost > 0 else 0
        
        # Engagement metrics
        if channel in ['Email', 'SMS']:
            open_rate = np.random.beta(3, 2) * 0.4  # 0-40% open rate
            click_through_rate = open_rate * np.random.beta(2, 8) if responded else 0
        elif channel == 'Social Media':
            engagement_score = np.random.beta(2, 5) * 100  # 0-100 engagement score
            click_through_rate = engagement_score / 1000 if responded else 0
            open_rate = None
        else:
            engagement_score = np.random.beta(2, 3) * 100
            click_through_rate = engagement_score / 2000 if responded else 0
            open_rate = None
        
        time_spent = int(np.random.exponential(60)) if responded else 0  # seconds
        
        marketing_data.append({
            'campaign_id': f'CAMP_{len(marketing_data)+1:05d}',
            'customer_id': customer['customer_id'],
            'campaign_name': campaign,
            'channel': channel,
            'campaign_date': (datetime.now() - timedelta(days=np.random.randint(1, 365))).strftime('%Y-%m-%d'),
            'responded': responded,
            'converted': converted,
            'response_rate': round(response_rate, 4),
            'conversion_rate': round(conversion_rate, 4),
            'campaign_cost': round(campaign_cost, 2),
            'campaign_revenue': round(campaign_revenue, 2),
            'roi': round(roi, 2),
            'click_through_rate': round(click_through_rate, 4) if click_through_rate else None,
            'engagement_score': round(engagement_score, 1) if 'engagement_score' in locals() else None,
            'time_spent_seconds': time_spent
        })
    
    return pd.DataFrame(marketing_data)

def main():
    """Generate all sample datasets."""
    
    print("🎯 Generating AI Clustering Platform Sample Data...")
    print("=" * 60)

    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    # Generate customers data
    print("📊 Generating customer segmentation data...")
    customers_df = generate_customers_data(500)
    customers_df.to_csv('data/customers.csv', index=False)
    print(f"✅ Generated {len(customers_df)} customer records")
    print(f"   - Features: {list(customers_df.columns)}")
    print(f"   - Segments: {customers_df['segment_true'].value_counts().to_dict()}")
    
    # Generate sales data
    print("\n🛒 Generating sales transaction data...")
    sales_df = generate_sales_data(customers_df, 1000)
    sales_df.to_csv('data/sales_data.csv', index=False)
    print(f"✅ Generated {len(sales_df)} sales transactions")
    print(f"   - Features: {list(sales_df.columns)}")
    print(f"   - Categories: {sales_df['product_category'].value_counts().head().to_dict()}")
    
    # Generate marketing data  
    print("\n📧 Generating marketing campaign data...")
    marketing_df = generate_marketing_data(customers_df, 300)
    marketing_df.to_csv('data/marketing_data.csv', index=False)
    print(f"✅ Generated {len(marketing_df)} marketing interactions")
    print(f"   - Features: {list(marketing_df.columns)}")
    print(f"   - Channels: {marketing_df['channel'].value_counts().to_dict()}")
    
    print("\n" + "=" * 60)
    print("🎉 Sample data generation complete!")
    print("\n📁 Files created in 'sample_data/' directory:")
    print("   - customers.csv: Customer segmentation dataset")
    print("   - sales_data.csv: Transaction and purchase data") 
    print("   - marketing_data.csv: Campaign response data")
    print("\n💡 These datasets are ready to use in the clustering platform!")
    print("   You can upload them directly or use the built-in sample data options.")
    
    # Show quick stats
    print("\n📊 Quick Statistics:")
    print(f"   Total Customers: {len(customers_df):,}")
    print(f"   Total Sales: {len(sales_df):,}")
    print(f"   Total Marketing Interactions: {len(marketing_df):,}")
    print(f"   Date Range: {min(sales_df['purchase_date'])} to {max(sales_df['purchase_date'])}")
    print(f"   Customer Age Range: {customers_df['age'].min()}-{customers_df['age'].max()} years")
    print(f"   Income Range: ${customers_df['income'].min():,} - ${customers_df['income'].max():,}")

if __name__ == "__main__":
    main()