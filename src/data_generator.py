"""
Data Generator for Simulated Ads Dataset
Generates realistic click-through rate data with user, ad, and temporal features
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta


def generate_ads_dataset(output_path, n_samples=50000, random_state=42):
    """
    Generate a simulated ads dataset with realistic CTR patterns.
    
    Parameters:
    -----------
    output_path : str
        Path to save the generated CSV file
    n_samples : int
        Number of records to generate
    random_state : int
        Random seed for reproducibility
    """
    
    np.random.seed(random_state)
    
    print(f"Generating {n_samples} ad impression records...")
    
    # User IDs and properties
    n_users = int(np.sqrt(n_samples))
    user_ids = np.random.randint(1, n_users, n_samples)
    
    # Ad categories
    categories = ['Electronics', 'Fashion', 'Home', 'Sports', 'Books', 'Health', 'Travel', 'Food']
    ad_categories = np.random.choice(categories, n_samples)
    
    # Ad IDs
    n_ads = int(np.sqrt(n_samples) * 0.5)
    ad_ids = np.random.randint(1, n_ads, n_samples)
    
    # Time features
    base_date = datetime(2024, 1, 1)
    timestamps = [base_date + timedelta(days=int(x)) for x in np.random.normal(180, 100, n_samples)]
    
    # Hour of day
    hour_of_day = np.random.randint(0, 24, n_samples)
    
    # Device type
    devices = np.random.choice(['Mobile', 'Desktop', 'Tablet'], n_samples, p=[0.6, 0.3, 0.1])
    
    # User engagement features
    user_click_history = np.random.exponential(scale=5, size=n_samples).astype(int)
    user_session_count = np.random.exponential(scale=10, size=n_samples).astype(int)
    
    # Recency features: days since last interaction
    days_since_last_interaction = np.random.exponential(scale=7, size=n_samples).astype(int)
    
    # Ad relevance and matching features
    category_affinity = np.random.uniform(0, 1, n_samples)  # 0-1 score
    semantic_similarity = np.random.uniform(0, 1, n_samples)  # Embedding similarity
    
    # Engagement likelihood features
    engagement_rate = np.random.beta(2, 5, n_samples)  # User's historical engagement rate
    category_click_rate = np.random.beta(2, 8, n_samples)  # Category's baseline CTR
    
    # CTR calculation with feature interactions
    # Higher probability when:
    # - Semantic similarity is high
    # - Category affinity is high
    # - User has recent interactions
    # - Device is mobile (typically higher engagement)
    
    ctr_probability = (
        0.02 +  # Base CTR
        0.03 * semantic_similarity +  # Semantic relevance
        0.02 * category_affinity +  # Category match
        0.02 * engagement_rate +  # User engagement propensity
        0.01 * category_click_rate +  # Category baseline
        0.01 * np.exp(-days_since_last_interaction / 10) +  # Recency bonus (exponential decay)
        0.005 * (devices == 'Mobile').astype(float)  # Mobile device boost
    )
    
    # Clip to valid probability range
    ctr_probability = np.clip(ctr_probability, 0.001, 0.3)
    
    # Generate click labels
    clicks = (np.random.random(n_samples) < ctr_probability).astype(int)
    
    # Create DataFrame
    df = pd.DataFrame({
        'user_id': user_ids,
        'ad_id': ad_ids,
        'category': ad_categories,
        'timestamp': timestamps,
        'hour_of_day': hour_of_day,
        'device': devices,
        'user_click_history': user_click_history,
        'user_session_count': user_session_count,
        'days_since_last_interaction': days_since_last_interaction,
        'category_affinity': category_affinity,
        'semantic_similarity': semantic_similarity,
        'engagement_rate': engagement_rate,
        'category_click_rate': category_click_rate,
        'click': clicks,
    })
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    
    print(f"✓ Dataset saved to {output_path}")
    print(f"  Total samples: {len(df)}")
    print(f"  Positive samples (clicks): {df['click'].sum()} ({df['click'].mean():.2%})")
    print(f"  Features: {df.shape[1] - 1}")
    print(f"\nDataset Summary:")
    print(df.head())
    
    return df


if __name__ == "__main__":
    generate_ads_dataset('data/ads_dataset.csv', n_samples=50000)
