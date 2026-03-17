"""
Feature Engineering for CTR Prediction
Handles extraction and transformation of recency, semantic similarity, and engagement features
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split


class FeatureEngineer:
    """Feature engineering pipeline for CTR prediction"""
    
    def __init__(self, test_size=0.2, random_state=42):
        self.test_size = test_size
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_groups = {
            'recency': [],
            'semantic': [],
            'engagement': [],
            'basic': []
        }
        
    def extract_recency_features(self, df):
        """
        Extract user recency features:
        - Days since last interaction
        - Interaction frequency
        """
        features = pd.DataFrame()
        features['days_since_last_interaction'] = df['days_since_last_interaction']
        features['interaction_frequency'] = df['user_session_count']
        
        self.feature_groups['recency'] = features.columns.tolist()
        return features
    
    def extract_semantic_features(self, df):
        """
        Extract semantic similarity features:
        - Semantic similarity score
        - Category affinity
        """
        features = pd.DataFrame()
        features['semantic_similarity'] = df['semantic_similarity']
        features['category_affinity'] = df['category_affinity']
        
        self.feature_groups['semantic'] = features.columns.tolist()
        return features
    
    def extract_engagement_features(self, df):
        """
        Extract historical engagement features:
        - User click history
        - Historical engagement rate
        - Category click rate
        """
        features = pd.DataFrame()
        features['user_click_history'] = df['user_click_history']
        features['engagement_rate'] = df['engagement_rate']
        features['category_click_rate'] = df['category_click_rate']
        
        self.feature_groups['engagement'] = features.columns.tolist()
        return features
    
    def extract_basic_features(self, df):
        """
        Extract basic features:
        - Hour of day
        - Device type
        - User and Ad IDs (encoded)
        """
        features = pd.DataFrame()
        
        # Hour features (cyclical encoding)
        features['hour_sin'] = np.sin(2 * np.pi * df['hour_of_day'] / 24)
        features['hour_cos'] = np.cos(2 * np.pi * df['hour_of_day'] / 24)
        
        # Device encoding
        if 'device_Mobile' not in features.columns:
            device_dummies = pd.get_dummies(df['device'], prefix='device')
            features = pd.concat([features, device_dummies], axis=1)
        
        # Category encoding
        if 'category_encoded' not in features.columns:
            le = LabelEncoder()
            features['category_encoded'] = le.fit_transform(df['category'])
            self.label_encoders['category'] = le
        
        self.feature_groups['basic'] = features.columns.tolist()
        return features
    
    def prepare_data(self, csv_path, test_size=None):
        """
        Load data and prepare features for modeling.
        
        Parameters:
        -----------
        csv_path : str
            Path to the ads dataset CSV
        test_size : float, optional
            Override default test size
            
        Returns:
        --------
        X_train, X_test, y_train, y_test, scaler : tuple
            Training and test sets with target variable
        """
        if test_size is not None:
            self.test_size = test_size
            
        # Load data
        print(f"Loading data from {csv_path}...")
        df = pd.read_csv(csv_path)
        
        # Extract feature groups
        print("Extracting features...")
        recency_features = self.extract_recency_features(df)
        semantic_features = self.extract_semantic_features(df)
        engagement_features = self.extract_engagement_features(df)
        basic_features = self.extract_basic_features(df)
        
        # Combine all features
        X = pd.concat([
            recency_features,
            semantic_features,
            engagement_features,
            basic_features
        ], axis=1)
        
        y = df['click'].values
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y  # Maintain class distribution
        )
        
        # Scale features
        print("Scaling features...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns)
        
        print(f"✓ Data prepared")
        print(f"  Training set: {X_train_scaled.shape}")
        print(f"  Test set: {X_test_scaled.shape}")
        print(f"  Features: {X_train_scaled.shape[1]}")
        print(f"  Feature groups: {list(self.feature_groups.keys())}")
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def get_feature_groups(self):
        """Return the feature groupings"""
        return self.feature_groups
    
    def get_group_features(self, group_name):
        """Get feature names for a specific group"""
        return self.feature_groups.get(group_name, [])


def create_feature_subsets(X_train, X_test, feature_groups):
    """
    Create feature subsets for ablation studies.
    
    Parameters:
    -----------
    X_train, X_test : DataFrame
        Training and test feature sets
    feature_groups : dict
        Dictionary mapping group names to feature lists
        
    Returns:
    --------
    subsets : dict
        Dictionary of feature subsets for ablation
    """
    all_features = set(X_train.columns)
    subsets = {}
    
    # All features
    subsets['all_features'] = list(all_features)
    
    # Each feature group
    for group_name in feature_groups:
        subsets[f'only_{group_name}'] = feature_groups[group_name]
    
    # All except each group
    for group_name in feature_groups:
        excluded_features = list(all_features - set(feature_groups[group_name]))
        subsets[f'without_{group_name}'] = excluded_features
    
    return subsets
