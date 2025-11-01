"""
Baseline Smart Product Pricing Solution:-

A simple but effective baseline solution for the Smart Product Pricing Challenge.
This solution focuses on the most important text features and uses a straightforward
ensemble approach.

Key Features:
- Text feature engineering from catalog_content
- Simple quantity and brand detection
- Ensemble of Random Forest and Ridge regression
- Fast and reliable
"""

import os
import re
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
from tqdm import tqdm

class BaselinePricingPredictor:
    """
    Baseline product price predictor using essential features
    """
    
    def __init__(self):
        self.text_vectorizer = None
        self.rf_model = None
        self.ridge_model = None
        self.scaler = StandardScaler()
        
    def extract_features(self, catalog_content):
        """
        Extract key features from catalog content
        """
        if pd.isna(catalog_content):
            return {
                'text_length': 0,
                'word_count': 0,
                'quantity': 0,
                'has_quantity': 0,
                'has_brand': 0,
                'quality_score': 0,
                'price_keywords': 0
            }
        
        text = str(catalog_content).lower()
        
        # Basic text features
        features = {
            'text_length': len(text),
            'word_count': len(text.split()),
            'quantity': 0,
            'has_quantity': 0,
            'has_brand': 0,
            'quality_score': 0,
            'price_keywords': 0
        }
        
        # Extract quantity (very important for pricing)
        quantity_patterns = [
            r'(\d+(?:\.\d+)?)\s*(pack|packs|count|pieces|items)',
            r'(\d+(?:\.\d+)?)\s*(oz|ounce|ounces|lb|pound|pounds|kg|kilogram|grams?|ml|l)',
            r'(\d+(?:\.\d+)?)\s*(inch|inches|cm|centimeter)'
        ]
        
        max_quantity = 0
        for pattern in quantity_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                try:
                    quantity = float(match[0])
                    max_quantity = max(max_quantity, quantity)
                except:
                    pass
        
        features['quantity'] = max_quantity
        features['has_quantity'] = 1 if max_quantity > 0 else 0
        
        # Brand detection
        brands = ['apple', 'nike', 'adidas', 'sony', 'samsung', 'canon', 'nikon', 'dell', 'hp', 'lenovo', 
                 'microsoft', 'google', 'amazon', 'tesla', 'bmw', 'mercedes', 'audi']
        features['has_brand'] = 1 if any(brand in text for brand in brands) else 0
        
        # Quality indicators
        quality_words = ['premium', 'professional', 'commercial', 'industrial', 'organic', 'natural', 'luxury']
        features['quality_score'] = sum(1 for word in quality_words if word in text)
        
        # Price-related keywords
        price_words = ['budget', 'economy', 'discount', 'sale', 'expensive', 'cheap', 'affordable', 'value']
        features['price_keywords'] = sum(1 for word in price_words if word in text)
        
        return features
    
    def prepare_data(self, df, fit_vectorizer=True):
        """
        Prepare features for training/prediction
        """
        print("Extracting text features...")
        
        # Extract manual features
        manual_features = []
        for _, row in tqdm(df.iterrows(), total=len(df)):
            features = self.extract_features(row['catalog_content'])
            manual_features.append(features)
        
        manual_df = pd.DataFrame(manual_features)
        
        # TF-IDF features
        print("Extracting TF-IDF features...")
        if fit_vectorizer:
            self.text_vectorizer = TfidfVectorizer(
                max_features=1000,
                stop_words='english',
                ngram_range=(1, 2),
                min_df=5,
                max_df=0.8
            )
            tfidf_matrix = self.text_vectorizer.fit_transform(df['catalog_content'].fillna(''))
        else:
            tfidf_matrix = self.text_vectorizer.transform(df['catalog_content'].fillna(''))
        
        tfidf_df = pd.DataFrame(
            tfidf_matrix.toarray(),
            columns=[f'tfidf_{i}' for i in range(tfidf_matrix.shape[1])]
        )
        
        # Combine features
        all_features = pd.concat([manual_df, tfidf_df], axis=1)
        
        print(f"Total features: {all_features.shape[1]}")
        return all_features
    
    def train(self, X, y):
        """
        Train the ensemble models
        """
        print("Training models...")
        
        # Split data for validation
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features for Ridge regression
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        # Train Random Forest
        print("Training Random Forest...")
        self.rf_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=20,
            min_samples_split=10,
            random_state=42,
            n_jobs=-1
        )
        self.rf_model.fit(X_train, y_train)
        
        # Train Ridge regression
        print("Training Ridge regression...")
        self.ridge_model = Ridge(alpha=1.0)
        self.ridge_model.fit(X_train_scaled, y_train)
        
        # Evaluate models
        rf_pred = self.rf_model.predict(X_val)
        ridge_pred = self.ridge_model.predict(X_val_scaled)
        
        rf_smape = self.calculate_smape(y_val, rf_pred)
        ridge_smape = self.calculate_smape(y_val, ridge_pred)
        
        print(f"Random Forest SMAPE: {rf_smape:.4f}")
        print(f"Ridge SMAPE: {ridge_smape:.4f}")
        
        # Save models
        joblib.dump(self.rf_model, 'rf_model.pkl')
        joblib.dump(self.ridge_model, 'ridge_model.pkl')
        joblib.dump(self.scaler, 'scaler.pkl')
        if self.text_vectorizer:
            joblib.dump(self.text_vectorizer, 'text_vectorizer.pkl')
        
        return {'rf_smape': rf_smape, 'ridge_smape': ridge_smape}
    
    def predict(self, X):
        """
        Make predictions using ensemble
        """
        if self.rf_model is None or self.ridge_model is None:
            # Load models
            self.rf_model = joblib.load('rf_model.pkl')
            self.ridge_model = joblib.load('ridge_model.pkl')
            self.scaler = joblib.load('scaler.pkl')
        
        # Get predictions from both models
        rf_pred = self.rf_model.predict(X)
        X_scaled = self.scaler.transform(X)
        ridge_pred = self.ridge_model.predict(X_scaled)
        
        # Ensemble prediction (weighted average)
        ensemble_pred = 0.7 * rf_pred + 0.3 * ridge_pred
        
        return ensemble_pred
    
    def calculate_smape(self, actual, predicted):
        """Calculate SMAPE metric"""
        return np.mean(np.abs(actual - predicted) / ((np.abs(actual) + np.abs(predicted)) / 2)) * 100

def main():
    """
    Main pipeline for baseline pricing prediction
    """
    print("Baseline Smart Product Pricing Solution")
    print("=" * 40)
    
    # Initialize predictor
    predictor = BaselinePricingPredictor()
    
    # Load data
    print("Loading data...")
    train_df = pd.read_csv('dataset/train.csv')
    test_df = pd.read_csv('dataset/test.csv')
    
    print(f"Training data: {train_df.shape}")
    print(f"Test data: {test_df.shape}")
    
    # Prepare features
    print("Preparing features...")
    X_train = predictor.prepare_data(train_df, fit_vectorizer=True)
    y_train = train_df['price']
    
    # Train models
    scores = predictor.train(X_train, y_train)
    
    # Make predictions
    print("Making predictions on test set...")
    X_test = predictor.prepare_data(test_df, fit_vectorizer=False)
    predictions = predictor.predict(X_test)
    
    # Create output
    output_df = pd.DataFrame({
        'sample_id': test_df['sample_id'],
        'price': predictions
    })
    
    # Ensure positive prices
    output_df['price'] = np.maximum(output_df['price'], 0.01)
    
    # Save predictions
    output_path = 'dataset/test_out.csv'
    output_df.to_csv(output_path, index=False)
    
    print(f"\nPredictions saved to {output_path}")
    print(f"Prediction statistics:")
    print(output_df['price'].describe())
    
    return output_df

if __name__ == "__main__":
    main()
