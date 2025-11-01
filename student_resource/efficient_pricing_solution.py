"""
Efficient Smart Product Pricing Solution:-

A more efficient version focusing on the most impactful features
for the Smart Product Pricing Challenge.

Key Features:
- Text feature engineering from catalog_content
- Lightweight image feature extraction
- Ensemble modeling with optimized hyperparameters
- Fast training and prediction
"""

import os
import re
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

# Image Processing
from PIL import Image
import requests
from io import BytesIO

# Utils
from tqdm import tqdm

class EfficientPricingPredictor:
    """
    Efficient product price predictor using optimized feature extraction
    """
    
    def __init__(self, model_save_path='models/'):
        self.model_save_path = model_save_path
        self.text_vectorizer = None
        self.ensemble_models = {}
        self.scaler = StandardScaler()
        
        # Create directory
        os.makedirs(model_save_path, exist_ok=True)
        
    def extract_text_features(self, catalog_content):
        """
        Extract key features from catalog content efficiently
        """
        features = {}
        
        if pd.isna(catalog_content):
            return self._get_empty_text_features()
        
        text = str(catalog_content).lower()
        
        # Essential text statistics
        features['text_length'] = len(text)
        features['word_count'] = len(text.split())
        features['avg_word_length'] = len(text.replace(' ', '')) / max(len(text.split()), 1)
        
        # Quantity extraction (most important for pricing)
        quantity_patterns = [
            r'(\d+(?:\.\d+)?)\s*(pack|packs|count|pieces|items)',
            r'(\d+(?:\.\d+)?)\s*(oz|ounce|ounces|lb|pound|pounds|kg|kilogram|grams?|ml|l)',
            r'(\d+(?:\.\d+)?)\s*(inch|inches|cm|centimeter)'
        ]
        
        quantities = []
        for pattern in quantity_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                try:
                    quantities.append(float(match[0]))
                except:
                    pass
        
        features['max_quantity'] = max(quantities) if quantities else 0
        features['has_quantity'] = 1 if quantities else 0
        features['quantity_count'] = len(quantities)
        
        # Brand detection (affects pricing significantly)
        premium_brands = ['apple', 'nike', 'adidas', 'sony', 'samsung', 'canon', 'nikon', 'dell', 'hp', 'lenovo']
        features['has_premium_brand'] = 1 if any(brand in text for brand in premium_brands) else 0
        
        # Product category detection
        categories = {
            'electronics': ['phone', 'laptop', 'computer', 'tablet', 'camera', 'headphone', 'speaker'],
            'food': ['food', 'snack', 'candy', 'chocolate', 'coffee', 'tea', 'sauce', 'spice'],
            'clothing': ['shirt', 'pants', 'dress', 'shoe', 'jacket', 'hat', 'clothing'],
            'home': ['furniture', 'bed', 'chair', 'table', 'lamp', 'decor', 'kitchen'],
            'beauty': ['cosmetic', 'makeup', 'skincare', 'shampoo', 'soap', 'perfume'],
            'sports': ['fitness', 'gym', 'exercise', 'sport', 'ball', 'racket', 'equipment']
        }
        
        for category, keywords in categories.items():
            features[f'category_{category}'] = 1 if any(keyword in text for keyword in keywords) else 0
        
        # Quality indicators
        quality_words = ['premium', 'professional', 'commercial', 'industrial', 'organic', 'natural']
        features['quality_score'] = sum(1 for word in quality_words if word in text)
        
        # Price-related keywords
        price_words = ['luxury', 'budget', 'economy', 'discount', 'sale', 'expensive', 'cheap', 'affordable']
        features['price_keyword_score'] = sum(1 for word in price_words if word in text)
        
        # Formatting features
        features['has_bullet_points'] = text.count('•') + text.count('*') + text.count('-')
        features['has_numbers'] = 1 if re.search(r'\d', text) else 0
        features['has_emoji'] = 1 if any(ord(char) > 127 for char in text) else 0
        
        return features
    
    def _get_empty_text_features(self):
        """Return empty features for missing text"""
        return {
            'text_length': 0, 'word_count': 0, 'avg_word_length': 0,
            'max_quantity': 0, 'has_quantity': 0, 'quantity_count': 0,
            'has_premium_brand': 0, 'quality_score': 0, 'price_keyword_score': 0,
            'has_bullet_points': 0, 'has_numbers': 0, 'has_emoji': 0
        } + {f'category_{cat}': 0 for cat in ['electronics', 'food', 'clothing', 'home', 'beauty', 'sports']}
    
    def extract_simple_image_features(self, image_link):
        """
        Extract simple features from images (aspect ratio, dominant colors)
        """
        if pd.isna(image_link):
            return np.array([1.0, 0.0, 0.0, 0.0])  # Default features
        
        try:
            response = requests.get(image_link, timeout=5)
            if response.status_code == 200:
                image = Image.open(BytesIO(response.content))
                
                # Basic image properties
                width, height = image.size
                aspect_ratio = width / height if height > 0 else 1.0
                
                # Convert to RGB and get basic color stats
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                
                # Sample some pixels for color analysis
                pixels = np.array(image)
                sample_pixels = pixels[::50, ::50].reshape(-1, 3)  # Sample every 50th pixel
                
                # Calculate average color values
                avg_r = np.mean(sample_pixels[:, 0]) / 255.0
                avg_g = np.mean(sample_pixels[:, 1]) / 255.0
                avg_b = np.mean(sample_pixels[:, 2]) / 255.0
                
                return np.array([aspect_ratio, avg_r, avg_g, avg_b])
                
        except Exception as e:
            pass
        
        return np.array([1.0, 0.0, 0.0, 0.0])  # Default on error
    
    def prepare_features(self, df, fit_vectorizer=True):
        """
        Prepare all features efficiently
        """
        print("Extracting text features...")
        text_features_list = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing text"):
            features = self.extract_text_features(row['catalog_content'])
            text_features_list.append(features)
        
        text_features_df = pd.DataFrame(text_features_list)
        
        # TF-IDF features (reduced dimensions for efficiency)
        print("Extracting TF-IDF features...")
        if fit_vectorizer:
            self.text_vectorizer = TfidfVectorizer(
                max_features=500,  # Reduced from 1000
                stop_words='english',
                ngram_range=(1, 2),
                min_df=10,  # Increased for efficiency
                max_df=0.8
            )
            tfidf_features = self.text_vectorizer.fit_transform(df['catalog_content'].fillna(''))
        else:
            tfidf_features = self.text_vectorizer.transform(df['catalog_content'].fillna(''))
        
        tfidf_df = pd.DataFrame(tfidf_features.toarray(), 
                               columns=[f'tfidf_{i}' for i in range(tfidf_features.shape[1])])
        
        # Simple image features (sample only for efficiency)
        print("Extracting image features...")
        image_features = np.zeros((len(df), 4))
        
        # Sample images for training efficiency
        sample_size = min(500, len(df))
        sample_indices = np.random.choice(len(df), sample_size, replace=False)
        
        for idx in tqdm(sample_indices, desc="Processing images"):
            image_features[idx] = self.extract_simple_image_features(df.iloc[idx]['image_link'])
        
        # Fill remaining with average
        avg_features = np.mean(image_features[sample_indices], axis=0)
        for i in range(len(df)):
            if i not in sample_indices:
                image_features[i] = avg_features
        
        image_df = pd.DataFrame(image_features, columns=['aspect_ratio', 'avg_r', 'avg_g', 'avg_b'])
        
        # Combine all features
        all_features = pd.concat([text_features_df, tfidf_df, image_df], axis=1)
        
        print(f"Total features: {all_features.shape[1]}")
        return all_features
    
    def train_models(self, X, y):
        """
        Train optimized ensemble models
        """
        print("Training ensemble models...")
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        # Optimized models
        models_config = {
            'random_forest': RandomForestRegressor(
                n_estimators=200, 
                max_depth=20,
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=42, 
                n_jobs=-1
            ),
            'gradient_boosting': GradientBoostingRegressor(
                n_estimators=200,
                max_depth=8,
                learning_rate=0.1,
                subsample=0.8,
                random_state=42
            ),
            'ridge': Ridge(alpha=10.0)  # Optimized alpha
        }
        
        # Train and evaluate
        model_scores = {}
        for name, model in models_config.items():
            print(f"Training {name}...")
            
            if name == 'ridge':
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_val_scaled)
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_val)
            
            # Calculate SMAPE
            smape = self.calculate_smape(y_val, y_pred)
            model_scores[name] = smape
            
            print(f"{name} SMAPE: {smape:.4f}")
            
            # Save model
            self.ensemble_models[name] = model
            joblib.dump(model, os.path.join(self.model_save_path, f'{name}_model.pkl'))
        
        # Save components
        joblib.dump(self.scaler, os.path.join(self.model_save_path, 'scaler.pkl'))
        if self.text_vectorizer:
            joblib.dump(self.text_vectorizer, os.path.join(self.model_save_path, 'text_vectorizer.pkl'))
        
        print(f"Best model: {min(model_scores, key=model_scores.get)}")
        return model_scores
    
    def calculate_smape(self, actual, predicted):
        """Calculate SMAPE metric"""
        return np.mean(np.abs(actual - predicted) / ((np.abs(actual) + np.abs(predicted)) / 2)) * 100
    
    def predict(self, df):
        """
        Make predictions on new data
        """
        print("Preparing features for prediction...")
        X = self.prepare_features(df, fit_vectorizer=False)
        
        # Load models if needed
        if not self.ensemble_models:
            self.load_models()
        
        # Make predictions
        predictions = {}
        for name, model in self.ensemble_models.items():
            if name == 'ridge':
                X_scaled = self.scaler.transform(X)
                predictions[name] = model.predict(X_scaled)
            else:
                predictions[name] = model.predict(X)
        
        # Weighted ensemble
        weights = {'random_forest': 0.5, 'gradient_boosting': 0.4, 'ridge': 0.1}
        final_predictions = np.zeros(len(df))
        
        for name, pred in predictions.items():
            final_predictions += weights.get(name, 0.33) * pred
        
        return final_predictions
    
    def load_models(self):
        """Load pre-trained models"""
        for model_name in ['random_forest', 'gradient_boosting', 'ridge']:
            model_path = os.path.join(self.model_save_path, f'{model_name}_model.pkl')
            if os.path.exists(model_path):
                self.ensemble_models[model_name] = joblib.load(model_path)
        
        scaler_path = os.path.join(self.model_save_path, 'scaler.pkl')
        if os.path.exists(scaler_path):
            self.scaler = joblib.load(scaler_path)
        
        vectorizer_path = os.path.join(self.model_save_path, 'text_vectorizer.pkl')
        if os.path.exists(vectorizer_path):
            self.text_vectorizer = joblib.load(vectorizer_path)

def main():
    """
    Main pipeline for efficient pricing prediction
    """
    print("Efficient Smart Product Pricing Solution")
    print("=" * 45)
    
    # Initialize predictor
    predictor = EfficientPricingPredictor()
    
    # Load data
    print("Loading data...")
    train_df = pd.read_csv('dataset/train.csv')
    test_df = pd.read_csv('dataset/test.csv')
    
    print(f"Training: {train_df.shape}, Test: {test_df.shape}")
    
    # Prepare features
    print("Preparing training features...")
    X_train = predictor.prepare_features(train_df, fit_vectorizer=True)
    y_train = train_df['price']
    
    # Train models
    model_scores = predictor.train_models(X_train, y_train)
    
    # Make predictions
    print("Making predictions...")
    test_predictions = predictor.predict(test_df)
    
    # Create output
    output_df = pd.DataFrame({
        'sample_id': test_df['sample_id'],
        'price': test_predictions
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
