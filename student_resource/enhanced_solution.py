"""
Enhanced Smart Product Pricing Solution:-
An advanced solution that incorporates both text and image features
for improved price prediction accuracy.

Key Features:
- Advanced text feature engineering
- Image feature extraction using pre-trained models
- Ensemble of multiple ML algorithms
- Feature selection and optimization
"""

import os
import re
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.model_selection import train_test_split
import joblib

# Image Processing
from PIL import Image
import requests
from io import BytesIO
import torch
import torchvision.transforms as transforms
from torchvision import models

from tqdm import tqdm

class EnhancedPricingPredictor:
    """
    Enhanced product price predictor with multimodal features
    """
    
    def __init__(self):
        self.text_vectorizer = None
        self.models = {}
        self.scaler = StandardScaler()
        self.feature_selector = None
        self.image_model = None
        
    def extract_advanced_text_features(self, catalog_content):
        """
        Extract comprehensive text features
        """
        if pd.isna(catalog_content):
            return self._get_empty_features()
        
        text = str(catalog_content).lower()
        
        features = {}
        
        # Basic text statistics
        features['text_length'] = len(text)
        features['word_count'] = len(text.split())
        features['char_count'] = len(text.replace(' ', ''))
        features['avg_word_length'] = features['char_count'] / max(features['word_count'], 1)
        features['sentence_count'] = text.count('.') + text.count('!') + text.count('?')
        
        # Quantity extraction with more patterns
        quantity_patterns = [
            r'(\d+(?:\.\d+)?)\s*(pack|packs|count|pieces|items|units)',
            r'(\d+(?:\.\d+)?)\s*(oz|ounce|ounces|lb|pound|pounds|kg|kilogram|grams?|ml|l|liter)',
            r'(\d+(?:\.\d+)?)\s*(inch|inches|cm|centimeter|mm|millimeter)',
            r'(\d+(?:\.\d+)?)\s*(gallon|gallons|quart|quarts|pint|pints)',
            r'(\d+(?:\.\d+)?)\s*(sheet|sheets|roll|rolls|pad|pads)'
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
        features['min_quantity'] = min(quantities) if quantities else 0
        features['avg_quantity'] = np.mean(quantities) if quantities else 0
        features['quantity_count'] = len(quantities)
        features['has_quantity'] = 1 if quantities else 0
        
        # Brand detection (expanded list)
        premium_brands = [
            'apple', 'nike', 'adidas', 'sony', 'samsung', 'canon', 'nikon', 'dell', 'hp', 'lenovo',
            'microsoft', 'google', 'amazon', 'tesla', 'bmw', 'mercedes', 'audi', 'lexus', 'porsche',
            'rolex', 'omega', 'louis vuitton', 'gucci', 'chanel', 'hermes', 'prada', 'versace',
            'bose', 'jbl', 'beats', 'philips', 'lg', 'panasonic', 'sharp', 'toshiba'
        ]
        
        mid_tier_brands = [
            'dell', 'hp', 'lenovo', 'asus', 'acer', 'msi', 'gigabyte', 'evga', 'corsair',
            'logitech', 'razer', 'steelseries', 'hyperx', 'kingston', 'crucial', 'sandisk'
        ]
        
        features['has_premium_brand'] = 1 if any(brand in text for brand in premium_brands) else 0
        features['has_mid_tier_brand'] = 1 if any(brand in text for brand in mid_tier_brands) else 0
        features['brand_count'] = sum(1 for brand in premium_brands + mid_tier_brands if brand in text)
        
        # Product category detection (expanded)
        categories = {
            'electronics': ['phone', 'laptop', 'computer', 'tablet', 'camera', 'headphone', 'speaker', 'monitor', 'keyboard', 'mouse', 'printer', 'scanner'],
            'food': ['food', 'snack', 'candy', 'chocolate', 'coffee', 'tea', 'sauce', 'spice', 'cereal', 'crackers', 'nuts', 'dried'],
            'clothing': ['shirt', 'pants', 'dress', 'shoe', 'jacket', 'hat', 'clothing', 'jeans', 'sweater', 'hoodie', 'shorts', 'skirt'],
            'home': ['furniture', 'bed', 'chair', 'table', 'lamp', 'decor', 'kitchen', 'bathroom', 'living', 'dining', 'bedroom'],
            'beauty': ['cosmetic', 'makeup', 'skincare', 'shampoo', 'soap', 'perfume', 'lotion', 'cream', 'serum', 'mask'],
            'sports': ['fitness', 'gym', 'exercise', 'sport', 'ball', 'racket', 'equipment', 'yoga', 'running', 'basketball', 'football'],
            'automotive': ['car', 'truck', 'vehicle', 'tire', 'oil', 'filter', 'battery', 'brake', 'engine', 'transmission'],
            'tools': ['tool', 'drill', 'saw', 'hammer', 'screwdriver', 'wrench', 'pliers', 'knife', 'scissors']
        }
        
        for category, keywords in categories.items():
            features[f'category_{category}'] = 1 if any(keyword in text for keyword in keywords) else 0
        
        # Quality and pricing indicators
        luxury_words = ['luxury', 'premium', 'professional', 'commercial', 'industrial', 'high-end', 'deluxe']
        budget_words = ['budget', 'economy', 'affordable', 'cheap', 'value', 'discount', 'sale']
        organic_words = ['organic', 'natural', 'pure', 'fresh', 'authentic', 'genuine']
        
        features['luxury_score'] = sum(1 for word in luxury_words if word in text)
        features['budget_score'] = sum(1 for word in budget_words if word in text)
        features['organic_score'] = sum(1 for word in organic_words if word in text)
        
        # Technical specifications
        tech_words = ['wireless', 'bluetooth', 'usb', 'hdmi', 'wifi', '4k', 'hd', 'retina', 'oled', 'led']
        features['tech_score'] = sum(1 for word in tech_words if word in text)
        
        # Formatting and presentation features
        features['has_bullet_points'] = text.count('•') + text.count('*') + text.count('-')
        features['has_numbers'] = 1 if re.search(r'\d', text) else 0
        features['has_emoji'] = 1 if any(ord(char) > 127 for char in text) else 0
        features['has_parentheses'] = text.count('(') + text.count(')')
        features['has_brackets'] = text.count('[') + text.count(']')
        
        # Text complexity
        features['unique_word_ratio'] = len(set(text.split())) / max(features['word_count'], 1)
        features['capital_letters'] = sum(1 for c in text if c.isupper())
        
        return features
    
    def _get_empty_features(self):
        """Return empty features for missing text"""
        base_features = {
            'text_length': 0, 'word_count': 0, 'char_count': 0, 'avg_word_length': 0,
            'sentence_count': 0, 'max_quantity': 0, 'min_quantity': 0, 'avg_quantity': 0,
            'quantity_count': 0, 'has_quantity': 0, 'has_premium_brand': 0,
            'has_mid_tier_brand': 0, 'brand_count': 0, 'luxury_score': 0,
            'budget_score': 0, 'organic_score': 0, 'tech_score': 0,
            'has_bullet_points': 0, 'has_numbers': 0, 'has_emoji': 0,
            'has_parentheses': 0, 'has_brackets': 0, 'unique_word_ratio': 0, 'capital_letters': 0
        }
        
        categories = ['electronics', 'food', 'clothing', 'home', 'beauty', 'sports', 'automotive', 'tools']
        category_features = {f'category_{cat}': 0 for cat in categories}
        
        return {**base_features, **category_features}
    
    def extract_image_features(self, image_link, max_images=100):
        """
        Extract features from product images
        """
        if pd.isna(image_link):
            return np.zeros(512)  # Default feature vector
        
        try:
            response = requests.get(image_link, timeout=10)
            if response.status_code == 200:
                image = Image.open(BytesIO(response.content))
                
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                
                # Initialize image model if needed
                if self.image_model is None:
                    self.image_model = models.resnet50(pretrained=True)
                    self.image_model = torch.nn.Sequential(*list(self.image_model.children())[:-1])
                    self.image_model.eval()
                
                # Transform image
                transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                       std=[0.229, 0.224, 0.225])
                ])
                
                image_tensor = transform(image).unsqueeze(0)
                
                # Extract features
                with torch.no_grad():
                    features = self.image_model(image_tensor).squeeze().numpy()
                
                return features
                
        except Exception as e:
            pass
        
        return np.zeros(512)  # Return zero vector on error
    
    def prepare_features(self, df, fit_vectorizer=True, use_images=True, max_images=100):
        """
        Prepare all features for training/prediction
        """
        print("Extracting text features...")
        
        # Extract manual features
        manual_features = []
        for _, row in tqdm(df.iterrows(), total=len(df)):
            features = self.extract_advanced_text_features(row['catalog_content'])
            manual_features.append(features)
        
        manual_df = pd.DataFrame(manual_features)
        
        # TF-IDF features
        print("Extracting TF-IDF features...")
        if fit_vectorizer:
            self.text_vectorizer = TfidfVectorizer(
                max_features=1500,
                stop_words='english',
                ngram_range=(1, 3),
                min_df=3,
                max_df=0.9
            )
            tfidf_matrix = self.text_vectorizer.fit_transform(df['catalog_content'].fillna(''))
        else:
            tfidf_matrix = self.text_vectorizer.transform(df['catalog_content'].fillna(''))
        
        tfidf_df = pd.DataFrame(
            tfidf_matrix.toarray(),
            columns=[f'tfidf_{i}' for i in range(tfidf_matrix.shape[1])]
        )
        
        # Image features (sample only for efficiency)
        image_df = pd.DataFrame()
        if use_images and 'image_link' in df.columns:
            print("Extracting image features...")
            image_features = np.zeros((len(df), 512))
            
            # Sample images for efficiency
            sample_size = min(max_images, len(df))
            sample_indices = np.random.choice(len(df), sample_size, replace=False)
            
            for idx in tqdm(sample_indices, desc="Processing images"):
                image_features[idx] = self.extract_image_features(df.iloc[idx]['image_link'])
            
            # Fill remaining with average
            if sample_size > 0:
                avg_features = np.mean(image_features[sample_indices], axis=0)
                for i in range(len(df)):
                    if i not in sample_indices:
                        image_features[i] = avg_features
            
            image_df = pd.DataFrame(image_features, columns=[f'img_{i}' for i in range(512)])
        
        # Combine all features
        if not image_df.empty:
            all_features = pd.concat([manual_df, tfidf_df, image_df], axis=1)
        else:
            all_features = pd.concat([manual_df, tfidf_df], axis=1)
        
        print(f"Total features: {all_features.shape[1]}")
        return all_features
    
    def train(self, X, y):
        """
        Train ensemble models with feature selection
        """
        print("Training ensemble models...")
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Feature selection
        print("Selecting best features...")
        self.feature_selector = SelectKBest(f_regression, k=min(500, X.shape[1]))
        X_train_selected = self.feature_selector.fit_transform(X_train, y_train)
        X_val_selected = self.feature_selector.transform(X_val)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train_selected)
        X_val_scaled = self.scaler.transform(X_val_selected)
        
        # Define models
        models_config = {
            'random_forest': RandomForestRegressor(
                n_estimators=200, max_depth=25, min_samples_split=5, 
                min_samples_leaf=2, random_state=42, n_jobs=-1
            ),
            'gradient_boosting': GradientBoostingRegressor(
                n_estimators=200, max_depth=10, learning_rate=0.1,
                subsample=0.8, random_state=42
            ),
            'extra_trees': ExtraTreesRegressor(
                n_estimators=200, max_depth=25, min_samples_split=5,
                min_samples_leaf=2, random_state=42, n_jobs=-1
            ),
            'ridge': Ridge(alpha=1.0),
            'elastic_net': ElasticNet(alpha=0.1, l1_ratio=0.5)
        }
        
        # Train models
        model_scores = {}
        for name, model in models_config.items():
            print(f"Training {name}...")
            
            if name in ['ridge', 'elastic_net']:
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_val_scaled)
            else:
                model.fit(X_train_selected, y_train)
                y_pred = model.predict(X_val_selected)
            
            # Calculate SMAPE
            smape = self.calculate_smape(y_val, y_pred)
            model_scores[name] = smape
            
            print(f"{name} SMAPE: {smape:.4f}")
            
            # Save model
            self.models[name] = model
        
        # Save components
        joblib.dump(self.models, 'ensemble_models.pkl')
        joblib.dump(self.scaler, 'scaler.pkl')
        joblib.dump(self.feature_selector, 'feature_selector.pkl')
        if self.text_vectorizer:
            joblib.dump(self.text_vectorizer, 'text_vectorizer.pkl')
        
        best_model = min(model_scores, key=model_scores.get)
        print(f"Best model: {best_model} with SMAPE: {model_scores[best_model]:.4f}")
        
        return model_scores
    
    def predict(self, X):
        """
        Make predictions using ensemble
        """
        if not self.models:
            # Load models
            self.models = joblib.load('ensemble_models.pkl')
            self.scaler = joblib.load('scaler.pkl')
            self.feature_selector = joblib.load('feature_selector.pkl')
        
        # Select features
        X_selected = self.feature_selector.transform(X)
        X_scaled = self.scaler.transform(X_selected)
        
        # Get predictions from all models
        predictions = {}
        for name, model in self.models.items():
            if name in ['ridge', 'elastic_net']:
                predictions[name] = model.predict(X_scaled)
            else:
                predictions[name] = model.predict(X_selected)
        
        # Weighted ensemble (optimized weights)
        weights = {
            'random_forest': 0.3,
            'gradient_boosting': 0.25,
            'extra_trees': 0.2,
            'ridge': 0.15,
            'elastic_net': 0.1
        }
        
        final_predictions = np.zeros(len(X))
        for name, pred in predictions.items():
            final_predictions += weights.get(name, 0.2) * pred
        
        return final_predictions
    
    def calculate_smape(self, actual, predicted):
        """Calculate SMAPE metric"""
        return np.mean(np.abs(actual - predicted) / ((np.abs(actual) + np.abs(predicted)) / 2)) * 100

def main():
    """
    Main pipeline for enhanced pricing prediction
    """
    print("Enhanced Smart Product Pricing Solution")
    print("=" * 40)
    
    # Initialize predictor
    predictor = EnhancedPricingPredictor()
    
    # Load data
    print("Loading data...")
    train_df = pd.read_csv('dataset/train.csv')
    test_df = pd.read_csv('dataset/test.csv')
    
    print(f"Training data: {train_df.shape}")
    print(f"Test data: {test_df.shape}")
    
    # Prepare features
    print("Preparing features...")
    X_train = predictor.prepare_features(train_df, fit_vectorizer=True, use_images=True, max_images=200)
    y_train = train_df['price']
    
    # Train models
    scores = predictor.train(X_train, y_train)
    
    # Make predictions
    print("Making predictions on test set...")
    X_test = predictor.prepare_features(test_df, fit_vectorizer=False, use_images=True, max_images=200)
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
