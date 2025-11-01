"""
Smart Product Pricing Challenge Solution:-

This solution implements a comprehensive ML approach for predicting product prices
using both textual features (catalog_content) and visual features (product images).

Key Features:
- Text feature engineering from catalog_content
- Image feature extraction using pre-trained models
- Ensemble modeling approach
- Robust preprocessing and validation
"""

import os
import re
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

# Deep Learning Libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision import models

# Text Processing
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string

# Image Processing
from PIL import Image
import requests
from io import BytesIO

# Utils
from utils import download_images
from tqdm import tqdm
import pickle

class ProductPricingPredictor:
    """
    Main class for product price prediction using multimodal features
    """
    
    def __init__(self, model_save_path='models/', image_cache_path='image_cache/'):
        self.model_save_path = model_save_path
        self.image_cache_path = image_cache_path
        self.text_vectorizer = None
        self.image_model = None
        self.ensemble_models = {}
        self.scaler = StandardScaler()
        
        # Create directories
        os.makedirs(model_save_path, exist_ok=True)
        os.makedirs(image_cache_path, exist_ok=True)
        
        # Download required NLTK data
        try:
            nltk.download('stopwords', quiet=True)
            nltk.download('punkt', quiet=True)
        except:
            pass
        
        self.stop_words = set(stopwords.words('english'))
        
    def extract_text_features(self, catalog_content):
        """
        Extract comprehensive features from catalog content
        """
        features = {}
        
        if pd.isna(catalog_content):
            return self._get_empty_text_features()
        
        text = str(catalog_content).lower()
        
        # Basic text statistics
        features['text_length'] = len(text)
        features['word_count'] = len(text.split())
        features['sentence_count'] = text.count('.') + text.count('!') + text.count('?')
        features['char_count'] = len(text.replace(' ', ''))
        features['avg_word_length'] = features['char_count'] / max(features['word_count'], 1)
        
        # Brand detection
        features['has_brand'] = any(brand in text for brand in ['nike', 'adidas', 'apple', 'samsung', 'sony', 'canon', 'nikon'])
        
        # Product type detection
        product_types = ['food', 'electronics', 'clothing', 'books', 'toys', 'beauty', 'health', 'home', 'sports']
        for ptype in product_types:
            features[f'product_type_{ptype}'] = 1 if ptype in text else 0
        
        # Quantity and size extraction
        quantity_patterns = [
            r'(\d+)\s*(pack|packs|count|pieces|items)',
            r'(\d+)\s*(oz|ounce|ounces|lb|pound|pounds|kg|kilogram|grams|g)',
            r'(\d+)\s*(inch|inches|cm|centimeter)',
            r'(\d+)\s*(ml|milliliter|l|liter)'
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
        
        # Price-related keywords
        price_keywords = ['premium', 'luxury', 'budget', 'economy', 'discount', 'sale', 'expensive', 'cheap']
        features['price_keyword_score'] = sum(1 for keyword in price_keywords if keyword in text)
        
        # Special characters and formatting
        features['has_emoji'] = 1 if any(ord(char) > 127 for char in text) else 0
        features['has_bullet_points'] = text.count('•') + text.count('*') + text.count('-')
        features['has_numbers'] = 1 if re.search(r'\d', text) else 0
        
        # Material/quality indicators
        quality_indicators = ['organic', 'natural', 'premium', 'professional', 'commercial', 'industrial']
        features['quality_score'] = sum(1 for indicator in quality_indicators if indicator in text)
        
        return features
    
    def _get_empty_text_features(self):
        """Return empty features for missing text"""
        return {
            'text_length': 0, 'word_count': 0, 'sentence_count': 0, 'char_count': 0,
            'avg_word_length': 0, 'has_brand': 0, 'has_quantity': 0, 'quantity': 0,
            'price_keyword_score': 0, 'has_emoji': 0, 'has_bullet_points': 0,
            'has_numbers': 0, 'quality_score': 0
        } + {f'product_type_{ptype}': 0 for ptype in ['food', 'electronics', 'clothing', 'books', 'toys', 'beauty', 'health', 'home', 'sports']}
    
    def extract_image_features(self, image_link, use_cache=True):
        """
        Extract features from product images using pre-trained CNN
        """
        if pd.isna(image_link):
            return np.zeros(512)  # Default feature vector
        
        # Check cache first
        if use_cache:
            cache_file = os.path.join(self.image_cache_path, f"{hash(image_link)}.npy")
            if os.path.exists(cache_file):
                return np.load(cache_file)
        
        try:
            # Download and process image
            response = requests.get(image_link, timeout=10)
            if response.status_code == 200:
                image = Image.open(BytesIO(response.content))
                
                # Convert to RGB if needed
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                
                # Resize and normalize
                transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                       std=[0.229, 0.224, 0.225])
                ])
                
                image_tensor = transform(image).unsqueeze(0)
                
                # Extract features using pre-trained ResNet
                if self.image_model is None:
                    self.image_model = models.resnet50(pretrained=True)
                    self.image_model = nn.Sequential(*list(self.image_model.children())[:-1])  # Remove final layer
                    self.image_model.eval()
                
                with torch.no_grad():
                    features = self.image_model(image_tensor).squeeze().numpy()
                
                # Cache the features
                if use_cache:
                    np.save(cache_file, features)
                
                return features
                
        except Exception as e:
            print(f"Error processing image {image_link}: {e}")
        
        return np.zeros(512)  # Return zero vector on error
    
    def prepare_features(self, df, fit_vectorizer=True):
        """
        Prepare all features for training/prediction
        """
        print("Extracting text features...")
        text_features_list = []
        for _, row in tqdm(df.iterrows(), total=len(df)):
            features = self.extract_text_features(row['catalog_content'])
            text_features_list.append(features)
        
        # Convert to DataFrame
        text_features_df = pd.DataFrame(text_features_list)
        
        # TF-IDF features from catalog content
        print("Extracting TF-IDF features...")
        if fit_vectorizer:
            self.text_vectorizer = TfidfVectorizer(
                max_features=1000,
                stop_words='english',
                ngram_range=(1, 2),
                min_df=5,
                max_df=0.7
            )
            tfidf_features = self.text_vectorizer.fit_transform(df['catalog_content'].fillna(''))
        else:
            tfidf_features = self.text_vectorizer.transform(df['catalog_content'].fillna(''))
        
        # Convert sparse matrix to dense
        tfidf_df = pd.DataFrame(tfidf_features.toarray(), 
                               columns=[f'tfidf_{i}' for i in range(tfidf_features.shape[1])])
        
        # Extract image features (sample only for training due to computational cost)
        print("Extracting image features...")
        if len(df) > 1000:  # For large datasets, sample images
            sample_indices = np.random.choice(len(df), min(1000, len(df)), replace=False)
            image_features = np.zeros((len(df), 512))
            
            for i, idx in enumerate(tqdm(sample_indices)):
                image_features[idx] = self.extract_image_features(df.iloc[idx]['image_link'])
            
            # Fill remaining with average features
            avg_features = np.mean(image_features[sample_indices], axis=0)
            for i in range(len(df)):
                if i not in sample_indices:
                    image_features[i] = avg_features
        else:
            image_features = np.zeros((len(df), 512))
            for i, row in enumerate(tqdm(df.iterrows(), total=len(df))):
                image_features[i] = self.extract_image_features(row[1]['image_link'])
        
        image_df = pd.DataFrame(image_features, columns=[f'img_feat_{i}' for i in range(512)])
        
        # Combine all features
        all_features = pd.concat([text_features_df, tfidf_df, image_df], axis=1)
        
        return all_features
    
    def train_models(self, X, y):
        """
        Train ensemble of models
        """
        print("Training ensemble models...")
        
        # Split data for validation
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        # Define models
        models_config = {
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
            'gradient_boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'ridge': Ridge(alpha=1.0),
            'lasso': Lasso(alpha=0.1)
        }
        
        # Train and evaluate models
        model_scores = {}
        for name, model in models_config.items():
            print(f"Training {name}...")
            
            # Train model
            if name in ['ridge', 'lasso']:
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_val_scaled)
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_val)
            
            # Calculate SMAPE score
            smape = self.calculate_smape(y_val, y_pred)
            model_scores[name] = smape
            
            print(f"{name} SMAPE: {smape:.4f}")
            
            # Save model
            self.ensemble_models[name] = model
            joblib.dump(model, os.path.join(self.model_save_path, f'{name}_model.pkl'))
        
        # Save other components
        joblib.dump(self.scaler, os.path.join(self.model_save_path, 'scaler.pkl'))
        if self.text_vectorizer:
            joblib.dump(self.text_vectorizer, os.path.join(self.model_save_path, 'text_vectorizer.pkl'))
        
        print(f"Model training completed. Best model: {min(model_scores, key=model_scores.get)}")
        return model_scores
    
    def calculate_smape(self, actual, predicted):
        """Calculate Symmetric Mean Absolute Percentage Error"""
        return np.mean(np.abs(actual - predicted) / ((np.abs(actual) + np.abs(predicted)) / 2)) * 100
    
    def predict(self, df):
        """
        Make predictions on new data
        """
        print("Preparing features for prediction...")
        X = self.prepare_features(df, fit_vectorizer=False)
        
        # Load models if not already loaded
        if not self.ensemble_models:
            self.load_models()
        
        # Make predictions with ensemble
        predictions = {}
        for name, model in self.ensemble_models.items():
            if name in ['ridge', 'lasso']:
                X_scaled = self.scaler.transform(X)
                predictions[name] = model.predict(X_scaled)
            else:
                predictions[name] = model.predict(X)
        
        # Ensemble prediction (weighted average)
        weights = {'random_forest': 0.4, 'gradient_boosting': 0.3, 'ridge': 0.2, 'lasso': 0.1}
        final_predictions = np.zeros(len(df))
        
        for name, pred in predictions.items():
            final_predictions += weights.get(name, 0.25) * pred
        
        return final_predictions
    
    def load_models(self):
        """Load pre-trained models"""
        for model_name in ['random_forest', 'gradient_boosting', 'ridge', 'lasso']:
            model_path = os.path.join(self.model_save_path, f'{model_name}_model.pkl')
            if os.path.exists(model_path):
                self.ensemble_models[model_name] = joblib.load(model_path)
        
        # Load other components
        scaler_path = os.path.join(self.model_save_path, 'scaler.pkl')
        if os.path.exists(scaler_path):
            self.scaler = joblib.load(scaler_path)
        
        vectorizer_path = os.path.join(self.model_save_path, 'text_vectorizer.pkl')
        if os.path.exists(vectorizer_path):
            self.text_vectorizer = joblib.load(vectorizer_path)

def main():
    """
    Main training and prediction pipeline
    """
    print("Smart Product Pricing Challenge Solution")
    print("=" * 50)
    
    # Initialize predictor
    predictor = ProductPricingPredictor()
    
    # Load data
    print("Loading training data...")
    train_df = pd.read_csv('dataset/train.csv')
    test_df = pd.read_csv('dataset/test.csv')
    
    print(f"Training data shape: {train_df.shape}")
    print(f"Test data shape: {test_df.shape}")
    
    # Prepare features for training
    print("Preparing training features...")
    X_train = predictor.prepare_features(train_df, fit_vectorizer=True)
    y_train = train_df['price']
    
    print(f"Feature matrix shape: {X_train.shape}")
    
    # Train models
    model_scores = predictor.train_models(X_train, y_train)
    
    # Make predictions on test set
    print("Making predictions on test set...")
    test_predictions = predictor.predict(test_df)
    
    # Create output file
    output_df = pd.DataFrame({
        'sample_id': test_df['sample_id'],
        'price': test_predictions
    })
    
    # Ensure all prices are positive
    output_df['price'] = np.maximum(output_df['price'], 0.01)
    
    # Save predictions
    output_path = 'dataset/test_out.csv'
    output_df.to_csv(output_path, index=False)
    
    print(f"Predictions saved to {output_path}")
    print(f"Prediction statistics:")
    print(output_df['price'].describe())
    
    return output_df

if __name__ == "__main__":
    main()
