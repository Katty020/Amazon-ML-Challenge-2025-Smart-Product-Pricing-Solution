"""
Advanced Smart Product Pricing Solution
======================================

This solution addresses the key limitations in the current approach:
1. Enhanced quantity and unit extraction with better parsing
2. Advanced text feature engineering with domain-specific patterns
3. Improved image processing with better sampling and feature extraction
4. Advanced ensemble methods with hyperparameter optimization
5. Better handling of outliers and data preprocessing
"""

import os
import re
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor, VotingRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet, HuberRegressor
from sklearn.preprocessing import StandardScaler, RobustScaler, PowerTransformer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.feature_selection import SelectKBest, f_regression, RFE
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
import joblib

# Advanced ML
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor

# Image Processing
from PIL import Image
import requests
from io import BytesIO
import torch
import torchvision.transforms as transforms
from torchvision import models
import cv2

# Text Processing
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import string

from tqdm import tqdm
import pickle

class AdvancedPricingPredictor:
    """
    Advanced product price predictor with sophisticated feature engineering
    """
    
    def __init__(self):
        self.text_vectorizer = None
        self.count_vectorizer = None
        self.models = {}
        self.scaler = RobustScaler()  # More robust to outliers
        self.feature_selector = None
        self.image_model = None
        self.price_transformer = None
        
        # Download NLTK data
        try:
            nltk.download('stopwords', quiet=True)
            nltk.download('punkt', quiet=True)
            nltk.download('averaged_perceptron_tagger', quiet=True)
        except:
            pass
        
        self.stop_words = set(stopwords.words('english'))
        
    def extract_advanced_quantity_features(self, catalog_content):
        """
        Enhanced quantity extraction with better pattern matching
        """
        if pd.isna(catalog_content):
            return self._get_empty_quantity_features()
        
        text = str(catalog_content)
        features = {}
        
        # Enhanced quantity patterns
        quantity_patterns = [
            # Pack quantities
            r'(\d+(?:\.\d+)?)\s*(pack|packs|count|pieces|items|units|bottles?|cans?|bags?|boxes?)\s*(?:of\s*)?(\d+(?:\.\d+)?)?',
            # Weight/volume quantities
            r'(\d+(?:\.\d+)?)\s*(oz|ounce|ounces|lb|pound|pounds|kg|kilogram|grams?|g|ml|milliliter|liter|l|gallon|gallons|quart|quarts|pint|pints)',
            # Dimension quantities
            r'(\d+(?:\.\d+)?)\s*(inch|inches|in|cm|centimeter|mm|millimeter|ft|feet|foot)',
            # Sheet/roll quantities
            r'(\d+(?:\.\d+)?)\s*(sheet|sheets|roll|rolls|pad|pads|tablet|tablets)',
            # Special patterns
            r'(\d+(?:\.\d+)?)\s*(fl\s*oz|fluid\s*ounce|fluid\s*ounces)',
        ]
        
        # Extract all quantities
        all_quantities = []
        quantity_types = []
        
        for pattern in quantity_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                try:
                    quantity = float(match.group(1))
                    unit = match.group(2).lower() if match.group(2) else 'unknown'
                    
                    # Handle pack quantities
                    if match.group(3) and 'pack' in unit:
                        pack_size = float(match.group(3))
                        all_quantities.append(quantity * pack_size)
                        quantity_types.append('pack_total')
                    else:
                        all_quantities.append(quantity)
                        quantity_types.append(unit)
                except:
                    pass
        
        # Value: X pattern (most reliable)
        value_matches = re.findall(r'Value:\s*(\d+(?:\.\d+)?)', text)
        if value_matches:
            value_quantity = float(value_matches[0])
            all_quantities.append(value_quantity)
            quantity_types.append('value')
        
        # Process quantities
        if all_quantities:
            features['max_quantity'] = max(all_quantities)
            features['min_quantity'] = min(all_quantities)
            features['avg_quantity'] = np.mean(all_quantities)
            features['quantity_count'] = len(all_quantities)
            features['has_quantity'] = 1
            
            # Quantity type features
            features['has_pack'] = 1 if any('pack' in t for t in quantity_types) else 0
            features['has_weight'] = 1 if any(t in ['oz', 'ounce', 'lb', 'pound', 'kg', 'gram'] for t in quantity_types) else 0
            features['has_volume'] = 1 if any(t in ['ml', 'liter', 'gallon', 'quart'] for t in quantity_types) else 0
            features['has_dimension'] = 1 if any(t in ['inch', 'cm', 'mm'] for t in quantity_types) else 0
            features['has_value'] = 1 if 'value' in quantity_types else 0
            
            # Quantity statistics
            features['quantity_std'] = np.std(all_quantities)
            features['quantity_range'] = features['max_quantity'] - features['min_quantity']
            
        else:
            features.update(self._get_empty_quantity_features())
        
        return features
    
    def _get_empty_quantity_features(self):
        """Return empty quantity features"""
        return {
            'max_quantity': 0, 'min_quantity': 0, 'avg_quantity': 0,
            'quantity_count': 0, 'has_quantity': 0, 'has_pack': 0,
            'has_weight': 0, 'has_volume': 0, 'has_dimension': 0,
            'has_value': 0, 'quantity_std': 0, 'quantity_range': 0
        }
    
    def extract_advanced_text_features(self, catalog_content):
        """
        Comprehensive text feature extraction with domain-specific patterns
        """
        if pd.isna(catalog_content):
            return self._get_empty_text_features()
        
        text = str(catalog_content).lower()
        
        features = {}
        
        # Basic text statistics
        features['text_length'] = len(text)
        features['word_count'] = len(text.split())
        features['char_count'] = len(text.replace(' ', ''))
        features['avg_word_length'] = features['char_count'] / max(features['word_count'], 1)
        features['sentence_count'] = len(sent_tokenize(text))
        
        # Advanced text complexity
        words = text.split()
        unique_words = set(words)
        features['unique_word_ratio'] = len(unique_words) / max(len(words), 1)
        features['avg_sentence_length'] = len(words) / max(features['sentence_count'], 1)
        
        # Brand detection (expanded and more sophisticated)
        premium_brands = [
            'apple', 'nike', 'adidas', 'sony', 'samsung', 'canon', 'nikon', 'dell', 'hp', 'lenovo',
            'microsoft', 'google', 'amazon', 'tesla', 'bmw', 'mercedes', 'audi', 'lexus', 'porsche',
            'rolex', 'omega', 'louis vuitton', 'gucci', 'chanel', 'hermes', 'prada', 'versace',
            'bose', 'jbl', 'beats', 'philips', 'lg', 'panasonic', 'sharp', 'toshiba', 'intel', 'amd',
            'nvidia', 'geforce', 'corsair', 'logitech', 'razer', 'steelseries', 'hyperx'
        ]
        
        mid_tier_brands = [
            'dell', 'hp', 'lenovo', 'asus', 'acer', 'msi', 'gigabyte', 'evga', 'corsair',
            'logitech', 'razer', 'steelseries', 'hyperx', 'kingston', 'crucial', 'sandisk',
            'western digital', 'seagate', 'toshiba', 'panasonic'
        ]
        
        budget_brands = [
            'generic', 'store brand', 'off brand', 'no name', 'unbranded'
        ]
        
        features['has_premium_brand'] = 1 if any(brand in text for brand in premium_brands) else 0
        features['has_mid_tier_brand'] = 1 if any(brand in text for brand in mid_tier_brands) else 0
        features['has_budget_brand'] = 1 if any(brand in text for brand in budget_brands) else 0
        features['brand_count'] = sum(1 for brand in premium_brands + mid_tier_brands if brand in text)
        
        # Enhanced product category detection
        categories = {
            'electronics': ['phone', 'laptop', 'computer', 'tablet', 'camera', 'headphone', 'speaker', 
                           'monitor', 'keyboard', 'mouse', 'printer', 'scanner', 'cable', 'charger',
                           'battery', 'memory', 'storage', 'processor', 'graphics', 'motherboard'],
            'food': ['food', 'snack', 'candy', 'chocolate', 'coffee', 'tea', 'sauce', 'spice', 'cereal', 
                    'crackers', 'nuts', 'dried', 'organic', 'natural', 'gluten-free', 'vegan', 'keto'],
            'clothing': ['shirt', 'pants', 'dress', 'shoe', 'jacket', 'hat', 'clothing', 'jeans', 
                        'sweater', 'hoodie', 'shorts', 'skirt', 'socks', 'underwear', 'bra', 'lingerie'],
            'home': ['furniture', 'bed', 'chair', 'table', 'lamp', 'decor', 'kitchen', 'bathroom', 
                    'living', 'dining', 'bedroom', 'mattress', 'pillow', 'blanket', 'curtain'],
            'beauty': ['cosmetic', 'makeup', 'skincare', 'shampoo', 'soap', 'perfume', 'lotion', 
                      'cream', 'serum', 'mask', 'nail', 'lipstick', 'foundation', 'concealer'],
            'sports': ['fitness', 'gym', 'exercise', 'sport', 'ball', 'racket', 'equipment', 'yoga', 
                      'running', 'basketball', 'football', 'tennis', 'golf', 'cycling', 'swimming'],
            'automotive': ['car', 'truck', 'vehicle', 'tire', 'oil', 'filter', 'battery', 'brake', 
                          'engine', 'transmission', 'spark plug', 'air filter', 'wiper blade'],
            'tools': ['tool', 'drill', 'saw', 'hammer', 'screwdriver', 'wrench', 'pliers', 'knife', 
                     'scissors', 'tape measure', 'level', 'screw', 'bolt', 'nail'],
            'health': ['vitamin', 'supplement', 'medicine', 'health', 'medical', 'fitness', 'wellness',
                      'protein', 'multivitamin', 'omega', 'probiotic', 'herbal', 'natural remedy'],
            'books': ['book', 'novel', 'textbook', 'manual', 'guide', 'dictionary', 'encyclopedia',
                     'magazine', 'journal', 'ebook', 'kindle', 'hardcover', 'paperback']
        }
        
        for category, keywords in categories.items():
            features[f'category_{category}'] = 1 if any(keyword in text for keyword in keywords) else 0
        
        # Quality and pricing indicators (enhanced)
        luxury_words = ['luxury', 'premium', 'professional', 'commercial', 'industrial', 'high-end', 
                       'deluxe', 'exclusive', 'limited edition', 'designer', 'artisan', 'handcrafted']
        budget_words = ['budget', 'economy', 'affordable', 'cheap', 'value', 'discount', 'sale', 
                       'clearance', 'wholesale', 'bulk', 'generic', 'store brand']
        organic_words = ['organic', 'natural', 'pure', 'fresh', 'authentic', 'genuine', 'real', 
                        'unprocessed', 'chemical-free', 'non-gmo', 'gluten-free', 'vegan', 'kosher']
        
        features['luxury_score'] = sum(1 for word in luxury_words if word in text)
        features['budget_score'] = sum(1 for word in budget_words if word in text)
        features['organic_score'] = sum(1 for word in organic_words if word in text)
        
        # Technical specifications (enhanced)
        tech_words = ['wireless', 'bluetooth', 'usb', 'hdmi', 'wifi', '4k', 'hd', 'retina', 'oled', 
                     'led', 'lcd', 'ips', 'refresh rate', 'resolution', 'megapixel', 'processor',
                     'memory', 'storage', 'ssd', 'hdd', 'ram', 'graphics', 'gpu', 'cpu']
        features['tech_score'] = sum(1 for word in tech_words if word in text)
        
        # Material indicators
        material_words = ['steel', 'aluminum', 'plastic', 'wood', 'leather', 'fabric', 'cotton', 
                         'wool', 'silk', 'metal', 'glass', 'ceramic', 'rubber', 'synthetic']
        features['material_score'] = sum(1 for word in material_words if word in text)
        
        # Formatting and presentation features
        features['has_bullet_points'] = text.count('•') + text.count('*') + text.count('-')
        features['has_numbers'] = 1 if re.search(r'\d', text) else 0
        features['has_emoji'] = 1 if any(ord(char) > 127 for char in text) else 0
        features['has_parentheses'] = text.count('(') + text.count(')')
        features['has_brackets'] = text.count('[') + text.count(']')
        features['has_capitals'] = sum(1 for c in text if c.isupper())
        features['capital_ratio'] = features['has_capitals'] / max(len(text), 1)
        
        # Product description quality indicators
        features['has_description'] = 1 if 'description:' in text else 0
        features['has_bullet_point'] = 1 if 'bullet point' in text else 0
        features['has_features'] = 1 if 'features:' in text else 0
        features['has_specifications'] = 1 if 'specifications:' in text else 0
        
        # Price-related keywords (enhanced)
        price_indicators = ['free shipping', 'fast delivery', 'warranty', 'guarantee', 'return policy',
                           'money back', 'satisfaction guaranteed', 'customer service', 'support']
        features['price_indicator_score'] = sum(1 for indicator in price_indicators if indicator in text)
        
        return features
    
    def _get_empty_text_features(self):
        """Return empty text features"""
        base_features = {
            'text_length': 0, 'word_count': 0, 'char_count': 0, 'avg_word_length': 0,
            'sentence_count': 0, 'unique_word_ratio': 0, 'avg_sentence_length': 0,
            'has_premium_brand': 0, 'has_mid_tier_brand': 0, 'has_budget_brand': 0,
            'brand_count': 0, 'luxury_score': 0, 'budget_score': 0, 'organic_score': 0,
            'tech_score': 0, 'material_score': 0, 'has_bullet_points': 0,
            'has_numbers': 0, 'has_emoji': 0, 'has_parentheses': 0, 'has_brackets': 0,
            'has_capitals': 0, 'capital_ratio': 0, 'has_description': 0,
            'has_bullet_point': 0, 'has_features': 0, 'has_specifications': 0,
            'price_indicator_score': 0
        }
        
        categories = ['electronics', 'food', 'clothing', 'home', 'beauty', 'sports', 
                     'automotive', 'tools', 'health', 'books']
        category_features = {f'category_{cat}': 0 for cat in categories}
        
        return {**base_features, **category_features}
    
    def extract_enhanced_image_features(self, image_link, max_images=500):
        """
        Enhanced image feature extraction with better sampling
        """
        if pd.isna(image_link):
            return np.zeros(2048)  # Larger feature vector
        
        try:
            response = requests.get(image_link, timeout=15)
            if response.status_code == 200:
                image = Image.open(BytesIO(response.content))
                
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                
                # Initialize image model if needed
                if self.image_model is None:
                    self.image_model = models.resnet152(pretrained=True)
                    self.image_model = torch.nn.Sequential(*list(self.image_model.children())[:-1])
                    self.image_model.eval()
                
                # Enhanced transforms
                transform = transforms.Compose([
                    transforms.Resize((256, 256)),
                    transforms.CenterCrop((224, 224)),
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
        
        return np.zeros(2048)  # Return zero vector on error
    
    def prepare_advanced_features(self, df, fit_vectorizers=True, use_images=True, max_images=500):
        """
        Prepare all features with advanced preprocessing
        """
        print("Extracting advanced text features...")
        
        # Extract manual features
        manual_features = []
        quantity_features = []
        
        for _, row in tqdm(df.iterrows(), total=len(df)):
            text_feats = self.extract_advanced_text_features(row['catalog_content'])
            quantity_feats = self.extract_advanced_quantity_features(row['catalog_content'])
            manual_features.append(text_feats)
            quantity_features.append(quantity_feats)
        
        manual_df = pd.DataFrame(manual_features)
        quantity_df = pd.DataFrame(quantity_features)
        
        # Advanced text vectorization
        print("Extracting advanced TF-IDF features...")
        if fit_vectorizers:
            self.text_vectorizer = TfidfVectorizer(
                max_features=2500,  # Increased
                stop_words='english',
                ngram_range=(1, 4),  # Extended to 4-grams
                min_df=2,
                max_df=0.95,
                sublinear_tf=True,
                use_idf=True
            )
            self.count_vectorizer = CountVectorizer(
                max_features=1000,
                stop_words='english',
                ngram_range=(1, 2),
                min_df=3,
                max_df=0.9
            )
            
            tfidf_matrix = self.text_vectorizer.fit_transform(df['catalog_content'].fillna(''))
            count_matrix = self.count_vectorizer.fit_transform(df['catalog_content'].fillna(''))
        else:
            tfidf_matrix = self.text_vectorizer.transform(df['catalog_content'].fillna(''))
            count_matrix = self.count_vectorizer.transform(df['catalog_content'].fillna(''))
        
        tfidf_df = pd.DataFrame(
            tfidf_matrix.toarray(),
            columns=[f'tfidf_{i}' for i in range(tfidf_matrix.shape[1])]
        )
        count_df = pd.DataFrame(
            count_matrix.toarray(),
            columns=[f'count_{i}' for i in range(count_matrix.shape[1])]
        )
        
        # Enhanced image features
        image_df = pd.DataFrame()
        if use_images and 'image_link' in df.columns:
            print("Extracting enhanced image features...")
            image_features = np.zeros((len(df), 2048))
            
            # Better sampling strategy
            sample_size = min(max_images, len(df))
            if sample_size < len(df):
                # Stratified sampling by price (if available)
                if 'price' in df.columns:
                    price_bins = pd.cut(df['price'], bins=5, labels=False)
                    sample_indices = []
                    for bin_id in range(5):
                        bin_indices = df[price_bins == bin_id].index
                        if len(bin_indices) > 0:
                            n_sample = max(1, sample_size // 5)
                            sampled = np.random.choice(bin_indices, 
                                                     min(n_sample, len(bin_indices)), 
                                                     replace=False)
                            sample_indices.extend(sampled)
                else:
                    sample_indices = np.random.choice(len(df), sample_size, replace=False)
            else:
                sample_indices = list(range(len(df)))
            
            for idx in tqdm(sample_indices, desc="Processing images"):
                image_features[idx] = self.extract_enhanced_image_features(df.iloc[idx]['image_link'])
            
            # Fill remaining with smart interpolation
            if len(sample_indices) > 0:
                processed_features = image_features[sample_indices]
                # Use k-means clustering to find representative features
                from sklearn.cluster import KMeans
                if len(processed_features) > 10:
                    kmeans = KMeans(n_clusters=min(10, len(processed_features)//2), random_state=42)
                    cluster_labels = kmeans.fit_predict(processed_features)
                    cluster_centers = kmeans.cluster_centers_
                    
                    # Fill unprocessed images with nearest cluster center
                    for i in range(len(df)):
                        if i not in sample_indices:
                            # Use text similarity to find best cluster
                            text_similarity = tfidf_matrix[i].dot(tfidf_matrix[sample_indices].T).toarray().flatten()
                            if len(text_similarity) > 0:
                                best_idx = np.argmax(text_similarity)
                                best_cluster = cluster_labels[best_idx]
                                image_features[i] = cluster_centers[best_cluster]
                            else:
                                image_features[i] = np.mean(processed_features, axis=0)
                else:
                    avg_features = np.mean(processed_features, axis=0)
                    for i in range(len(df)):
                        if i not in sample_indices:
                            image_features[i] = avg_features
            
            image_df = pd.DataFrame(image_features, columns=[f'img_{i}' for i in range(2048)])
        
        # Combine all features
        feature_dfs = [manual_df, quantity_df, tfidf_df, count_df]
        if not image_df.empty:
            feature_dfs.append(image_df)
        
        all_features = pd.concat(feature_dfs, axis=1)
        
        print(f"Total features: {all_features.shape[1]}")
        return all_features
    
    def train_advanced_models(self, X, y):
        """
        Train advanced ensemble models with hyperparameter optimization
        """
        print("Training advanced ensemble models...")
        
        # Transform target variable to handle skewness
        self.price_transformer = PowerTransformer(method='yeo-johnson', standardize=False)
        y_transformed = self.price_transformer.fit_transform(y.values.reshape(-1, 1)).flatten()
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(X, y_transformed, test_size=0.2, random_state=42)
        
        # Advanced feature selection
        print("Selecting best features...")
        self.feature_selector = SelectKBest(f_regression, k=min(1000, X.shape[1]))
        X_train_selected = self.feature_selector.fit_transform(X_train, y_train)
        X_val_selected = self.feature_selector.transform(X_val)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train_selected)
        X_val_scaled = self.scaler.transform(X_val_selected)
        
        # Define advanced models
        models_config = {
            'xgboost': xgb.XGBRegressor(
                n_estimators=500, max_depth=8, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1
            ),
            'lightgbm': lgb.LGBMRegressor(
                n_estimators=500, max_depth=8, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1,
                verbose=-1
            ),
            'catboost': CatBoostRegressor(
                iterations=500, depth=8, learning_rate=0.05,
                random_seed=42, verbose=False
            ),
            'random_forest': RandomForestRegressor(
                n_estimators=300, max_depth=20, min_samples_split=3,
                min_samples_leaf=2, random_state=42, n_jobs=-1
            ),
            'gradient_boosting': GradientBoostingRegressor(
                n_estimators=300, max_depth=8, learning_rate=0.05,
                subsample=0.8, random_state=42
            ),
            'extra_trees': ExtraTreesRegressor(
                n_estimators=300, max_depth=20, min_samples_split=3,
                min_samples_leaf=2, random_state=42, n_jobs=-1
            ),
            'ridge': Ridge(alpha=10.0),
            'elastic_net': ElasticNet(alpha=0.1, l1_ratio=0.5),
            'huber': HuberRegressor(epsilon=1.35, max_iter=1000)
        }
        
        # Train models
        model_scores = {}
        for name, model in models_config.items():
            print(f"Training {name}...")
            
            if name in ['ridge', 'elastic_net', 'huber']:
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_val_scaled)
            else:
                model.fit(X_train_selected, y_train)
                y_pred = model.predict(X_val_selected)
            
            # Calculate SMAPE on original scale
            y_pred_original = self.price_transformer.inverse_transform(y_pred.reshape(-1, 1)).flatten()
            y_val_original = self.price_transformer.inverse_transform(y_val.reshape(-1, 1)).flatten()
            
            smape = self.calculate_smape(y_val_original, y_pred_original)
            model_scores[name] = smape
            
            print(f"{name} SMAPE: {smape:.4f}")
            
            # Save model
            self.models[name] = model
        
        # Create voting regressor with best models
        best_models = sorted(model_scores.items(), key=lambda x: x[1])[:5]
        voting_estimators = [(name, model) for name, model in self.models.items() 
                           if name in [m[0] for m in best_models]]
        
        self.models['voting'] = VotingRegressor(voting_estimators)
        self.models['voting'].fit(X_train_selected, y_train)
        
        # Evaluate voting regressor
        y_pred_voting = self.models['voting'].predict(X_val_selected)
        y_pred_voting_original = self.price_transformer.inverse_transform(y_pred_voting.reshape(-1, 1)).flatten()
        smape_voting = self.calculate_smape(y_val_original, y_pred_voting_original)
        model_scores['voting'] = smape_voting
        
        print(f"Voting ensemble SMAPE: {smape_voting:.4f}")
        
        # Save components
        joblib.dump(self.models, 'advanced_models.pkl')
        joblib.dump(self.scaler, 'advanced_scaler.pkl')
        joblib.dump(self.feature_selector, 'advanced_feature_selector.pkl')
        joblib.dump(self.price_transformer, 'price_transformer.pkl')
        if self.text_vectorizer:
            joblib.dump(self.text_vectorizer, 'advanced_text_vectorizer.pkl')
        if self.count_vectorizer:
            joblib.dump(self.count_vectorizer, 'advanced_count_vectorizer.pkl')
        
        best_model = min(model_scores, key=model_scores.get)
        print(f"Best model: {best_model} with SMAPE: {model_scores[best_model]:.4f}")
        
        return model_scores
    
    def predict_advanced(self, X):
        """
        Make predictions using advanced ensemble
        """
        if not self.models:
            # Load models
            self.models = joblib.load('advanced_models.pkl')
            self.scaler = joblib.load('advanced_scaler.pkl')
            self.feature_selector = joblib.load('advanced_feature_selector.pkl')
            self.price_transformer = joblib.load('price_transformer.pkl')
        
        # Select features
        X_selected = self.feature_selector.transform(X)
        X_scaled = self.scaler.transform(X_selected)
        
        # Get predictions from voting regressor
        if 'voting' in self.models:
            predictions = self.models['voting'].predict(X_selected)
        else:
            # Fallback to weighted ensemble
            predictions = {}
            weights = {
                'xgboost': 0.25, 'lightgbm': 0.25, 'catboost': 0.2,
                'random_forest': 0.15, 'gradient_boosting': 0.1, 'extra_trees': 0.05
            }
            
            final_predictions = np.zeros(len(X))
            for name, model in self.models.items():
                if name in ['ridge', 'elastic_net', 'huber']:
                    pred = model.predict(X_scaled)
                else:
                    pred = model.predict(X_selected)
                
                weight = weights.get(name, 0.1)
                final_predictions += weight * pred
            
            predictions = final_predictions
        
        # Transform back to original scale
        predictions_original = self.price_transformer.inverse_transform(predictions.reshape(-1, 1)).flatten()
        
        return predictions_original
    
    def calculate_smape(self, actual, predicted):
        """Calculate SMAPE metric"""
        return np.mean(np.abs(actual - predicted) / ((np.abs(actual) + np.abs(predicted)) / 2)) * 100

def main():
    """
    Main pipeline for advanced pricing prediction
    """
    print("Advanced Smart Product Pricing Solution")
    print("=" * 50)
    
    # Initialize predictor
    predictor = AdvancedPricingPredictor()
    
    # Load data
    print("Loading data...")
    train_df = pd.read_csv('dataset/train.csv')
    test_df = pd.read_csv('dataset/test.csv')
    
    print(f"Training data: {train_df.shape}")
    print(f"Test data: {test_df.shape}")
    
    # Prepare features
    print("Preparing advanced features...")
    # Use_images=False initially for speed and reliability; enable later if you have GPU/time
    X_train = predictor.prepare_advanced_features(train_df, fit_vectorizers=True, use_images=False, max_images=0)
    y_train = train_df['price']
    
    # Train models
    scores = predictor.train_advanced_models(X_train, y_train)
    
    # Make predictions
    print("Making predictions on test set...")
    X_test = predictor.prepare_advanced_features(test_df, fit_vectorizers=False, use_images=False, max_images=0)
    predictions = predictor.predict_advanced(X_test)
    
    # Create output
    output_df = pd.DataFrame({
        'sample_id': test_df['sample_id'],
        'price': predictions
    })
    
    # Ensure positive prices and handle outliers
    output_df['price'] = np.maximum(output_df['price'], 0.01)
    output_df['price'] = np.minimum(output_df['price'], 10000)  # Cap extreme outliers
    
    # Save predictions
    # Save to expected filename for validator/portal
    output_path = 'dataset/test_out.csv'
    output_df.to_csv(output_path, index=False)
    
    print(f"\nPredictions saved to {output_path}")
    print(f"Prediction statistics:")
    print(output_df['price'].describe())
    
    return output_df

if __name__ == "__main__":
    main()
