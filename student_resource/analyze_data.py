import pandas as pd
import re
import numpy as np

# Load training data
train = pd.read_csv('dataset/train.csv')
print("Training data shape:", train.shape)

# Price analysis
print("\nPrice statistics:")
print(train['price'].describe())
print(f"Price range: {train['price'].min()} to {train['price'].max()}")
print(f"Median price: {train['price'].median()}")

# Text analysis
train['text_len'] = train['catalog_content'].str.len()
print(f"\nText length statistics:")
print(train['text_len'].describe())

# Quantity extraction analysis
print("\nAnalyzing quantity patterns...")
quantities = []
units = []
for text in train['catalog_content'].head(5000):
    # Look for Value: X pattern
    value_match = re.search(r'Value: ([\d.]+)', str(text))
    if value_match:
        quantities.append(float(value_match.group(1)))
    
    # Look for Unit: X pattern
    unit_match = re.search(r'Unit: ([^\n]+)', str(text))
    if unit_match:
        units.append(unit_match.group(1).strip())

print(f"Found {len(quantities)} quantities in sample")
if quantities:
    print("Quantity statistics:", pd.Series(quantities).describe())

print(f"\nUnique units found: {set(units[:50])}")

# Brand analysis
brands = ['apple', 'nike', 'adidas', 'sony', 'samsung', 'canon', 'nikon', 'dell', 'hp', 'lenovo']
brand_counts = {}
for brand in brands:
    count = train['catalog_content'].str.lower().str.contains(brand, na=False).sum()
    if count > 0:
        brand_counts[brand] = count

print(f"\nBrand presence in dataset:")
for brand, count in brand_counts.items():
    print(f"{brand}: {count}")

# Category analysis
categories = ['food', 'electronics', 'clothing', 'books', 'toys', 'beauty', 'health', 'home', 'sports']
category_counts = {}
for cat in categories:
    count = train['catalog_content'].str.lower().str.contains(cat, na=False).sum()
    if count > 0:
        category_counts[cat] = count

print(f"\nCategory presence in dataset:")
for cat, count in category_counts.items():
    print(f"{cat}: {count}")

# Price vs quantity correlation
print("\nAnalyzing price-quantity relationship...")
price_quantity_data = []
for _, row in train.head(10000).iterrows():
    value_match = re.search(r'Value: ([\d.]+)', str(row['catalog_content']))
    if value_match:
        quantity = float(value_match.group(1))
        price_quantity_data.append({'price': row['price'], 'quantity': quantity})

if price_quantity_data:
    df_pq = pd.DataFrame(price_quantity_data)
    correlation = df_pq['price'].corr(df_pq['quantity'])
    print(f"Price-Quantity correlation: {correlation:.3f}")
