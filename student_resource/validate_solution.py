"""
Solution Validation Script:-

This script validates the solution output and checks for common issues.
"""

import pandas as pd
import numpy as np
import os

def validate_output():
    """Validate the solution output"""
    print("Validating Smart Product Pricing Solution Output")
    print("=" * 50)
    
    # Check if output file exists
    output_file = 'dataset/test_out.csv'
    if not os.path.exists(output_file):
        print("❌ Output file not found: dataset/test_out.csv")
        return False
    
    # Check sample file exists
    sample_file = 'dataset/sample_test_out.csv'
    if not os.path.exists(sample_file):
        print("❌ Sample file not found: dataset/sample_test_out.csv")
        return False
    
    # Load files
    try:
        output_df = pd.read_csv(output_file)
        sample_df = pd.read_csv(sample_file)
        test_df = pd.read_csv('dataset/test.csv')
    except Exception as e:
        print(f"❌ Error loading files: {e}")
        return False
    
    print("✅ Files loaded successfully")
    
    # Check output format
    expected_columns = ['sample_id', 'price']
    if list(output_df.columns) != expected_columns:
        print(f"❌ Incorrect columns. Expected: {expected_columns}, Got: {list(output_df.columns)}")
        return False
    
    print("✅ Column format correct")
    
    # Check number of predictions
    expected_rows = len(test_df)
    actual_rows = len(output_df)
    
    if actual_rows != expected_rows:
        print(f"❌ Incorrect number of predictions. Expected: {expected_rows}, Got: {actual_rows}")
        return False
    
    print(f"✅ Correct number of predictions: {actual_rows}")
    
    # Check sample_id matching
    test_ids = set(test_df['sample_id'])
    output_ids = set(output_df['sample_id'])
    
    if test_ids != output_ids:
        missing_ids = test_ids - output_ids
        extra_ids = output_ids - test_ids
        print(f"❌ Sample ID mismatch:")
        if missing_ids:
            print(f"   Missing IDs: {len(missing_ids)}")
        if extra_ids:
            print(f"   Extra IDs: {len(extra_ids)}")
        return False
    
    print("✅ Sample IDs match perfectly")
    
    # Check price values
    prices = output_df['price']
    
    # Check for negative prices
    negative_prices = (prices < 0).sum()
    if negative_prices > 0:
        print(f"❌ Found {negative_prices} negative prices")
        return False
    
    print("✅ All prices are non-negative")
    
    # Check for zero prices
    zero_prices = (prices == 0).sum()
    if zero_prices > 0:
        print(f"⚠️  Found {zero_prices} zero prices (may cause issues)")
    
    # Check for missing values
    missing_prices = prices.isna().sum()
    if missing_prices > 0:
        print(f"❌ Found {missing_prices} missing price values")
        return False
    
    print("✅ No missing price values")
    
    # Check price statistics
    print(f"\n📊 Price Statistics:")
    print(f"   Min: ${prices.min():.2f}")
    print(f"   Max: ${prices.max():.2f}")
    print(f"   Mean: ${prices.mean():.2f}")
    print(f"   Median: ${prices.median():.2f}")
    print(f"   Std: ${prices.std():.2f}")
    
    # Check for reasonable price range
    if prices.max() > 10000:
        print(f"⚠️  Very high maximum price: ${prices.max():.2f}")
    
    if prices.min() < 0.01:
        print(f"⚠️  Very low minimum price: ${prices.min():.2f}")
    
    # Check data types
    if output_df['sample_id'].dtype != 'int64':
        print(f"⚠️  Sample ID should be integer, got: {output_df['sample_id'].dtype}")
    
    if not pd.api.types.is_numeric_dtype(prices):
        print(f"❌ Price column should be numeric, got: {prices.dtype}")
        return False
    
    print("✅ Data types correct")
    
    print(f"\n🎉 Validation completed successfully!")
    print(f"✅ Output file is ready for submission")
    
    return True

def compare_with_sample():
    """Compare output format with sample"""
    print("\n📋 Comparing with sample format:")
    
    try:
        output_df = pd.read_csv('dataset/test_out.csv')
        sample_df = pd.read_csv('dataset/sample_test_out.csv')
        
        print(f"Sample format: {sample_df.columns.tolist()}")
        print(f"Our format: {output_df.columns.tolist()}")
        
        if list(output_df.columns) == list(sample_df.columns):
            print("✅ Format matches sample perfectly")
        else:
            print("❌ Format doesn't match sample")
            
        print(f"\nSample price range: ${sample_df['price'].min():.2f} - ${sample_df['price'].max():.2f}")
        print(f"Our price range: ${output_df['price'].min():.2f} - ${output_df['price'].max():.2f}")
        
    except Exception as e:
        print(f"❌ Error comparing with sample: {e}")

def main():
    """Main validation function"""
    success = validate_output()
    compare_with_sample()
    
    if success:
        print(f"\n🚀 Ready to submit!")
        print(f"📁 Upload: dataset/test_out.csv")
    else:
        print(f"\n❌ Validation failed. Please fix issues before submitting.")
    
    return success

if __name__ == "__main__":
    main()
