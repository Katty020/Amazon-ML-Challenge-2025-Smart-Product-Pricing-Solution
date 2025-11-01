"""
Smart Product Pricing Challenge - Solution Runner:- 

This script provides an easy way to run the pricing prediction solution.
It includes options for different solution approaches and handles dependencies.

Usage:
    python run_solution.py [--solution baseline|enhanced|full]
"""

import os
import sys
import argparse
import subprocess

def install_requirements():
    """Install required packages"""
    print("Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✓ Requirements installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to install requirements: {e}")
        return False

def run_baseline():
    """Run the baseline solution"""
    print("Running baseline solution...")
    try:
        import baseline_solution
        result = baseline_solution.main()
        print("✓ Baseline solution completed successfully")
        return result
    except Exception as e:
        print(f"✗ Baseline solution failed: {e}")
        return None

def run_enhanced():
    """Run the enhanced solution"""
    print("Running enhanced solution...")
    try:
        # Check if torch is available
        try:
            import torch
            import torchvision
        except ImportError:
            print("PyTorch not available. Installing...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "torch", "torchvision"])
        
        import enhanced_solution
        result = enhanced_solution.main()
        print("✓ Enhanced solution completed successfully")
        return result
    except Exception as e:
        print(f"✗ Enhanced solution failed: {e}")
        return None

def run_full():
    """Run the full solution"""
    print("Running full solution...")
    try:
        # Install additional dependencies for full solution
        try:
            import torch
            import torchvision
            import nltk
        except ImportError:
            print("Installing additional dependencies...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "torch", "torchvision", "nltk"])
        
        import smart_pricing_solution
        result = smart_pricing_solution.main()
        print("✓ Full solution completed successfully")
        return result
    except Exception as e:
        print(f"✗ Full solution failed: {e}")
        return None

def check_data():
    """Check if required data files exist"""
    required_files = [
        'dataset/train.csv',
        'dataset/test.csv',
        'dataset/sample_test_out.csv'
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print(f"✗ Missing required files: {missing_files}")
        return False
    
    print("✓ All required data files found")
    return True

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Smart Product Pricing Challenge Solution Runner')
    parser.add_argument('--solution', choices=['baseline', 'enhanced', 'full', 'advanced'], 
                       default='baseline', help='Solution approach to run')
    parser.add_argument('--install-deps', action='store_true', 
                       help='Install dependencies before running')
    
    args = parser.parse_args()
    
    print("Smart Product Pricing Challenge - Solution Runner")
    print("=" * 50)
    
    # Check data files
    if not check_data():
        print("Please ensure all required data files are present in the dataset/ directory")
        return
    
    # Install dependencies if requested
    if args.install_deps:
        if not install_requirements():
            print("Failed to install requirements. Please install manually.")
            return
    
    # Run selected solution
    if args.solution == 'baseline':
        result = run_baseline()
    elif args.solution == 'enhanced':
        result = run_enhanced()
    elif args.solution == 'full':
        result = run_full()
    elif args.solution == 'advanced':
        print("Running advanced solution...")
        try:
            # Install additional dependencies for advanced solution
            req_file = os.path.join(os.path.dirname(__file__), 'requirements_advanced.txt')
            if os.path.exists(req_file):
                print("Installing advanced requirements...")
                subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", req_file])

            import advanced_pricing_solution
            result = advanced_pricing_solution.main()
            print("✓ Advanced solution completed successfully")
        except Exception as e:
            print(f"✗ Advanced solution failed: {e}")
            result = None
    
    if result is not None:
        print(f"\n✓ Solution completed successfully!")
        print(f"Output file: dataset/test_out.csv")
        print(f"Number of predictions: {len(result)}")
        
        # Show sample predictions
        print(f"\nSample predictions:")
        print(result.head(10))
    else:
        print("\n✗ Solution failed. Please check the error messages above.")

if __name__ == "__main__":
    main()
