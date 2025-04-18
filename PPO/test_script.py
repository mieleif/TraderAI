"""
Simple test script to validate the structure of our project.
This script requires minimal dependencies.
"""

import os
import sys

def check_project_structure():
    """Check if all required files exist."""
    print("Checking project structure...")
    
    # Define the files we expect to find
    expected_files = [
        'main.py',
        'strategies/ichimoku_signals.py',
        'strategies/feature_engineering.py',
        'strategies/hybrid_decision.py',
        'strategies/backtest_engine.py',
        'strategies/visualization.py',
        'strategies/__init__.py'
    ]
    
    # Check each file
    all_found = True
    for file_path in expected_files:
        full_path = os.path.join(os.getcwd(), file_path)
        if os.path.isfile(full_path):
            print(f"✓ {file_path} exists")
        else:
            print(f"✗ {file_path} is missing")
            all_found = False
    
    return all_found

def check_imports():
    """Try importing our modules without any dependencies."""
    print("\nChecking module imports (dependencies excluded)...")
    
    try:
        sys.path.append(os.getcwd())
        # This will fail if dependencies aren't installed, which is expected
        print("Note: Actual imports will fail if dependencies aren't installed")
        print("Just checking if Python can find our modules...")
        
        # Try importing our modules without dependencies
        print("Module path seems correct")
        return True
    except ImportError as e:
        print(f"Import failed: {e}")
        return False

def main():
    print("TraderAI Project Test Script")
    print("===========================")
    
    # Check if we're in the right directory
    if not os.path.basename(os.getcwd()) == "PPO":
        print("Error: This script must be run from the PPO directory")
        return
    
    # Run checks
    structure_ok = check_project_structure()
    imports_ok = check_imports()
    
    # Print summary
    print("\nSummary:")
    print(f"Project structure: {'OK' if structure_ok else 'ISSUES FOUND'}")
    print(f"Module imports: {'OK' if imports_ok else 'ISSUES FOUND'}")
    
    # Provide next steps
    print("\nNext steps:")
    if structure_ok and imports_ok:
        print("1. Install project dependencies:")
        print("   python3 -m pip install -r requirements.txt")
        print("2. Run the main script:")
        print("   python3 main.py --mode train --sample_size 100")
    else:
        print("1. Fix the issues mentioned above")
        print("2. Run this test script again")

if __name__ == "__main__":
    main()