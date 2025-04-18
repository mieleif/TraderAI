"""
Script to update requirements.txt with additional dependencies
"""

import os

# Define the base requirements from original file
base_requirements = [
    "numpy==1.23.5",
    "pandas==1.5.3",
    "tensorflow==2.10.0",
    "gym==0.26.2",
    "scikit-learn==1.2.2",
    "tqdm==4.65.0",
    "matplotlib==3.7.1",
    "ipython==8.12.0"
]

# Additional requirements for our new modules
additional_requirements = [
    "seaborn>=0.12.0",     # For enhanced visualizations
    "mplfinance>=0.12.9b0", # For financial charts
    "scipy>=1.10.0",       # Scientific computing
]

# Combine requirements
all_requirements = base_requirements + additional_requirements

# Write to requirements.txt
with open('requirements.txt', 'w') as f:
    f.write('\n'.join(all_requirements))

print(f"Updated requirements.txt with {len(all_requirements)} dependencies")
print("New dependencies added:")
for req in additional_requirements:
    print(f"  - {req}")