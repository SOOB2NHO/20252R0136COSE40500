#!/bin/bash
# Setup script for GitHub repository

cd "/Users/hosubin/Desktop/KU/4학년/Computer Science Colloquium/20252R0136COSE40500"

# Initialize git if not already initialized
if [ ! -d .git ]; then
    git init
fi

# Ensure branch is named main
git branch -M main

# Add remote if not exists
if ! git remote | grep -q origin; then
    git remote add origin https://github.com/SOOB2NHO/20252R0136COSE40500.git
fi

# Add all files
git add .

# Create first commit
git commit -m "first commit"

echo "Setup complete!"
echo "To push to GitHub, run: git push -u origin main"

