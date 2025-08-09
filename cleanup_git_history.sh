#!/bin/bash

# Exit on error
set -e

echo "Starting git history cleanup..."

# Create a backup of the current branch
BACKUP_BRANCH="backup-before-cleanup-$(date +%Y%m%d%H%M%S)"
git checkout -b "$BACKUP_BRANCH"
echo "Created backup branch: $BACKUP_BRANCH"

# Remove large files from history
echo "Removing large files from git history..."
git filter-repo --force --path model/ --invert-paths
git filter-repo --force --path venv/ --invert-paths
git filter-repo --force --path joint-disfluency-detector-and-parser/best_models/ --invert-paths

# Clean up and optimize the repository
echo "Cleaning up and optimizing repository..."
git reflog expire --expire=now --all
git gc --prune=now --aggressive

echo "Git history cleanup complete!"
echo "Please verify the changes and then force push with: git push --force origin main"
