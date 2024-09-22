#!/bin/bash

set -e  # Exit immediately if a command exits with a non-zero status.

# This script downloads a PR from GitHub and saves the modified files to a directory.
# Note jq is required to run this script.
# Usage: ./script.sh <PR_NUMBER> <GITHUB_REPO>

PR_NUMBER=$1
GITHUB_REPO=$2  # Format: owner/repo, e.g., "Clubroom/machine-learning-research"
BASE_BRANCH="main"
BRANCH_NAME="pr-$PR_NUMBER"
REMOTE_BRANCH="pull/$PR_NUMBER/head"

if [ -z "$GITHUB_REPO" ]; then
    echo "Please provide the GitHub repository in the format 'owner/repo'"
    exit 1
fi

echo "Fetching PR $PR_NUMBER..."
git fetch origin $REMOTE_BRANCH:$BRANCH_NAME

echo "Getting PR title..."
if command -v gh &> /dev/null; then
    PR_TITLE=$(gh pr view $PR_NUMBER --repo $GITHUB_REPO --json title --jq .title)
else
    PR_TITLE=$(curl -s -H "Accept: application/vnd.github.v3+json" \
        "https://api.github.com/repos/$GITHUB_REPO/pulls/$PR_NUMBER" | \
        jq -r .title)
fi

if [ -z "$PR_TITLE" ]; then
    echo "Failed to fetch PR title. Using PR number as title."
    PR_TITLE="PR-$PR_NUMBER"
fi
echo "PR Title: $PR_TITLE"

# Sanitize PR title for use in directory name
PR_TITLE_SAFE=$(echo "$PR_TITLE" | tr ' ' '_' | tr -cd '[:alnum:]_-')

# Create a directory to store the files
DIR_NAME="${PR_NUMBER}_${PR_TITLE_SAFE}"
mkdir -p "$DIR_NAME"
echo "Created directory: $DIR_NAME"

echo "Getting list of modified files..."
MODIFIED_FILES=$(git diff --name-only origin/$BASE_BRANCH $BRANCH_NAME)
echo "Modified files:"
echo "$MODIFIED_FILES"

if [ -z "$MODIFIED_FILES" ]; then
    echo "No modified files found."
    exit 1
fi

# Initialize pr_structure.txt
echo "Directory structure of modified files:" > "$DIR_NAME/pr_structure.txt"

for file in $MODIFIED_FILES; do
    echo "Processing file: $file"
    
    # Get the diff for the file
    DIFF_CONTENT=$(git diff "origin/$BASE_BRANCH" "$BRANCH_NAME" -- "$file")
    
    # Check if the file is only created or deleted
    if [[ $DIFF_CONTENT == *"new file mode"* || $DIFF_CONTENT == *"deleted file mode"* ]]; then
        if [[ $(echo "$DIFF_CONTENT" | grep -vE "^(diff|index|---|\+\+\+|@@|new file mode|deleted file mode)") ]]; then
            echo "File $file has content changes. Processing..."
        else
            echo "File $file is only created or deleted. Skipping..."
            continue
        fi
    fi

    # Add file path to pr_structure.txt
    echo "$file" >> "$DIR_NAME/pr_structure.txt"

    # Generate filename incorporating directory structure
    FILEPATH_SAFE=$(echo "$file" | tr '/' '_')
    NEW_FILENAME="${FILEPATH_SAFE}.new"
    DIFF_FILENAME="${FILEPATH_SAFE}.diff"

    echo "Getting full file content..."
    if git show "$BRANCH_NAME:$file" > "$DIR_NAME/$NEW_FILENAME" 2>/dev/null; then
        echo "Saved full file content."
    else
        echo "File $file doesn't exist in PR branch. Skipping full content."
        echo "File deleted in PR" > "$DIR_NAME/$NEW_FILENAME"
    fi

    echo "Saving diff..."
    echo "$DIFF_CONTENT" > "$DIR_NAME/$DIFF_FILENAME"

    echo "Processed $file"
done

echo "Files have been saved in the '$DIR_NAME' directory."
ls -R "$DIR_NAME"