#!/bin/bash

# Path to your local Git repository
REPO_PATH="$(pwd)"

echo "Archiving plugin at: $REPO_PATH"

# Path to store the zip file
ARCHIVES_PATH="$REPO_PATH/archives"

# Ensure the archives directory exists
mkdir -p $ARCHIVES_PATH

# Get the current version and commit hash
VERSION=$(awk '/^version:/ && NF==2 {print $2; exit}' fiftyone.yml)

echo "Version: $VERSION"

COMMIT=$(git rev-parse HEAD)

echo "Commit: $COMMIT"

FILENAME="voxelgpt-$VERSION-$COMMIT.zip"
OUTPUT="$ARCHIVES_PATH/$FILENAME"

# Archive the repository using git archive command
git archive --format=zip --output=$OUTPUT HEAD

echo "Plugin successfully archived! Created file:"
echo "$OUTPUT"