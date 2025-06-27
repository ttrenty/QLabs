#!/bin/bash
# This script formats the code and then checks if any files were changed.
# 'set -e' ensures that the script will exit immediately if a command fails.
set -e

echo "🎨 Formatting Mojo code..."
# Run the formatter. Because of 'set -e', if this fails, the script stops.
pixi run format

# Now, check for changes.
if ! git diff --quiet; then
  echo "❌ Error: Code is not formatted. The following files were changed by the formatter:"
  git diff --stat # Show a summary of changes
  echo "\nRun 'pixi run format' locally and commit the changes."
  exit 1
else
  echo "✅ Code formatting is correct."
fi