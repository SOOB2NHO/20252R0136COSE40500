#!/bin/bash
# Setup script to create 10 commits

cd "/Users/hosubin/Desktop/KU/4학년/Computer Science Colloquium/20252R0136COSE40500"

# Ensure we're on main branch
git checkout -b main 2>/dev/null || git checkout main

# Commit 3: Add README
git add README.md
git commit -m "Add README.md with project description" || echo "Commit 3 may already exist"

# Commit 4: Improve GPU config comments
git add evaluate.py
git commit -m "Improve GPU configuration comments" || echo "Commit 4 may already exist"

# Commit 5: Add type hints improvements
# (This will be done via file edits)

# Commit 6: Improve error handling
# (This will be done via file edits)

# Commit 7: Add logging improvements
# (This will be done via file edits)

# Commit 8: Code refactoring
# (This will be done via file edits)

# Commit 9: Documentation updates
# (This will be done via file edits)

# Commit 10: Final improvements
# (This will be done via file edits)

echo "Setup complete. Check with: git log --oneline"

