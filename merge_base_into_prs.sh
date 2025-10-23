#!/bin/bash

# Script to merge base branch into all PRs targeting that branch
# Usage: ./merge_base_into_prs.sh <base-branch-name>

# Don't exit on error - we need to handle merge conflicts
set +e

if [ $# -eq 0 ]; then
    echo "Usage: $0 <base-branch-name>"
    echo "Example: $0 dev-3.x-folding"
    exit 1
fi

BASE_BRANCH="$1"

echo "=========================================="
echo "Merging '$BASE_BRANCH' into all PRs targeting it"
echo "=========================================="
echo ""

# Get current branch to return to later
ORIGINAL_BRANCH=$(git rev-parse --abbrev-ref HEAD)

# Fetch latest changes
echo "Fetching latest changes from origin..."
git fetch origin

# Get list of PRs targeting the base branch
echo ""
echo "Getting list of PRs targeting '$BASE_BRANCH'..."
PRS=$(gh pr list --base "$BASE_BRANCH" --limit 100 --json number,headRefName --jq '.[] | "\(.number):\(.headRefName)"')

if [ -z "$PRS" ]; then
    echo "No PRs found targeting '$BASE_BRANCH'"
    exit 0
fi

# Count PRs
PR_COUNT=$(echo "$PRS" | wc -l | xargs)
echo "Found $PR_COUNT PRs targeting '$BASE_BRANCH'"
echo ""

# Process each PR
PR_NUM=0
while IFS=: read -r pr_number head_branch; do
    PR_NUM=$((PR_NUM + 1))
    echo "=========================================="
    echo "[$PR_NUM/$PR_COUNT] Processing PR #$pr_number: $head_branch"
    echo "=========================================="

    # Checkout the PR branch
    echo "Checking out PR #$pr_number..."
    gh pr checkout "$pr_number"

    # Merge the base branch into it
    echo "Merging '$BASE_BRANCH' into '$head_branch'..."
    git merge "origin/$BASE_BRANCH" --no-edit
    MERGE_EXIT_CODE=$?

    if [ $MERGE_EXIT_CODE -eq 0 ]; then
        echo "✓ Merge successful"

        # Push the changes
        echo "Pushing changes to '$head_branch'..."
        if git push; then
            echo "✓ Successfully pushed changes for PR #$pr_number"
        else
            echo "✗ Failed to push changes for PR #$pr_number"
            echo "  You may need to resolve conflicts manually"
        fi
    else
        echo "✗ Merge conflict detected for PR #$pr_number"
        echo "  Opening new terminal to resolve conflicts..."
        echo ""
        echo "  Instructions for the new terminal:"
        echo "    1. Resolve the conflicts in your editor"
        echo "    2. Run: git add ."
        echo "    3. Run: git commit --no-edit"
        echo "    4. Run: git push"
        echo "    5. Type 'exit' to close the terminal"
        echo ""

        # Open a new terminal in the current directory
        # This works on macOS
        osascript <<EOF
tell application "Terminal"
    activate
    do script "cd \"$(pwd)\" && clear && echo '==========================================' && echo 'MERGE CONFLICT RESOLUTION' && echo 'PR #$pr_number: $head_branch' && echo 'Base branch: $BASE_BRANCH' && echo '==========================================' && echo '' && echo 'Conflicted files:' && git diff --name-only --diff-filter=U && echo '' && echo 'Steps to resolve:' && echo '  1. Edit the conflicted files above' && echo '  2. git add .' && echo '  3. git commit --no-edit' && echo '  4. git push' && echo '  5. exit (to close this terminal)' && echo '' && exec \$SHELL"
end tell
EOF

        # Wait for user to resolve conflicts
        echo ""
        echo "⏸  Paused: Waiting for you to resolve conflicts in the new terminal..."
        echo "   Press Enter when you have resolved conflicts and pushed changes..."
        read -r

        # Check if merge was completed
        if git rev-parse -q --verify MERGE_HEAD > /dev/null 2>&1; then
            echo "⚠  Warning: Merge still in progress. Aborting to continue safely."
            git merge --abort
            echo "   Skipping PR #$pr_number - please resolve manually"
        else
            echo "✓ Merge appears to be completed"
        fi
    fi

    echo ""
done <<< "$PRS"

# Return to original branch
echo "=========================================="
echo "Returning to original branch: $ORIGINAL_BRANCH"
git checkout "$ORIGINAL_BRANCH"

echo ""
echo "=========================================="
echo "Done! Processed $PR_COUNT PRs"
echo "=========================================="
