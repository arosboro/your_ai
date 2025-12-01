#!/bin/bash
# Release script for creating version tags and GitHub releases
# Usage: ./scripts/release.sh [version]
# Example: ./scripts/release.sh 0.1.0

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if we're in the project root
if [ ! -f "VERSION" ]; then
    echo -e "${RED}Error: Run this script from the project root directory${NC}"
    exit 1
fi

# Get version from argument or VERSION file
if [ -n "$1" ]; then
    VERSION="$1"
else
    VERSION=$(cat VERSION | tr -d '[:space:]')
fi

TAG="v${VERSION}"

echo -e "${YELLOW}Preparing release ${TAG}...${NC}"
echo ""

# Check for uncommitted changes
if [ -n "$(git status --porcelain)" ]; then
    echo -e "${RED}Error: There are uncommitted changes. Please commit or stash them first.${NC}"
    git status --short
    exit 1
fi

# Check we're on main branch
CURRENT_BRANCH=$(git branch --show-current)
if [ "$CURRENT_BRANCH" != "main" ]; then
    echo -e "${YELLOW}Warning: You're on branch '${CURRENT_BRANCH}', not 'main'.${NC}"
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check if tag already exists
if git rev-parse "$TAG" >/dev/null 2>&1; then
    echo -e "${RED}Error: Tag ${TAG} already exists.${NC}"
    echo "To delete it: git tag -d ${TAG} && git push origin :refs/tags/${TAG}"
    exit 1
fi

# Verify VERSION file matches
FILE_VERSION=$(cat VERSION | tr -d '[:space:]')
if [ "$VERSION" != "$FILE_VERSION" ]; then
    echo -e "${YELLOW}Warning: VERSION file contains '${FILE_VERSION}' but releasing '${VERSION}'${NC}"
    read -p "Update VERSION file? (Y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Nn]$ ]]; then
        echo "$VERSION" > VERSION
        git add VERSION
        git commit -m "chore: bump version to ${VERSION}"
    fi
fi

# Show what will be released
echo -e "${GREEN}Release summary:${NC}"
echo "  Version: ${VERSION}"
echo "  Tag: ${TAG}"
echo "  Branch: ${CURRENT_BRANCH}"
echo "  Commit: $(git rev-parse --short HEAD)"
echo ""

# Confirm
read -p "Create and push tag ${TAG}? (y/N) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 1
fi

# Create annotated tag
echo -e "${YELLOW}Creating tag ${TAG}...${NC}"
git tag -a "$TAG" -m "Release ${VERSION}"

# Push tag
echo -e "${YELLOW}Pushing tag to origin...${NC}"
git push origin "$TAG"

echo ""
echo -e "${GREEN}✅ Release ${TAG} created and pushed!${NC}"
echo ""
echo "Next steps:"
echo "  1. Go to https://github.com/arosboro/your_ai/releases"
echo "  2. Click 'Draft a new release'"
echo "  3. Select tag '${TAG}'"
echo "  4. Copy release notes from CHANGELOG.txt"
echo "  5. Publish release"
echo ""
echo "Or create release via GitHub CLI:"
echo "  gh release create ${TAG} --title 'Release ${VERSION}' --notes-file CHANGELOG.txt"

