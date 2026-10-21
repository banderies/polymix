#!/bin/sh
# Build dist/Polymix.app and publish it as the GitHub release v<version in Cargo.toml>:
#   scripts/release.sh     -> creates the release, or replaces its zip if it already exists
# The zip has the same name in every release, so .../releases/latest/download/<zip> (the
# README's download link) always gives the newest build. Bump `version` in Cargo.toml for a
# new release. Needs the GitHub CLI (gh), and HEAD committed and pushed.
set -e
cd "$(dirname "$0")/.."
VERSION=$(grep -m1 '^version' Cargo.toml | cut -d'"' -f2)
TAG="v$VERSION"
REPO=$(git remote get-url origin | sed -E 's#^(https://github\.com/|git@github\.com:)##; s#\.git$##')
ZIP="dist/Polymix-macos-$(uname -m).zip"

if ! git diff --quiet HEAD; then
  echo "Uncommitted changes: commit them first, so the release matches its tag." >&2
  exit 1
fi
git fetch -q origin
if ! git merge-base --is-ancestor HEAD origin/main; then
  echo "HEAD is not on origin/main: push it first." >&2
  exit 1
fi
# With several gh accounts logged in, use the repository owner's.
if [ -z "$GH_TOKEN" ] && TOKEN=$(gh auth token -u "${REPO%%/*}" 2>/dev/null); then
  export GH_TOKEN="$TOKEN"
fi

scripts/make_app.sh
rm -f "$ZIP"
ditto -c -k --keepParent dist/Polymix.app "$ZIP"

if gh release view "$TAG" -R "$REPO" >/dev/null 2>&1; then
  gh release upload "$TAG" "$ZIP" -R "$REPO" --clobber
else
  gh release create "$TAG" "$ZIP" -R "$REPO" --target "$(git rev-parse HEAD)" --title "Polymix $VERSION" --notes-file - <<NOTES
**Download:** \`$(basename "$ZIP")\` below. Unzip it and move \`Polymix.app\` to Applications
(macOS 11 or later, Apple Silicon).

The app is not signed with an Apple Developer ID, so macOS blocks it the first time: open it
once, then go to System Settings → Privacy & Security and click **Open Anyway**. Or, in
Terminal: \`xattr -dr com.apple.quarantine /Applications/Polymix.app\`

**Not a medical device.** Polymix is research software, not for diagnosis, treatment planning
or any other clinical decision.
NOTES
fi
echo "Published $TAG: https://github.com/$REPO/releases/tag/$TAG"
