#!/usr/bin/env bash
# Mac side: build a release tarball from a git ref of this repository (DESIGN.md §11.2). Read-only for the repo.
#   bash deploy/release.sh [--ref REF] [--out DIR] [--repo DIR]   (--repo: another checkout; tests)
# Writes DIR/r1mc3-<version>-<git7>.tar.gz and its .sha256; the tarball holds one directory r1mc3-<id>/ with the
# tracked files at REF (git archive: no venv, no build, no untracked files) and a MANIFEST (version, git, created, the
# sha256 of every file) that install.sh checks after unpacking on the backpack.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
repo="$(cd "$here/.." && pwd)"
ref="HEAD"; out="$repo/out/releases"
while [ $# -gt 0 ]; do
  case "$1" in
    --ref) ref="$2"; shift 2 ;;
    --out) out="$2"; shift 2 ;;
    --repo) repo="$(cd "$2" && pwd)"; shift 2 ;;
    *) echo "unknown option $1" >&2; exit 2 ;;
  esac
done
sha256() { if command -v sha256sum >/dev/null 2>&1; then sha256sum "$@"; else shasum -a 256 "$@"; fi; }
git -C "$repo" rev-parse --verify --quiet "$ref^{commit}" >/dev/null || { echo "not a commit: $ref" >&2; exit 2; }
sha="$(git -C "$repo" rev-parse --short=7 "$ref^{commit}")"
full="$(git -C "$repo" rev-parse "$ref^{commit}")"
version="$(git -C "$repo" show "$ref:pyproject.toml" | sed -n 's/^version = "\(.*\)"/\1/p' | sed -n 1p)"  # not head: pipefail
[ -n "$version" ] || { echo "no version in pyproject.toml at $ref" >&2; exit 2; }
id="$version-$sha"
tmp="$(mktemp -d)"; trap 'rm -rf "$tmp"' EXIT
mkdir -p "$tmp/r1mc3-$id" "$out"
git -C "$repo" archive --format=tar "$ref" | tar -x -C "$tmp/r1mc3-$id"
(
  cd "$tmp/r1mc3-$id"
  {
    echo "id=$id"
    echo "version=$version"
    echo "git=$full"
    echo "created_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "files:"
    find . -type f ! -name MANIFEST | LC_ALL=C sort | while IFS= read -r f; do sha256 "$f"; done
  } > MANIFEST
)
# HW1 (fix #6): the Mac's bsdtar stored every file's com.apple.provenance xattr (731 "unknown extended header keyword"
# warnings from GNU tar on the backpack) and uid 501 / the Mac user as the owner. No xattrs / ACLs / Mac metadata, the
# owner root (0); install.sh also unpacks with --no-same-owner. GNU tar (a Linux build host) takes other words.
if tar --version 2>/dev/null | grep -q 'GNU tar'; then
  tar_meta=(--no-xattrs --no-acls --owner=0 --group=0 --numeric-owner)
else
  tar_meta=(--no-xattrs --no-acls --no-mac-metadata --uid 0 --gid 0 --uname root --gname root)
fi
COPYFILE_DISABLE=1 tar "${tar_meta[@]}" -czf "$out/r1mc3-$id.tar.gz" -C "$tmp" "r1mc3-$id"
(cd "$out" && sha256 "r1mc3-$id.tar.gz" > "r1mc3-$id.tar.gz.sha256")
echo "$out/r1mc3-$id.tar.gz"
