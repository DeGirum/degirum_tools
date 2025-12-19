#!/bin/bash

# Script to install MediaMTX from GitHub releases

# Set MediaMTX version
MTX_VERSION="v1.12.3"

# Determine OS and architecture
OS="$(uname | tr '[:upper:]' '[:lower:]')"  # 'linux' or 'darwin'
ARCH="$(uname -m)"
case "$ARCH" in
  x86_64) ARCH="amd64" ;;
  aarch64 | arm64) ARCH="arm64" ;;
  *) echo "Unsupported architecture: $ARCH" && exit 1 ;;
esac

# Construct download URL
FILENAME="mediamtx_${MTX_VERSION}_${OS}_${ARCH}.tar.gz"
URL="https://github.com/bluenviron/mediamtx/releases/download/${MTX_VERSION}/${FILENAME}"

echo "Downloading $URL"
if ! curl -fsSL -o "$FILENAME" "$URL"; then
  echo "❌ Failed to download MediaMTX from $URL"
  exit 1
fi

TMPDIR="$(mktemp -d)"

# Extract into temp dir
tar -xzf "$FILENAME" -C "$TMPDIR"

# Copy all extracted files to /usr/local/bin
sudo cp -a "$TMPDIR"/* /usr/local/bin/
sudo chmod +x /usr/local/bin/mediamtx

# Clean up
rm -rf "$TMPDIR" "$FILENAME"