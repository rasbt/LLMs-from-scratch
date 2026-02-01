#!/usr/bin/env bash
set -euo pipefail

APP_NAME="buffer-app"
INSTALL_DIR="/usr/local/share/${APP_NAME}"
BIN_PATH="/usr/local/bin/${APP_NAME}"

if [[ "$EUID" -ne 0 ]]; then
  echo "Please run as root (e.g., sudo ./install.sh)" >&2
  exit 1
fi

# Remove any prior install
rm -rf "$INSTALL_DIR"
rm -f "$BIN_PATH"

# Create install dir
mkdir -p "$INSTALL_DIR"

# Copy app files
cp -R "$(dirname "$0")/public" "$INSTALL_DIR/"
cp "$(dirname "$0")/server.js" "$INSTALL_DIR/"
cp "$(dirname "$0")/README.md" "$INSTALL_DIR/"

# Install launcher
install -m 0755 "$(dirname "$0")/bin/buffer-app" "$BIN_PATH"

cat <<EOF
Installed ${APP_NAME}.
Run: ${APP_NAME}
Then open: http://localhost:3000
Set PORT env var to change port, e.g. PORT=8080 ${APP_NAME}
EOF
