#!/bin/bash
# Apply the color extraction fix to occt-import-js fork

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FORK_DIR="$(dirname "$SCRIPT_DIR")/../occt-import-js-fork"

if [ ! -d "$FORK_DIR" ]; then
    echo "Error: occt-import-js-fork not found at $FORK_DIR"
    exit 1
fi

# Copy the patched source file
cp "$SCRIPT_DIR/importer-xcaf-patched.cpp" "$FORK_DIR/occt-import-js/src/importer-xcaf.cpp"

echo "Patch applied successfully!"
echo "You can now build occt-import-js with: cd $FORK_DIR && ./tools/build_wasm_mac.sh"
