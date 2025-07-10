#!/bin/sh

# shellcheck disable=SC3030
exclude_items=(
    "*build*"
    "*.git*"
    "*.idea*"
    "*.vscode*"
    "README.md"
    "LICENSE.txt"
    "VERSION"
    "*.xml"
    "*.png"
    "*.css"
    "*.properties"
    "*.bat"
    "*.sh"
    "*.py"
    "*.mp3"
    "*.h264"
    "*.opus"
    "*.mov"
    "*.js"
    "*.pfx"
    "*.appxManifest"
    "scripts/*"
    "external/*"
)

# shellcheck disable=SC3054
exclude_string=$(IFS=,; echo "${exclude_items[*]}")

code2prompt "$1" --exclude "$exclude_string" --tokens format --no-clipboard --no-ignore --output-file "$2"