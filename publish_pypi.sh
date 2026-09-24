#!/usr/bin/env bash
# Script to build and upload executive-comm-coach to PyPI or TestPyPI
set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "$DIR"

echo "=== Building Distribution Artifacts ==="
rm -rf dist/ build/ *.egg-info
./venv/bin/python -m build

echo "=== Verifying Package Metadata with Twine ==="
./venv/bin/twine check dist/*

echo ""
echo "=== Distribution Ready in ./dist/ ==="
ls -lh dist/
echo ""
echo "To publish to TestPyPI:"
echo "  ./venv/bin/twine upload --repository testpypi dist/*"
echo ""
echo "To publish to Production PyPI:"
echo "  ./venv/bin/twine upload dist/*"
echo ""
