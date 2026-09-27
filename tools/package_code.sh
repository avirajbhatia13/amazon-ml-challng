#!/bin/sh
# Zip code/ for upload as a Kaggle dataset -> dist/er-code.zip
cd "$(dirname "$0")/.." || exit 1
mkdir -p dist
rm -f dist/er-code.zip
zip -rq dist/er-code.zip code -x "code/__pycache__/*"
echo "created dist/er-code.zip ($(du -h dist/er-code.zip | cut -f1))"
