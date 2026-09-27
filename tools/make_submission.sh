#!/bin/sh
# Build the final post-challenge zip -> dist/submission.zip
#   output/matching_results.tsv
#   output/candidate_pairs.tsv
#   code/business_entity_resolution/   (pipeline + requirements.txt + kaggle notebook)
#   documentation_template.md
#
# usage: tools/make_submission.sh <folder with the two downloaded .tsv files> [documentation.md]
#   e.g. tools/make_submission.sh ~/Downloads documentation_template.md
cd "$(dirname "$0")/.." || exit 1
src=${1:?"give the folder that holds matching_results.tsv and candidate_pairs.tsv"}
doc=${2:-documentation_template.md}
for f in "$src/matching_results.tsv" "$src/candidate_pairs.tsv" "$doc"; do
    [ -f "$f" ] || { echo "missing: $f"; exit 1; }
done

stage=$(mktemp -d)
mkdir -p "$stage/output"
cp "$src/matching_results.tsv" "$src/candidate_pairs.tsv" "$stage/output/"
mkdir -p "$stage/code"
cp -R code "$stage/code/business_entity_resolution"
rm -rf "$stage/code/business_entity_resolution/__pycache__"
cp requirements.txt "$stage/code/business_entity_resolution/"
cp notebooks/kaggle_run.ipynb "$stage/code/business_entity_resolution/"
cp "$doc" "$stage/documentation_template.md"

mkdir -p dist
rm -f dist/submission.zip
(cd "$stage" && zip -rq "$OLDPWD/dist/submission.zip" output code documentation_template.md)
rm -rf "$stage"
echo "created dist/submission.zip ($(du -h dist/submission.zip | cut -f1))"
unzip -l dist/submission.zip | tail -n +4 | awk '{print $4}' | grep -v '^$' | head -30
