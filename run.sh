#!/bin/bash
git add run.sh
# git add A1Q1_AMATH_495.ipynb
git add main.py
git add README.md
git add param/params.json
git add requirements.txt
git add .gitignore
git commit -m "$1"
git push
echo "Done."