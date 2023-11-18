#!/bin/bash
git add run.sh
git add push.sh
git add main.py
git add README.md
git add param/
git add src/
git add requirements.txt
git add .gitignore
git commit -m "$1"
git push
echo "Done."