#!/bin/bash
git add run.sh
git add push.sh
git add main.py
git add README.md
git add param/params.json
git add src/metrics.py
git add src/dcg.py
git add requirements.txt
git add .gitignore
git commit -m "$1"
git push
echo "Done."