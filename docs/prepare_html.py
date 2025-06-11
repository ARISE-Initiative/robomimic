"""
This script renames the "_static" folder to "static" in the HTML build directory
and updates all references in the HTML files to point to the new folder name.
"""

import os
import shutil

HTML_DIR = "_build/html"
OLD_STATIC = "_static"
NEW_STATIC = "static"

old_static_path = os.path.join(HTML_DIR, OLD_STATIC)
new_static_path = os.path.join(HTML_DIR, NEW_STATIC)

# 1. Rename the folder if it exists
if os.path.isdir(old_static_path):
    if os.path.exists(new_static_path):
        shutil.rmtree(new_static_path)
    os.rename(old_static_path, new_static_path)

# 2. Replace all references in files under HTML_DIR
for root, _, files in os.walk(HTML_DIR):
    for fname in files:
        fpath = os.path.join(root, fname)
        # Only process text files (e.g., .html, .js, .css)
        # if fname.endswith(('.html', '.js', '.css', '.json', '.txt')):
        if fname.endswith('.html'):
            with open(fpath, 'r', encoding='utf-8') as f:
                content = f.read()
            if OLD_STATIC in content:
                content = content.replace(OLD_STATIC, NEW_STATIC)
                with open(fpath, 'w', encoding='utf-8') as f:
                    f.write(content)
