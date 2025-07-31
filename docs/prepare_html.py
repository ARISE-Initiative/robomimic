"""
This script renames the "_static" and "_sources" folders to "static" and "sources" 
respectively in the HTML build directory, and updates all referencesin the HTML 
files to point to the new folder name.
"""

import os
import shutil

HTML_DIR = "_build/html"
OLD_DIRS = ["_static", "_sources"]
NEW_DIRS = ["static", "sources"]

for OLD_DIR, NEW_DIR in zip(OLD_DIRS, NEW_DIRS):
    old_dir_path = os.path.join(HTML_DIR, OLD_DIR)
    new_dir_path = os.path.join(HTML_DIR, NEW_DIR)

    # 1. Rename the folder if it exists
    if os.path.isdir(old_dir_path):
        if os.path.exists(new_dir_path):
            shutil.rmtree(new_dir_path)
        os.rename(old_dir_path, new_dir_path)

    # 2. Replace all references in files under HTML_DIR
    for root, _, files in os.walk(HTML_DIR):
        for fname in files:
            fpath = os.path.join(root, fname)
            assert (
                "_build/html" in fpath
            ), "This script should only be run in the HTML build directory."
            # Only process html files
            if fname.endswith(".html"):
                with open(fpath, "r", encoding="utf-8") as f:
                    content = f.read()
                if OLD_DIR in content:
                    content = content.replace(OLD_DIR, NEW_DIR)
                    with open(fpath, "w", encoding="utf-8") as f:
                        f.write(content)
