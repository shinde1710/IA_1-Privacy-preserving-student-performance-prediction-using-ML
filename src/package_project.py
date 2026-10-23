"""
package_project.py
------------------
Packages the IA 2 project into a clean zip archive for distribution.
Excludes bytecode, temporary caches, and raw data files as per guidelines.
"""

import os
import zipfile

def package_project(root_dir: str, zip_filename: str):
    root_dir = os.path.abspath(root_dir)
    parent_dir = os.path.dirname(root_dir)
    zip_path = os.path.join(parent_dir, zip_filename)
    
    # Exclude patterns
    exclude_dirs = {"__pycache__", ".pytest_cache", ".ipynb_checkpoints", ".git"}
    exclude_files = {".DS_Store", "student-mat.csv", "student_performance.zip"}
    exclude_exts = {".pyc", ".pyo"}
    
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zip_out:
        for foldername, subfolders, filenames in os.walk(root_dir):
            subfolders[:] = [d for d in subfolders if d not in exclude_dirs]
            
            for filename in filenames:
                ext = os.path.splitext(filename)[1]
                if filename in exclude_files or ext in exclude_exts:
                    continue
                    
                full_path = os.path.join(foldername, filename)
                rel_path = os.path.relpath(full_path, root_dir)
                archive_name = os.path.join("student_performance_privacy", rel_path)
                zip_out.write(full_path, archive_name)
                print(f"Added: {archive_name}")
                
    print(f"\n[OK] Successfully created zip archive at:\n{zip_path} ({os.path.getsize(zip_path)} bytes)")
    return zip_path

if __name__ == "__main__":
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    package_project(base, "student_performance_privacy_IA2.zip")
