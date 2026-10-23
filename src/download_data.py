"""
download_data.py
Utility script to download and extract the UCI Student Performance dataset.
Uses standard library urllib and zipfile so it works in any standard Python environment.
"""

import os
import sys
import zipfile
import urllib.request
import shutil

UCI_URL = "https://archive.ics.uci.edu/static/public/320/student+performance.zip"
FALLBACK_GITHUB_RAW = "https://raw.githubusercontent.com/arunkumar-singh/UCI-Student-Performance-Data-Set/master/student-mat.csv"

def download_and_extract(data_dir: str) -> str:
    """
    Downloads student+performance.zip from UCI Machine Learning Repository
    and extracts student-mat.csv into data_dir.
    Returns path to student-mat.csv.
    """
    os.makedirs(data_dir, exist_ok=True)
    target_csv = os.path.join(data_dir, "student-mat.csv")
    
    if os.path.exists(target_csv) and os.path.getsize(target_csv) > 1000:
        print(f"[OK] student-mat.csv already exists at {target_csv} ({os.path.getsize(target_csv)} bytes)")
        return target_csv

    zip_path = os.path.join(data_dir, "student_performance.zip")
    print(f"Downloading dataset from {UCI_URL} ...")
    
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    req = urllib.request.Request(UCI_URL, headers=headers)
    
    download_success = False
    try:
        with urllib.request.urlopen(req, timeout=30) as response, open(zip_path, 'wb') as out_file:
            shutil.copyfileobj(response, out_file)
        print(f"[OK] Downloaded {zip_path} ({os.path.getsize(zip_path)} bytes)")
        
        # Extract student-mat.csv
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Check if student-mat.csv is inside
            file_list = zip_ref.namelist()
            print(f"Files found in archive: {file_list}")
            
            # The archive may contain student.zip or direct csvs
            if "student-mat.csv" in file_list:
                zip_ref.extract("student-mat.csv", path=data_dir)
            elif "student.zip" in file_list:
                zip_ref.extract("student.zip", path=data_dir)
                inner_zip = os.path.join(data_dir, "student.zip")
                with zipfile.ZipFile(inner_zip, 'r') as inner_ref:
                    inner_ref.extract("student-mat.csv", path=data_dir)
                if os.path.exists(inner_zip):
                    os.remove(inner_zip)
            else:
                for fname in file_list:
                    if fname.endswith("student-mat.csv"):
                        zip_ref.extract(fname, path=data_dir)
                        extracted = os.path.join(data_dir, fname)
                        shutil.move(extracted, target_csv)
                        break
        
        # Clean up downloaded zip to keep project clean
        if os.path.exists(zip_path):
            os.remove(zip_path)
            
        if os.path.exists(target_csv) and os.path.getsize(target_csv) > 1000:
            download_success = True
            print(f"[OK] Successfully extracted {target_csv}")
    except Exception as e:
        print(f"[WARN] Direct UCI zip download failed: {e}")
        
    if not download_success:
        print("Attempting fallback direct download from verified public mirror...")
        try:
            req_fallback = urllib.request.Request(FALLBACK_GITHUB_RAW, headers=headers)
            with urllib.request.urlopen(req_fallback, timeout=30) as response, open(target_csv, 'wb') as out_file:
                shutil.copyfileobj(response, out_file)
            print(f"[OK] Fallback download succeeded: {target_csv} ({os.path.getsize(target_csv)} bytes)")
            download_success = True
        except Exception as e2:
            print(f"[ERROR] Fallback download also failed: {e2}")
            raise RuntimeError(f"Unable to download student-mat.csv automatically. Please place student-mat.csv into {data_dir}")

    return target_csv

if __name__ == "__main__":
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
    res = download_and_extract(base_dir)
    print(f"Dataset ready at: {res}")
