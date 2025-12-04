"""
Download Person Detection Dataset from Multiple Sources
========================================================

This script downloads person detection datasets from various platforms:
1. Kaggle - Multiple person detection datasets
2. Roboflow - Person counting datasets
3. COCO Dataset - Person annotations
4. Open Images - Person images

The dataset will have two classes:
- single_person: Images with exactly 1 person
- multiple_person: Images with 2+ persons
"""

import os
import requests
import zipfile
import shutil
from pathlib import Path
import json

# ============================================================================
# DATASET SOURCES
# ============================================================================

DATASET_SOURCES = {
    "kaggle": {
        "name": "Person Detection Dataset",
        "datasets": [
            "kaggle datasets download -d constantinwerner/human-detection-dataset",
            "kaggle datasets download -d ashishjangra27/face-mask-12k-images-dataset",
            "kaggle datasets download -d tapakah68/yolo-crowd-human-detection-dataset"
        ]
    },
    "roboflow": {
        "name": "Person Counting Dataset",
        "url": "https://universe.roboflow.com/person-detection/person-counting-dataset",
        "note": "Download manually from Roboflow Universe"
    },
    "coco": {
        "name": "COCO Person Subset",
        "url": "http://images.cocodataset.org/zips/val2017.zip",
        "annotations": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
    }
}

# ============================================================================
# CONFIGURATION
# ============================================================================

OUTPUT_DIR = "person dataset improved"
TRAIN_DIR = os.path.join(OUTPUT_DIR, "data", "train")
VAL_DIR = os.path.join(OUTPUT_DIR, "data", "val")

SINGLE_PERSON_DIR_TRAIN = os.path.join(TRAIN_DIR, "single_person")
MULTIPLE_PERSON_DIR_TRAIN = os.path.join(TRAIN_DIR, "multiple_person")
SINGLE_PERSON_DIR_VAL = os.path.join(VAL_DIR, "single_person")
MULTIPLE_PERSON_DIR_VAL = os.path.join(VAL_DIR, "multiple_person")

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def create_directories():
    """Create directory structure for dataset"""
    print("Creating directory structure...")
    os.makedirs(SINGLE_PERSON_DIR_TRAIN, exist_ok=True)
    os.makedirs(MULTIPLE_PERSON_DIR_TRAIN, exist_ok=True)
    os.makedirs(SINGLE_PERSON_DIR_VAL, exist_ok=True)
    os.makedirs(MULTIPLE_PERSON_DIR_VAL, exist_ok=True)
    print("✓ Directories created")

def download_file(url, destination):
    """Download file from URL with progress"""
    print(f"Downloading from {url}...")
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(destination, 'wb') as f:
        downloaded = 0
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    percent = (downloaded / total_size) * 100
                    print(f"\rProgress: {percent:.1f}%", end='')
    print("\n✓ Download complete")

def extract_zip(zip_path, extract_to):
    """Extract zip file"""
    print(f"Extracting {zip_path}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print("✓ Extraction complete")

# ============================================================================
# DATASET DOWNLOAD METHODS
# ============================================================================

def download_kaggle_datasets():
    """
    Download datasets from Kaggle
    
    Prerequisites:
    1. Install kaggle: pip install kaggle
    2. Setup Kaggle API credentials:
       - Go to https://www.kaggle.com/account
       - Click "Create New API Token"
       - Place kaggle.json in ~/.kaggle/ (Linux/Mac) or C:\\Users\\<username>\\.kaggle\\ (Windows)
    """
    print("\n" + "="*80)
    print("DOWNLOADING FROM KAGGLE")
    print("="*80)
    
    try:
        import kaggle
        print("✓ Kaggle API found")
    except ImportError:
        print("❌ Kaggle API not installed")
        print("Install with: pip install kaggle")
        print("Setup credentials: https://www.kaggle.com/docs/api")
        return False
    
    # Download human detection dataset
    print("\nDownloading Human Detection Dataset...")
    try:
        os.system("kaggle datasets download -d constantinwerner/human-detection-dataset -p temp_kaggle")
        extract_zip("temp_kaggle/human-detection-dataset.zip", "temp_kaggle/human_detection")
        print("✓ Human Detection Dataset downloaded")
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def download_coco_subset():
    """
    Download COCO validation set and filter person images
    """
    print("\n" + "="*80)
    print("DOWNLOADING COCO DATASET SUBSET")
    print("="*80)
    
    # Download validation images (smaller than train set)
    coco_images_url = "http://images.cocodataset.org/zips/val2017.zip"
    coco_annotations_url = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
    
    print("\nNote: COCO dataset is large (~1GB for validation set)")
    print("This will take several minutes to download...")
    
    response = input("Continue? (y/n): ")
    if response.lower() != 'y':
        print("Skipped COCO download")
        return False
    
    try:
        # Download images
        download_file(coco_images_url, "temp_coco/val2017.zip")
        extract_zip("temp_coco/val2017.zip", "temp_coco")
        
        # Download annotations
        download_file(coco_annotations_url, "temp_coco/annotations.zip")
        extract_zip("temp_coco/annotations.zip", "temp_coco")
        
        print("✓ COCO dataset downloaded")
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def download_roboflow_dataset():
    """
    Instructions for downloading from Roboflow
    """
    print("\n" + "="*80)
    print("ROBOFLOW DATASET")
    print("="*80)
    
    print("\nRoboflow has many person detection datasets.")
    print("To download:")
    print("1. Visit: https://universe.roboflow.com/")
    print("2. Search for: 'person detection' or 'person counting'")
    print("3. Choose a dataset (e.g., 'Person Detection' by Roboflow)")
    print("4. Click 'Download Dataset'")
    print("5. Select format: 'Folder Structure' or 'COCO'")
    print("6. Download and extract to 'temp_roboflow' folder")
    print("\nRecommended datasets:")
    print("- https://universe.roboflow.com/roboflow-100/people-detection-o6t89")
    print("- https://universe.roboflow.com/person-detection/person-counting-dataset")
    
    return False

# ============================================================================
# DATASET ORGANIZATION
# ============================================================================

def organize_dataset_from_coco():
    """
    Organize COCO images into single_person and multiple_person folders
    """
    print("\n" + "="*80)
    print("ORGANIZING COCO DATASET")
    print("="*80)
    
    try:
        import json
        from PIL import Image
        
        # Load COCO annotations
        with open("temp_coco/annotations/instances_val2017.json", 'r') as f:
            coco_data = json.load(f)
        
        # Count persons in each image
        image_person_count = {}
        for annotation in coco_data['annotations']:
            if annotation['category_id'] == 1:  # Person category
                image_id = annotation['image_id']
                image_person_count[image_id] = image_person_count.get(image_id, 0) + 1
        
        # Organize images
        single_count = 0
        multiple_count = 0
        
        for image_info in coco_data['images']:
            image_id = image_info['id']
            person_count = image_person_count.get(image_id, 0)
            
            if person_count == 0:
                continue  # Skip images without persons
            
            source_path = f"temp_coco/val2017/{image_info['file_name']}"
            
            if not os.path.exists(source_path):
                continue
            
            # Determine destination based on person count
            if person_count == 1:
                if single_count < 1000:  # Limit to 1000 images
                    dest_dir = SINGLE_PERSON_DIR_TRAIN if single_count < 800 else SINGLE_PERSON_DIR_VAL
                    shutil.copy(source_path, os.path.join(dest_dir, image_info['file_name']))
                    single_count += 1
            else:  # 2+ persons
                if multiple_count < 1000:  # Limit to 1000 images
                    dest_dir = MULTIPLE_PERSON_DIR_TRAIN if multiple_count < 800 else MULTIPLE_PERSON_DIR_VAL
                    shutil.copy(source_path, os.path.join(dest_dir, image_info['file_name']))
                    multiple_count += 1
            
            if single_count >= 1000 and multiple_count >= 1000:
                break
        
        print(f"✓ Organized {single_count} single person images")
        print(f"✓ Organized {multiple_count} multiple person images")
        return True
        
    except Exception as e:
        print(f"❌ Error organizing COCO dataset: {e}")
        return False

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function"""
    print("="*80)
    print("PERSON DETECTION DATASET DOWNLOADER")
    print("="*80)
    
    # Create directories
    create_directories()
    
    print("\nAvailable dataset sources:")
    print("1. Kaggle (requires API setup)")
    print("2. COCO Dataset (large download ~1GB)")
    print("3. Roboflow (manual download)")
    print("4. Use existing images (manual organization)")
    
    choice = input("\nSelect option (1-4): ")
    
    if choice == "1":
        download_kaggle_datasets()
    elif choice == "2":
        if download_coco_subset():
            organize_dataset_from_coco()
    elif choice == "3":
        download_roboflow_dataset()
    elif choice == "4":
        print("\nManual organization:")
        print(f"1. Place single person images in: {SINGLE_PERSON_DIR_TRAIN}")
        print(f"2. Place multiple person images in: {MULTIPLE_PERSON_DIR_TRAIN}")
        print(f"3. Place validation images in: {SINGLE_PERSON_DIR_VAL} and {MULTIPLE_PERSON_DIR_VAL}")
        print("4. Run train_person_model_improved.py")
    
    print("\n" + "="*80)
    print("DATASET DOWNLOAD COMPLETE")
    print("="*80)
    print(f"\nDataset location: {OUTPUT_DIR}")
    print("\nNext steps:")
    print("1. Verify images in the folders")
    print("2. Run: python train_person_model_improved.py")

if __name__ == "__main__":
    main()
