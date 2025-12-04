"""
Organize COCO Dataset for Person Detection
===========================================

This script processes COCO dataset annotations and organizes images
into single_person and multiple_person folders.

Prerequisites:
1. Download COCO validation set: http://images.cocodataset.org/zips/val2017.zip
2. Download annotations: http://images.cocodataset.org/annotations/annotations_trainval2017.zip
3. Extract both to temp_coco/ folder

Output structure:
person dataset improved/
    data/
        train/
            single_person/
            multiple_person/
        val/
            single_person/
            multiple_person/
"""

import json
import shutil
import os
from pathlib import Path
import random

# ============================================================================
# CONFIGURATION
# ============================================================================

COCO_IMAGES_DIR = "temp_coco/val2017"
COCO_ANNOTATIONS = "temp_coco/annotations/instances_val2017.json"

OUTPUT_DIR = "person dataset improved/data"
TRAIN_DIR = os.path.join(OUTPUT_DIR, "train")
VAL_DIR = os.path.join(OUTPUT_DIR, "val")

# Maximum images per class
MAX_SINGLE_PERSON = 2000
MAX_MULTIPLE_PERSON = 2000

# Train/val split ratio
TRAIN_RATIO = 0.8

# ============================================================================
# CREATE DIRECTORIES
# ============================================================================

def create_directories():
    """Create output directory structure"""
    dirs = [
        os.path.join(TRAIN_DIR, "single_person"),
        os.path.join(TRAIN_DIR, "multiple_person"),
        os.path.join(VAL_DIR, "single_person"),
        os.path.join(VAL_DIR, "multiple_person")
    ]
    
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
    
    print("✓ Directories created")

# ============================================================================
# LOAD AND PROCESS COCO ANNOTATIONS
# ============================================================================

def load_coco_annotations():
    """Load COCO annotations and count persons per image"""
    print("Loading COCO annotations...")
    
    with open(COCO_ANNOTATIONS, 'r') as f:
        coco_data = json.load(f)
    
    print(f"✓ Loaded {len(coco_data['images'])} images")
    print(f"✓ Loaded {len(coco_data['annotations'])} annotations")
    
    # Count persons in each image
    person_counts = {}
    for annotation in coco_data['annotations']:
        if annotation['category_id'] == 1:  # Person category ID in COCO
            image_id = annotation['image_id']
            person_counts[image_id] = person_counts.get(image_id, 0) + 1
    
    print(f"✓ Found {len(person_counts)} images with persons")
    
    return coco_data, person_counts

# ============================================================================
# ORGANIZE IMAGES
# ============================================================================

def organize_images(coco_data, person_counts):
    """Organize images into single_person and multiple_person folders"""
    print("\nOrganizing images...")
    
    single_person_images = []
    multiple_person_images = []
    
    # Categorize images
    for image_info in coco_data['images']:
        image_id = image_info['id']
        person_count = person_counts.get(image_id, 0)
        
        if person_count == 0:
            continue  # Skip images without persons
        
        source_path = os.path.join(COCO_IMAGES_DIR, image_info['file_name'])
        
        if not os.path.exists(source_path):
            continue
        
        if person_count == 1:
            single_person_images.append((source_path, image_info['file_name']))
        elif person_count >= 2:
            multiple_person_images.append((source_path, image_info['file_name']))
    
    print(f"Found {len(single_person_images)} single person images")
    print(f"Found {len(multiple_person_images)} multiple person images")
    
    # Limit and shuffle
    random.shuffle(single_person_images)
    random.shuffle(multiple_person_images)
    
    single_person_images = single_person_images[:MAX_SINGLE_PERSON]
    multiple_person_images = multiple_person_images[:MAX_MULTIPLE_PERSON]
    
    print(f"Using {len(single_person_images)} single person images")
    print(f"Using {len(multiple_person_images)} multiple person images")
    
    # Split into train and val
    single_train_count = int(len(single_person_images) * TRAIN_RATIO)
    multiple_train_count = int(len(multiple_person_images) * TRAIN_RATIO)
    
    # Copy single person images
    print("\nCopying single person images...")
    for i, (src, filename) in enumerate(single_person_images):
        if i < single_train_count:
            dest_dir = os.path.join(TRAIN_DIR, "single_person")
        else:
            dest_dir = os.path.join(VAL_DIR, "single_person")
        
        dest_path = os.path.join(dest_dir, filename)
        shutil.copy(src, dest_path)
        
        if (i + 1) % 100 == 0:
            print(f"  Copied {i + 1}/{len(single_person_images)} images")
    
    print(f"✓ Copied {len(single_person_images)} single person images")
    
    # Copy multiple person images
    print("\nCopying multiple person images...")
    for i, (src, filename) in enumerate(multiple_person_images):
        if i < multiple_train_count:
            dest_dir = os.path.join(TRAIN_DIR, "multiple_person")
        else:
            dest_dir = os.path.join(VAL_DIR, "multiple_person")
        
        dest_path = os.path.join(dest_dir, filename)
        shutil.copy(src, dest_path)
        
        if (i + 1) % 100 == 0:
            print(f"  Copied {i + 1}/{len(multiple_person_images)} images")
    
    print(f"✓ Copied {len(multiple_person_images)} multiple person images")
    
    return {
        'single_train': single_train_count,
        'single_val': len(single_person_images) - single_train_count,
        'multiple_train': multiple_train_count,
        'multiple_val': len(multiple_person_images) - multiple_train_count
    }

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function"""
    print("="*80)
    print("COCO PERSON DATASET ORGANIZER")
    print("="*80)
    
    # Check if COCO files exist
    if not os.path.exists(COCO_IMAGES_DIR):
        print(f"\n❌ Error: COCO images directory not found: {COCO_IMAGES_DIR}")
        print("\nPlease download COCO validation set:")
        print("http://images.cocodataset.org/zips/val2017.zip")
        print("\nExtract to: temp_coco/")
        return
    
    if not os.path.exists(COCO_ANNOTATIONS):
        print(f"\n❌ Error: COCO annotations not found: {COCO_ANNOTATIONS}")
        print("\nPlease download COCO annotations:")
        print("http://images.cocodataset.org/annotations/annotations_trainval2017.zip")
        print("\nExtract to: temp_coco/")
        return
    
    # Create directories
    create_directories()
    
    # Load annotations
    coco_data, person_counts = load_coco_annotations()
    
    # Organize images
    stats = organize_images(coco_data, person_counts)
    
    # Print summary
    print("\n" + "="*80)
    print("ORGANIZATION COMPLETE")
    print("="*80)
    print(f"\nDataset location: {OUTPUT_DIR}")
    print(f"\nTraining set:")
    print(f"  Single person: {stats['single_train']} images")
    print(f"  Multiple person: {stats['multiple_train']} images")
    print(f"  Total: {stats['single_train'] + stats['multiple_train']} images")
    print(f"\nValidation set:")
    print(f"  Single person: {stats['single_val']} images")
    print(f"  Multiple person: {stats['multiple_val']} images")
    print(f"  Total: {stats['single_val'] + stats['multiple_val']} images")
    print(f"\nGrand total: {sum(stats.values())} images")
    print("\n" + "="*80)
    print("\nNext step: Run training script")
    print("python train_person_improved.py")
    print("="*80)

if __name__ == "__main__":
    main()
