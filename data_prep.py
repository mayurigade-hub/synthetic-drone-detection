import os
import zipfile
import shutil
import random
import yaml
import argparse
from pathlib import Path

def extract_zip(zip_path, extract_to):
    """Extracts a zip file to a specified directory."""
    print(f"Extracting {zip_path} to {extract_to}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

def find_images_and_labels(directory):
    """Finds images and labels directories recursively."""
    img_dir, lbl_dir = None, None
    for path in Path(directory).rglob('*'):
        if path.is_dir():
            if path.name == 'images':
                img_dir = path
            elif path.name == 'labels':
                lbl_dir = path
    return img_dir, lbl_dir

def merge_and_split(rural_dir, urban_dir, output_dir, split_ratio=0.8):
    """Merges rural and urban datasets and splits into train/val."""
    output_path = Path(output_dir)
    train_img_dir = output_path / 'images' / 'train'
    val_img_dir = output_path / 'images' / 'val'
    train_lbl_dir = output_path / 'labels' / 'train'
    val_lbl_dir = output_path / 'labels' / 'val'

    # Create directories if they don't exist
    for d in [train_img_dir, val_img_dir, train_lbl_dir, val_lbl_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Collect all image-label pairs from rural and urban
    all_pairs = []
    
    for src_name, src_dir in [("Rural", rural_dir), ("Urban", urban_dir)]:
        if not src_dir:
            print(f"Warning: {src_name} source directory is not set. Skipping.")
            continue
            
        img_dir, lbl_dir = find_images_and_labels(src_dir)
        if not img_dir or not lbl_dir:
            print(f"Error: Could not find 'images' and 'labels' directories in {src_dir}")
            continue

        images = list(img_dir.glob('*'))
        print(f"Found {len(images)} images in {src_name} dataset.")
        
        for img_path in images:
            # Check if corresponding label file exists
            lbl_path = lbl_dir / f"{img_path.stem}.txt"
            if lbl_path.exists():
                all_pairs.append((img_path, lbl_path, src_name))
            else:
                print(f"Warning: Label file missing for {img_path.name}")

    if not all_pairs:
        print("No valid image-label pairs found. Cannot proceed.")
        return False

    # Shuffle for random split
    random.seed(42)
    random.shuffle(all_pairs)

    split_idx = int(len(all_pairs) * split_ratio)
    train_pairs = all_pairs[:split_idx]
    val_pairs = all_pairs[split_idx:]

    print(f"Total pairs found: {len(all_pairs)}")
    print(f"Splitting into {len(train_pairs)} training and {len(val_pairs)} validation samples.")

    # Copy files
    for dataset_type, pairs in [('train', train_pairs), ('val', val_pairs)]:
        target_img_dir = train_img_dir if dataset_type == 'train' else val_img_dir
        target_lbl_dir = train_lbl_dir if dataset_type == 'train' else val_lbl_dir

        for img_path, lbl_path, prefix in pairs:
            # Add prefix to filename to prevent name collisions
            new_name = f"{prefix.lower()}_{img_path.name}"
            new_lbl_name = f"{prefix.lower()}_{lbl_path.name}"

            shutil.copy(img_path, target_img_dir / new_name)
            shutil.copy(lbl_path, target_lbl_dir / new_lbl_name)

    print("Data merging and splitting completed successfully.")
    return True

def create_yaml(output_dir):
    """Generates the data.yaml file required for YOLO training."""
    output_path = Path(output_dir).resolve()
    
    yaml_data = {
        'path': str(output_path),
        'train': 'images/train',
        'val': 'images/val',
        'nc': 3,
        'names': ['drone', 'bird', 'manned_aircraft']
    }

    yaml_file_path = output_path / 'data.yaml'
    with open(yaml_file_path, 'w') as f:
        yaml.dump(yaml_data, f, default_flow_style=False)
    
    print(f"Created data.yaml at {yaml_file_path}")

def main():
    parser = argparse.ArgumentParser(description="Extract, merge, and split Vibe Sim datasets.")
    parser.add_argument('--rural_zip', type=str, default='rural_dataset.zip', help='Path to Rural dataset zip file')
    parser.add_argument('--urban_zip', type=str, default='urban_dataset.zip', help='Path to Urban dataset zip file')
    parser.add_argument('--output_dir', type=str, default='combined_dataset', help='Directory to store the merged dataset')
    parser.add_argument('--verify', action='store_true', help='Only verify dataset structure without extracting or copying')
    args = parser.parse_args()

    output_path = Path(args.output_dir)

    if args.verify:
        print("Verifying merged dataset...")
        train_images = list((output_path / 'images' / 'train').glob('*')) if (output_path / 'images' / 'train').exists() else []
        val_images = list((output_path / 'images' / 'val').glob('*')) if (output_path / 'images' / 'val').exists() else []
        
        print(f"Dataset location: {output_path.resolve()}")
        print(f"Training images count: {len(train_images)}")
        print(f"Validation images count: {len(val_images)}")
        
        yaml_path = output_path / 'data.yaml'
        if yaml_path.exists():
            print(f"data.yaml found at {yaml_path}")
            with open(yaml_path, 'r') as f:
                content = yaml.safe_load(f)
                print("data.yaml content:")
                print(yaml.dump(content, default_flow_style=False))
        else:
            print("Warning: data.yaml is missing!")
        return

    # Temporary directory to extract zip files
    temp_dir = Path('temp_extracted')
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir()

    rural_extracted = None
    urban_extracted = None

    # Handle Rural dataset
    if os.path.exists(args.rural_zip):
        rural_extracted = temp_dir / 'rural'
        extract_zip(args.rural_zip, rural_extracted)
    elif os.path.exists('rural_dataset') and os.path.isdir('rural_dataset'):
        rural_extracted = Path('rural_dataset')
        print("Using existing rural_dataset folder...")
    else:
        print(f"Warning: Rural dataset not found at {args.rural_zip} or folder 'rural_dataset'")

    # Handle Urban dataset
    if os.path.exists(args.urban_zip):
        urban_extracted = temp_dir / 'urban'
        extract_zip(args.urban_zip, urban_extracted)
    elif os.path.exists('urban_dataset') and os.path.isdir('urban_dataset'):
        urban_extracted = Path('urban_dataset')
        print("Using existing urban_dataset folder...")
    else:
        print(f"Warning: Urban dataset not found at {args.urban_zip} or folder 'urban_dataset'")

    if not rural_extracted and not urban_extracted:
        print("Error: Neither Rural nor Urban datasets were found. Please place the zip files in this directory.")
        return

    # Clean existing combined directory if it exists
    if output_path.exists():
        print(f"Cleaning existing output directory {args.output_dir}...")
        shutil.rmtree(output_path)

    # Merge and split
    success = merge_and_split(rural_extracted, urban_extracted, args.output_dir)
    
    # Cleanup temp extraction folder
    if temp_dir.exists():
        shutil.rmtree(temp_dir)

    if success:
        create_yaml(args.output_dir)
        print("\nDataset preparation completed successfully! Ready for training.")

if __name__ == '__main__':
    main()
