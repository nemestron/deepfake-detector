import os
from pathlib import Path
from PIL import Image
import shutil

def clean_dataset():
    """Remove corrupted images from dataset"""
    print("🧹 CLEANING DATASET - REMOVING CORRUPTED IMAGES...")
    
    data_path = Path(r"D:\A Image Classification\deepfake_detector\data\processed")
    corrupted_count = 0
    
    for split in ['train', 'val', 'test']:
        split_path = data_path / split
        for class_name in ['real', 'fake']:
            class_path = split_path / class_name
            if class_path.exists():
                images = list(class_path.glob('*.jpg'))
                print(f"🔍 Checking {split}/{class_name}: {len(images)} images")
                
                for img_path in images:
                    try:
                        # Try to open and verify image
                        with Image.open(img_path) as img:
                            img.verify()
                    except (IOError, SyntaxError) as e:
                        print(f"❌ Removing corrupted image: {img_path}")
                        os.remove(img_path)
                        corrupted_count += 1
    
    print(f"✅ Removed {corrupted_count} corrupted images")
    return corrupted_count

def count_clean_images():
    """Count images after cleaning"""
    print("\n📊 COUNTING CLEAN DATASET...")
    
    data_path = Path(r"D:\A Image Classification\deepfake_detector\data\processed")
    
    for split in ['train', 'val', 'test']:
        split_path = data_path / split
        real_count = len(list((split_path / 'real').glob('*.jpg')))
        fake_count = len(list((split_path / 'fake').glob('*.jpg')))
        total = real_count + fake_count
        print(f"   {split.upper()}: {total} images ({real_count} real, {fake_count} fake)")

if __name__ == "__main__":
    corrupted = clean_dataset()
    count_clean_images()
    
    if corrupted > 0:
        print(f"\n🎉 DATASET CLEANED! Removed {corrupted} corrupted images")
        print("🚀 Now run: python scripts/train_model_fixed.py")
    else:
        print("\n✅ No corrupted images found! Dataset is clean.")