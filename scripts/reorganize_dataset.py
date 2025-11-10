import os
import shutil
from pathlib import Path
import random

def reorganize_dataset():
    print("🔄 REORGANIZING DATASET FOR BALANCED TRAINING...")
    
    # Paths
    dataset_path = r"D:\A Image Classification\deepfake_detector\data\Dataset"
    processed_path = r"D:\A Image Classification\deepfake_detector\data\processed"
    
    data_dir = Path(dataset_path)
    processed_dir = Path(processed_path)
    
    # Create processed directory structure
    train_dir = processed_dir / "train"
    val_dir = processed_dir / "val"
    test_dir = processed_dir / "test"
    
    for dir_path in [train_dir, val_dir, test_dir]:
        (dir_path / "real").mkdir(parents=True, exist_ok=True)
        (dir_path / "fake").mkdir(parents=True, exist_ok=True)
    
    print("📁 Created directory structure:")
    print(f"   {train_dir}/")
    print(f"   {val_dir}/")
    print(f"   {test_dir}/")
    
    # Get all test real and fake images
    test_real_dir = data_dir / "Test" / "Real"
    test_fake_dir = data_dir / "Test" / "Fake"
    train_fake_dir = data_dir / "Train" / "Fake"
    
    test_real_images = list(test_real_dir.glob("*.jpg"))
    test_fake_images = list(test_fake_dir.glob("*.jpg"))
    train_fake_images = list(train_fake_dir.glob("*.jpg"))
    
    print(f"\n📊 ORIGINAL DATA COUNTS:")
    print(f"   Test Real: {len(test_real_images)}")
    print(f"   Test Fake: {len(test_fake_images)}")
    print(f"   Train Fake: {len(train_fake_images)}")
    
    # Shuffle and split test real images for training and validation
    random.shuffle(test_real_images)
    random.shuffle(test_fake_images)
    
    # Split ratios: 70% train, 15% val, 15% test
    real_train_count = int(len(test_real_images) * 0.7)
    real_val_count = int(len(test_real_images) * 0.15)
    
    fake_train_count = int(len(train_fake_images) * 0.7)
    fake_val_count = int(len(train_fake_images) * 0.15)
    
    print(f"\n📈 SPLITTING DATA:")
    print(f"   Real images: {real_train_count} train, {real_val_count} val, {len(test_real_images) - real_train_count - real_val_count} test")
    print(f"   Fake images: {fake_train_count} train, {fake_val_count} val, {len(train_fake_images) - fake_train_count - fake_val_count} test")
    
    # Copy real images
    print(f"\n📦 COPYING REAL IMAGES...")
    for i, img_path in enumerate(test_real_images):
        if i < real_train_count:
            dest = train_dir / "real" / img_path.name
        elif i < real_train_count + real_val_count:
            dest = val_dir / "real" / img_path.name
        else:
            dest = test_dir / "real" / img_path.name
        shutil.copy2(img_path, dest)
    
    # Copy fake images from training set
    print(f"📦 COPYING FAKE IMAGES...")
    for i, img_path in enumerate(train_fake_images):
        if i < fake_train_count:
            dest = train_dir / "fake" / img_path.name
        elif i < fake_train_count + fake_val_count:
            dest = val_dir / "fake" / img_path.name
        else:
            dest = test_dir / "fake" / img_path.name
        shutil.copy2(img_path, dest)
    
    # Verify the new structure
    print(f"\n✅ REORGANIZATION COMPLETE!")
    print(f"📊 NEW DATASET STRUCTURE:")
    
    for split_name, split_dir in [("TRAIN", train_dir), ("VALIDATION", val_dir), ("TEST", test_dir)]:
        real_count = len(list((split_dir / "real").glob("*.jpg")))
        fake_count = len(list((split_dir / "fake").glob("*.jpg")))
        total = real_count + fake_count
        balance_ratio = real_count / fake_count if fake_count > 0 else 0
        
        print(f"   {split_name}:")
        print(f"      Real: {real_count}")
        print(f"      Fake: {fake_count}")
        print(f"      Total: {total}")
        print(f"      Balance Ratio: {balance_ratio:.2f}")
    
    total_real = sum(len(list(d.glob("*.jpg"))) for d in [train_dir/"real", val_dir/"real", test_dir/"real"])
    total_fake = sum(len(list(d.glob("*.jpg"))) for d in [train_dir/"fake", val_dir/"fake", test_dir/"fake"])
    
    print(f"\n🎯 FINAL TOTALS:")
    print(f"   Total Real: {total_real}")
    print(f"   Total Fake: {total_fake}")
    print(f"   Grand Total: {total_real + total_fake}")
    
    return str(processed_dir)

if __name__ == "__main__":
    processed_path = reorganize_dataset()
    print(f"\n🎉 DATASET REORGANIZATION COMPLETE!")
    print(f"📁 Processed data location: {processed_path}")
    print(f"\n🚀 Next step: Proceed with data preprocessing and model training!")