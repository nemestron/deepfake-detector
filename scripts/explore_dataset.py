import os
from pathlib import Path
import matplotlib.pyplot as plt

def explore_dataset_comprehensive():
    print("🔍 COMPREHENSIVE DATASET ANALYSIS...")
    
    dataset_path = r"D:\A Image Classification\deepfake_detector\data\Dataset"
    data_dir = Path(dataset_path)
    
    if not data_dir.exists():
        print("❌ Dataset folder not found!")
        return None
    
    print("✅ Dataset folder found!")
    
    # Detailed structure analysis
    print("\n📁 DETAILED STRUCTURE ANALYSIS:")
    sets = ['Train', 'Test']
    classes = ['Real', 'Fake']
    
    summary = {}
    
    for set_name in sets:
        set_path = data_dir / set_name
        if set_path.exists():
            summary[set_name] = {}
            print(f"\n{set_name}:")
            for class_name in classes:
                class_path = set_path / class_name
                if class_path.exists():
                    image_files = list(class_path.glob('*.*'))
                    image_count = len([f for f in image_files if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']])
                    summary[set_name][class_name] = image_count
                    print(f"   {class_name}: {image_count} images")
                else:
                    print(f"   {class_name}: FOLDER NOT FOUND")
                    summary[set_name][class_name] = 0
    
    # Calculate totals
    total_train = summary.get('Train', {}).get('Real', 0) + summary.get('Train', {}).get('Fake', 0)
    total_test = summary.get('Test', {}).get('Real', 0) + summary.get('Test', {}).get('Fake', 0)
    total_images = total_train + total_test
    
    print(f"\n📊 DATASET SUMMARY:")
    print(f"   Training Set: {total_train} images")
    print(f"   Test Set: {total_test} images")
    print(f"   Total Images: {total_images} images")
    
    # Check for data balance
    print(f"\n⚖️ DATA BALANCE ANALYSIS:")
    if 'Train' in summary:
        train_real = summary['Train'].get('Real', 0)
        train_fake = summary['Train'].get('Fake', 0)
        if train_real > 0 and train_fake > 0:
            balance_ratio = train_real / train_fake
            print(f"   Training set balance (Real:Fake): {train_real}:{train_fake} ({balance_ratio:.2f})")
            if balance_ratio < 0.8 or balance_ratio > 1.2:
                print("   ⚠️  Training set is imbalanced!")
        else:
            print("   ❌ Training set missing one class!")
    
    if 'Test' in summary:
        test_real = summary['Test'].get('Real', 0)
        test_fake = summary['Test'].get('Fake', 0)
        if test_real > 0 and test_fake > 0:
            balance_ratio = test_real / test_fake
            print(f"   Test set balance (Real:Fake): {test_real}:{test_fake} ({balance_ratio:.2f})")
    
    # Check image dimensions (sample a few images)
    print(f"\n🖼️ IMAGE PROPERTIES (Sampling):")
    sample_images = []
    for set_name in sets:
        set_path = data_dir / set_name
        if set_path.exists():
            for class_name in classes:
                class_path = set_path / class_name
                if class_path.exists():
                    images = list(class_path.glob('*.jpg'))[:1]  # Sample 1 image per class
                    sample_images.extend(images)
    
    for img_path in sample_images[:4]:  # Show first 4 samples
        if img_path.exists():
            print(f"   {img_path.relative_to(data_dir)}")
        else:
            print(f"   {img_path} - File not found")
    
    return summary

def suggest_next_steps(summary):
    print(f"\n🎯 RECOMMENDED NEXT STEPS:")
    
    train_real = summary.get('Train', {}).get('Real', 0)
    train_fake = summary.get('Train', {}).get('Fake', 0)
    
    if train_real == 0:
        print("1. 🚨 CRITICAL: Training set has NO REAL images!")
        print("   - We need to find real images for training")
        print("   - Options: Use test set real images, find additional dataset")
    elif train_fake == 0:
        print("1. 🚨 CRITICAL: Training set has NO FAKE images!")
    else:
        print("1. ✅ Dataset structure looks good for training!")
        print("   - Proceed with data preprocessing and model training")
    
    print("2. Create data loaders with proper train/validation split")
    print("3. Implement data augmentation for better generalization")
    print("4. Start with a simple CNN model for baseline performance")

if __name__ == "__main__":
    summary = explore_dataset_comprehensive()
    if summary:
        suggest_next_steps(summary)
        print(f"\n🎉 DATASET ANALYSIS COMPLETE!")
    else:
        print("\n❌ Analysis failed.")