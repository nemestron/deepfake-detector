import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json
import matplotlib.pyplot as plt

print("🚀 STARTING DATA PREPROCESSING...")

# Dataset paths
data_path = r"D:\A Image Classification\deepfake_detector\data\processed"
output_path = r"D:\A Image Classification\deepfake_detector\data\preprocessed_data"

# Create output directory
Path(output_path).mkdir(exist_ok=True)

def count_images():
    """Count all images in the dataset"""
    print("📊 COUNTING IMAGES...")
    
    counts = {}
    total_images = 0
    
    for split in ['train', 'val', 'test']:
        split_path = Path(data_path) / split
        counts[split] = {}
        
        for class_name in ['real', 'fake']:
            class_path = split_path / class_name
            if class_path.exists():
                image_files = list(class_path.glob('*.jpg'))
                count = len(image_files)
                counts[split][class_name] = count
                total_images += count
                print(f"   {split}/{class_name}: {count} images")
            else:
                counts[split][class_name] = 0
                print(f"   ❌ {split}/{class_name}: folder not found")
    
    print(f"🎯 TOTAL IMAGES: {total_images}")
    return counts, total_images

def create_visualization(counts):
    """Create dataset visualization"""
    print("📈 CREATING VISUALIZATION...")
    
    plt.figure(figsize=(15, 5))
    
    # Plot 1: Overall distribution
    plt.subplot(1, 3, 1)
    total_real = sum([counts[split]['real'] for split in counts])
    total_fake = sum([counts[split]['fake'] for split in counts])
    
    plt.pie([total_real, total_fake], 
            labels=['Real', 'Fake'], 
            autopct='%1.1f%%',
            colors=['lightblue', 'lightcoral'])
    plt.title('Overall Distribution')
    
    # Plot 2: Split distribution
    plt.subplot(1, 3, 2)
    splits = list(counts.keys())
    real_counts = [counts[split]['real'] for split in splits]
    fake_counts = [counts[split]['fake'] for split in splits]
    
    x = np.arange(len(splits))
    width = 0.35
    
    plt.bar(x - width/2, real_counts, width, label='Real', color='lightblue')
    plt.bar(x + width/2, fake_counts, width, label='Fake', color='lightcoral')
    plt.xlabel('Split')
    plt.ylabel('Count')
    plt.title('Distribution by Split')
    plt.xticks(x, [s.upper() for s in splits])
    plt.legend()
    
    # Plot 3: Total per split
    plt.subplot(1, 3, 3)
    totals = [counts[split]['real'] + counts[split]['fake'] for split in splits]
    plt.bar([s.upper() for s in splits], totals, color='lightgreen')
    plt.xlabel('Split')
    plt.ylabel('Total Images')
    plt.title('Total Images per Split')
    
    # Add value labels
    for i, total in enumerate(totals):
        plt.text(i, total + 100, str(total), ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(Path(output_path) / 'dataset_analysis.png', dpi=300, bbox_inches='tight')
    print("✅ Visualization saved")

def save_dataset_summary(counts, total_images):
    """Save dataset summary to JSON"""
    print("💾 SAVING DATASET SUMMARY...")
    
    summary = {
        'total_images': total_images,
        'total_real': sum([counts[split]['real'] for split in counts]),
        'total_fake': sum([counts[split]['fake'] for split in counts]),
        'train_images': counts['train']['real'] + counts['train']['fake'],
        'val_images': counts['val']['real'] + counts['val']['fake'],
        'test_images': counts['test']['real'] + counts['test']['fake'],
        'train_real': counts['train']['real'],
        'train_fake': counts['train']['fake'],
        'val_real': counts['val']['real'],
        'val_fake': counts['val']['fake'],
        'test_real': counts['test']['real'],
        'test_fake': counts['test']['fake'],
        'image_size': '256x256',  # From our earlier analysis
        'status': 'READY_FOR_TRAINING'
    }
    
    with open(Path(output_path) / 'dataset_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    return summary

def main():
    """Main preprocessing function"""
    print(f"📁 Source: {data_path}")
    print(f"💾 Output: {output_path}")
    
    # Count images
    counts, total_images = count_images()
    
    # Create visualization
    create_visualization(counts)
    
    # Save summary
    summary = save_dataset_summary(counts, total_images)
    
    # Print final summary
    print(f"\n🎉 PREPROCESSING COMPLETED!")
    print(f"📊 DATASET SUMMARY:")
    print(f"   Total Images: {summary['total_images']}")
    print(f"   Real: {summary['total_real']} ({summary['total_real']/summary['total_images']*100:.1f}%)")
    print(f"   Fake: {summary['total_fake']} ({summary['total_fake']/summary['total_images']*100:.1f}%)")
    print(f"   Train: {summary['train_images']} images")
    print(f"   Validation: {summary['val_images']} images")
    print(f"   Test: {summary['test_images']} images")
    
    print(f"\n🚀 NEXT STEPS:")
    print("1. Dataset is ready for training!")
    print("2. Next: Create and train the deepfake detection model")
    print("3. Run: python scripts/train_model.py")

if __name__ == "__main__":
    main()