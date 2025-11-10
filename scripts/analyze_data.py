import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import Counter
import cv2

def analyze_dataset():
    print("📊 ANALYZING PROCESSED DATASET DISTRIBUTION...")
    
    # Updated to use processed data
    data_path = r"D:\A Image Classification\deepfake_detector\data\processed"
    
    # Find all image files
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    all_images = []
    
    for ext in image_extensions:
        all_images.extend(Path(data_path).rglob(ext))
    
    print(f"📸 Total images found: {len(all_images)}")
    
    if len(all_images) == 0:
        print("❌ No images found! Please check dataset extraction.")
        return
    
    # Analyze distribution by split and class
    distribution = Counter()
    image_sizes = []
    
    for img_path in all_images[:2000]:  # Sample for performance
        # Get split (train/val/test) and class (real/fake)
        parts = img_path.relative_to(data_path).parts
        if len(parts) >= 2:
            split = parts[0]  # train, val, test
            class_name = parts[1]  # real, fake
            distribution[f"{split}/{class_name}"] += 1
        
        # Get image dimensions
        try:
            img = cv2.imread(str(img_path))
            if img is not None:
                height, width = img.shape[:2]
                image_sizes.append((width, height))
        except Exception as e:
            continue
    
    # Create detailed plots
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Distribution by split and class
    plt.subplot(2, 2, 1)
    if distribution:
        categories, counts = zip(*distribution.most_common())
        bars = plt.bar(categories, counts, color=['skyblue', 'lightcoral', 'lightgreen', 'gold', 'lightpink', 'lightgray'])
        plt.title('Image Distribution by Split and Class')
        plt.xticks(rotation=45, ha='right')
        plt.ylabel('Number of Images')
        
        # Add value labels on bars
        for bar, count in zip(bars, counts):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50, 
                    str(count), ha='center', va='bottom')
    
    # Plot 2: Image size distribution
    plt.subplot(2, 2, 2)
    if image_sizes:
        widths, heights = zip(*image_sizes)
        plt.scatter(widths, heights, alpha=0.6, s=20)
        plt.title('Image Size Distribution')
        plt.xlabel('Width (pixels)')
        plt.ylabel('Height (pixels)')
        plt.grid(True, alpha=0.3)
        
        # Add statistics
        avg_width = np.mean(widths)
        avg_height = np.mean(heights)
        plt.axvline(avg_width, color='red', linestyle='--', alpha=0.8, label=f'Avg Width: {avg_width:.0f}px')
        plt.axhline(avg_height, color='blue', linestyle='--', alpha=0.8, label=f'Avg Height: {avg_height:.0f}px')
        plt.legend()
    
    # Plot 3: Class balance per split
    plt.subplot(2, 2, 3)
    splits = ['train', 'val', 'test']
    real_counts = []
    fake_counts = []
    
    for split in splits:
        real_counts.append(distribution.get(f"{split}/real", 0))
        fake_counts.append(distribution.get(f"{split}/fake", 0))
    
    x = np.arange(len(splits))
    width = 0.35
    
    plt.bar(x - width/2, real_counts, width, label='Real', color='lightblue')
    plt.bar(x + width/2, fake_counts, width, label='Fake', color='lightcoral')
    
    plt.xlabel('Data Split')
    plt.ylabel('Number of Images')
    plt.title('Class Balance Across Splits')
    plt.xticks(x, splits)
    plt.legend()
    
    # Add value labels
    for i, (real, fake) in enumerate(zip(real_counts, fake_counts)):
        plt.text(i - width/2, real + 50, str(real), ha='center')
        plt.text(i + width/2, fake + 50, str(fake), ha='center')
    
    # Plot 4: Overall class distribution
    plt.subplot(2, 2, 4)
    total_real = sum(real_counts)
    total_fake = sum(fake_counts)
    
    plt.pie([total_real, total_fake], labels=['Real', 'Fake'], autopct='%1.1f%%', 
            colors=['lightblue', 'lightcoral'], startangle=90)
    plt.title('Overall Class Distribution')
    
    plt.tight_layout()
    plt.savefig(r"D:\A Image Classification\deepfake_detector\data\processed\dataset_analysis.png", dpi=300, bbox_inches='tight')
    print("✅ Analysis plot saved!")
    
    # Print comprehensive summary
    print("\n📈 DATASET SUMMARY:")
    print(f"Total images: {len(all_images)}")
    print(f"Training set: {sum(real_counts[:1]) + sum(fake_counts[:1])} images")
    print(f"Validation set: {sum(real_counts[1:2]) + sum(fake_counts[1:2])} images")
    print(f"Test set: {sum(real_counts[2:]) + sum(fake_counts[2:])} images")
    
    if image_sizes:
        avg_width = np.mean([w for w, h in image_sizes])
        avg_height = np.mean([h for w, h in image_sizes])
        min_width = min([w for w, h in image_sizes])
        min_height = min([h for w, h in image_sizes])
        max_width = max([w for w, h in image_sizes])
        max_height = max([h for w, h in image_sizes])
        
        print(f"Image size range: {min_width}x{min_height} to {max_width}x{max_height}")
        print(f"Average image size: {avg_width:.0f}x{avg_height:.0f}")
    
    print("\n📂 DETAILED DISTRIBUTION:")
    for category, count in distribution.most_common():
        print(f"   {category}: {count} images")

def visualize_samples():
    print("\n🖼️ VISUALIZING SAMPLE IMAGES...")
    
    data_path = r"D:\A Image Classification\deepfake_detector\data\processed"
    
    # Get samples from each category
    sample_images = []
    categories = [
        ('train/real', 'Train Real'),
        ('train/fake', 'Train Fake'), 
        ('val/real', 'Val Real'),
        ('val/fake', 'Val Fake'),
        ('test/real', 'Test Real'),
        ('test/fake', 'Test Fake')
    ]
    
    for folder, label in categories:
        folder_path = Path(data_path) / folder
        if folder_path.exists():
            images = list(folder_path.glob('*.jpg'))[:1]
            if images:
                sample_images.append((images[0], label))
    
    # Display samples
    if sample_images:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.ravel()
        
        for idx, (img_path, label) in enumerate(sample_images):
            try:
                img = cv2.imread(str(img_path))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                axes[idx].imshow(img)
                axes[idx].set_title(f'{label}\n{img_path.name}\n{img.shape[1]}x{img.shape[0]}')
                axes[idx].axis('off')
                
                print(f"   {label}: {img_path.name} | Size: {img.shape[1]}x{img.shape[0]}")
            except Exception as e:
                print(f"   Error loading {img_path}: {e}")
                axes[idx].text(0.5, 0.5, 'Error loading image', 
                             horizontalalignment='center', verticalalignment='center')
                axes[idx].axis('off')
        
        plt.tight_layout()
        plt.savefig(r"D:\A Image Classification\deepfake_detector\data\processed\sample_images.png", dpi=300, bbox_inches='tight')
        print("✅ Sample images plot saved!")
    else:
        print("❌ No sample images found!")

if __name__ == "__main__":
    analyze_dataset()
    visualize_samples()
    print("\n🎉 DATA ANALYSIS COMPLETE!")
    print("📊 Analysis results saved to: data/processed/")