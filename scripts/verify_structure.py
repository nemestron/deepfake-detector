import os
from pathlib import Path

def verify_structure():
    print("🔍 VERIFYING DATASET STRUCTURE...")
    
    data_path = r"D:\A Image Classification\deepfake_detector\data\processed"
    
    if not os.path.exists(data_path):
        print("❌ Processed data folder not found!")
        return
    
    print("✅ Found processed data folder")
    
    # Check each expected folder
    expected_folders = ['train', 'val', 'test']
    
    for folder in expected_folders:
        folder_path = Path(data_path) / folder
        print(f"\n📁 Checking {folder}:")
        
        if folder_path.exists():
            real_path = folder_path / "real"
            fake_path = folder_path / "fake"
            
            if real_path.exists():
                real_files = list(real_path.glob("*.jpg"))
                print(f"   ✅ real: {len(real_files)} images")
            else:
                print(f"   ❌ real: folder not found")
                
            if fake_path.exists():
                fake_files = list(fake_path.glob("*.jpg"))
                print(f"   ✅ fake: {len(fake_files)} images")
            else:
                print(f"   ❌ fake: folder not found")
        else:
            print(f"   ❌ {folder}: folder not found")
    
    # Total count
    print(f"\n🎯 TOTAL COUNTS:")
    total = 0
    for folder in expected_folders:
        folder_path = Path(data_path) / folder
        if folder_path.exists():
            for class_name in ['real', 'fake']:
                class_path = folder_path / class_name
                if class_path.exists():
                    count = len(list(class_path.glob("*.jpg")))
                    total += count
                    print(f"   {folder}/{class_name}: {count}")
    
    print(f"   GRAND TOTAL: {total}")

if __name__ == "__main__":
    verify_structure()