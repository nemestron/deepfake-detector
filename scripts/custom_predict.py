import sys
import os
from pathlib import Path

# Add the project root to Python path
project_root = Path(r"D:\A Image Classification\deepfake_detector")
sys.path.append(str(project_root))

from scripts.batch_predict import BatchPredictor

def analyze_custom_images():
    """Analyze images in a custom folder"""
    print("🚀 CUSTOM IMAGE ANALYSIS - STARTING...")
    
    # Initialize predictor
    predictor = BatchPredictor()
    
    # Specify your custom folder path here
    custom_folder = input("📁 Enter the path to your images folder: ").strip().strip('"')
    
    # Or hardcode the path (uncomment the line below)
    # custom_folder = r"C:\Your\Custom\Images\Folder"
    
    if not custom_folder:
        print("❌ No folder path provided!")
        return
    
    custom_folder_path = Path(custom_folder)
    
    if not custom_folder_path.exists():
        print(f"❌ Folder not found: {custom_folder_path}")
        return
    
    # Set output path
    output_csv = project_root / "models" / "predictions" / "custom_results.csv"
    
    print(f"🔍 Analyzing folder: {custom_folder_path}")
    print(f"💾 Output will be saved to: {output_csv}")
    
    # Process the folder
    results = predictor.process_folder(
        folder_path=custom_folder_path,
        output_csv=output_csv,
        max_images=None  # Process all images, or set a limit like 100
    )
    
    if results:
        print(f"\n🎉 ANALYSIS COMPLETED!")
        print(f"   Processed {len(results)} images")
        print(f"   Results saved to: {output_csv}")
    else:
        print("\n❌ No results generated!")

if __name__ == "__main__":
    analyze_custom_images()