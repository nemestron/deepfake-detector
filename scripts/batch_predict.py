import tensorflow as tf
import numpy as np
import cv2
import pandas as pd
from pathlib import Path
import json
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import os
import sys

# Add the parent directory to path to import from predict_image
sys.path.append(str(Path(__file__).parent.parent))
from scripts.predict_image import DeepfakePredictor

print("🚀 BATCH PREDICTION SYSTEM - LOADING...")

class BatchPredictor:
    def __init__(self, model_path=None):
        """Initialize batch predictor with model"""
        print("🔧 INITIALIZING BATCH PREDICTOR...")
        self.predictor = DeepfakePredictor(model_path)
        self.results = []
    
    def process_folder(self, folder_path, output_csv=None, max_images=None):
        """Process all images in a folder with progress tracking"""
        folder_path = Path(folder_path)
        
        if not folder_path.exists():
            print(f"❌ Folder not found: {folder_path}")
            return None
        
        # Find all image files
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.JPG', '*.JPEG', '*.PNG', '*.BMP']
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(folder_path.rglob(ext))
        
        # Remove duplicates and limit if specified
        image_files = list(set(image_files))
        
        if max_images:
            image_files = image_files[:max_images]
        
        print(f"📁 Found {len(image_files)} images in {folder_path}")
        
        if len(image_files) == 0:
            print("❌ No images found!")
            return None
        
        # Process images with progress bar
        self.results = []
        successful = 0
        failed = 0
        
        print("🔄 PROCESSING IMAGES...")
        for image_path in tqdm(image_files, desc="Analyzing images"):
            try:
                result = self.predictor.predict_image(image_path)
                if result:
                    result['filename'] = image_path.name
                    result['folder'] = str(image_path.parent)
                    result['file_path'] = str(image_path)
                    self.results.append(result)
                    successful += 1
                else:
                    failed += 1
                    
            except Exception as e:
                print(f"\n❌ Error processing {image_path.name}: {e}")
                failed += 1
                continue
        
        print(f"✅ Successfully processed: {successful} images")
        if failed > 0:
            print(f"❌ Failed to process: {failed} images")
        
        # Save results and generate report
        if output_csv:
            self.save_results_csv(output_csv)
        
        report = self.generate_report()
        
        return self.results
    
    def save_results_csv(self, output_path):
        """Save results to CSV file"""
        if not self.results:
            print("❌ No results to save!")
            return
        
        # Create DataFrame with clean data
        clean_results = []
        for result in self.results:
            clean_result = {
                'filename': result.get('filename', ''),
                'predicted_class': result.get('predicted_class', ''),
                'confidence': result.get('confidence', 0),
                'raw_score': result.get('raw_score', 0),
                'is_real': result.get('is_real', False),
                'file_path': result.get('file_path', '')
            }
            clean_results.append(clean_result)
        
        df = pd.DataFrame(clean_results)
        
        # Ensure directory exists
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        df.to_csv(output_path, index=False)
        print(f"💾 Results saved to CSV: {output_path}")
        
        # Also save a summary CSV
        summary_path = output_path.parent / f"summary_{output_path.name}"
        summary_df = df.groupby('predicted_class').agg({
            'confidence': ['count', 'mean', 'std', 'min', 'max'],
            'raw_score': ['mean', 'std']
        }).round(4)
        summary_df.to_csv(summary_path)
        print(f"💾 Summary saved to: {summary_path}")
    
    def generate_report(self):
        """Generate comprehensive analysis report"""
        if not self.results:
            print("❌ No results to analyze!")
            return None
        
        print("\n" + "="*60)
        print("📊 BATCH PREDICTION REPORT")
        print("="*60)
        
        df = pd.DataFrame(self.results)
        
        # Basic statistics
        total_images = len(df)
        real_count = len(df[df['predicted_class'] == 'REAL'])
        fake_count = len(df[df['predicted_class'] == 'FAKE'])
        real_percentage = (real_count / total_images * 100) if total_images > 0 else 0
        fake_percentage = (fake_count / total_images * 100) if total_images > 0 else 0
        
        print(f"📈 SUMMARY STATISTICS:")
        print(f"   Total Images Processed: {total_images}")
        print(f"   Real Images Detected: {real_count} ({real_percentage:.1f}%)")
        print(f"   Fake Images Detected: {fake_count} ({fake_percentage:.1f}%)")
        print(f"   Average Confidence: {df['confidence'].mean():.4f}")
        print(f"   Confidence Std Dev: {df['confidence'].std():.4f}")
        
        # Confidence analysis
        print(f"\n🎯 CONFIDENCE ANALYSIS:")
        print(f"   Min Confidence: {df['confidence'].min():.4f}")
        print(f"   Max Confidence: {df['confidence'].max():.4f}")
        print(f"   Median Confidence: {df['confidence'].median():.4f}")
        
        # Class-specific confidence
        if real_count > 0:
            real_confidence = df[df['predicted_class'] == 'REAL']['confidence'].mean()
            print(f"   Avg Confidence (REAL): {real_confidence:.4f}")
        if fake_count > 0:
            fake_confidence = df[df['predicted_class'] == 'FAKE']['confidence'].mean()
            print(f"   Avg Confidence (FAKE): {fake_confidence:.4f}")
        
        # Save detailed report
        report_data = {
            'batch_summary': {
                'total_images': total_images,
                'real_detections': real_count,
                'fake_detections': fake_count,
                'real_percentage': float(real_percentage),
                'fake_percentage': float(fake_percentage),
                'avg_confidence': float(df['confidence'].mean()),
                'confidence_std': float(df['confidence'].std())
            },
            'confidence_stats': {
                'min': float(df['confidence'].min()),
                'max': float(df['confidence'].max()),
                'median': float(df['confidence'].median()),
                'q1': float(df['confidence'].quantile(0.25)),
                'q3': float(df['confidence'].quantile(0.75))
            },
            'class_confidence': {
                'real_avg_confidence': float(real_confidence) if real_count > 0 else 0,
                'fake_avg_confidence': float(fake_confidence) if fake_count > 0 else 0
            },
            'processing_info': {
                'timestamp': pd.Timestamp.now().isoformat(),
                'model_used': 'simple_deepfake_model.h5'
            }
        }
        
        # Save report
        output_dir = Path(r"D:\A Image Classification\deepfake_detector\models\predictions")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        report_path = output_dir / "batch_prediction_report.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Detailed report saved to: {report_path}")
        
        # Create visualization
        self.create_visualizations(df, output_dir)
        
        return report_data
    
    def create_visualizations(self, df, output_dir):
        """Create comprehensive visualizations of batch results"""
        print("🎨 CREATING BATCH ANALYSIS VISUALIZATIONS...")
        
        try:
            # Set style
            plt.style.use('default')
            sns.set_palette("husl")
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('Deepfake Batch Analysis Report', fontsize=16, fontweight='bold')
            
            # 1. Class distribution pie chart
            class_counts = df['predicted_class'].value_counts()
            colors = ['lightgreen', 'lightcoral']
            axes[0, 0].pie(class_counts.values, labels=class_counts.index, autopct='%1.1f%%', 
                          startangle=90, colors=colors)
            axes[0, 0].set_title('Class Distribution', fontweight='bold')
            
            # 2. Confidence distribution histogram
            axes[0, 1].hist(df['confidence'], bins=20, alpha=0.7, edgecolor='black', color='skyblue')
            axes[0, 1].axvline(df['confidence'].mean(), color='red', linestyle='--', 
                              label=f'Mean: {df["confidence"].mean():.3f}')
            axes[0, 1].set_xlabel('Confidence Score')
            axes[0, 1].set_ylabel('Frequency')
            axes[0, 1].set_title('Confidence Score Distribution', fontweight='bold')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
            
            # 3. Confidence by class box plot
            if len(df['predicted_class'].unique()) > 1:
                sns.boxplot(data=df, x='predicted_class', y='confidence', ax=axes[1, 0])
            else:
                # Handle case with only one class
                axes[1, 0].text(0.5, 0.5, f"Only {df['predicted_class'].iloc[0]} class found", 
                               ha='center', va='center', transform=axes[1, 0].transAxes)
            axes[1, 0].set_title('Confidence Scores by Class', fontweight='bold')
            axes[1, 0].set_xlabel('Predicted Class')
            axes[1, 0].set_ylabel('Confidence Score')
            
            # 4. Raw score distribution
            axes[1, 1].hist(df['raw_score'], bins=20, alpha=0.7, color='orange', edgecolor='black')
            axes[1, 1].axvline(0.5, color='red', linestyle='-', linewidth=2, label='Decision Threshold (0.5)')
            axes[1, 1].set_xlabel('Raw Prediction Score')
            axes[1, 1].set_ylabel('Frequency')
            axes[1, 1].set_title('Raw Prediction Score Distribution', fontweight='bold')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save visualization
            viz_path = output_dir / "batch_analysis_visualization.png"
            plt.savefig(viz_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            
            print(f"💾 Visualization saved to: {viz_path}")
            
        except Exception as e:
            print(f"❌ Visualization error: {e}")

def test_batch_prediction():
    """Test batch prediction with sample data"""
    print("🧪 TESTING BATCH PREDICTION SYSTEM...")
    
    try:
        # Initialize batch predictor
        batch_predictor = BatchPredictor()
        
        # Test with sample folder (use test data)
        test_folder = Path(r"D:\A Image Classification\deepfake_detector\data\processed\test")
        
        if not test_folder.exists():
            print("❌ Test folder not found!")
            return
        
        # Process batch (limit to 20 images for quick test)
        output_csv = Path(r"D:\A Image Classification\deepfake_detector\models\predictions\batch_results.csv")
        
        print("🚀 Starting batch processing...")
        results = batch_predictor.process_folder(test_folder, output_csv, max_images=20)
        
        if results:
            print("\n🎉 BATCH PREDICTION TEST COMPLETED SUCCESSFULLY!")
        else:
            print("\n❌ BATCH PREDICTION TEST FAILED!")
            
    except Exception as e:
        print(f"❌ Error in batch prediction test: {e}")

if __name__ == "__main__":
    test_batch_prediction()