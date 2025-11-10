import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import json
import argparse
import os
import sys

print("🎯 DEEPFACE PREDICTION SYSTEM - LOADING...")

class DeepfakePredictor:
    def __init__(self, model_path=None):
        self.model = None
        self.img_size = (128, 128)  # Match training size
        self.threshold = 0.5  # Default threshold
        
        # Set model path
        if model_path is None:
            base_dir = Path(r"D:\A Image Classification\deepfake_detector")
            model_path = base_dir / "models" / "simple_deepfake_model.h5"
        
        self.load_model(model_path)
    
    def load_model(self, model_path):
        """Load the trained model"""
        print(f"🔧 LOADING MODEL FROM: {model_path}")
        
        # Check if model exists
        if not Path(model_path).exists():
            print(f"❌ MODEL NOT FOUND: {model_path}")
            print("💡 Please train the model first using: python scripts/train_simple.py")
            sys.exit(1)
        
        try:
            self.model = tf.keras.models.load_model(model_path)
            print("✅ MODEL LOADED SUCCESSFULLY!")
            
        except Exception as e:
            print(f"❌ ERROR LOADING MODEL: {e}")
            print("💡 Try training the model first: python scripts/train_simple.py")
            sys.exit(1)
    
    def preprocess_image(self, image_path):
        """Preprocess image for prediction"""
        print(f"🖼️ PROCESSING IMAGE: {image_path}")
        
        try:
            # Read image
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError(f"Could not load image from {image_path}")
            
            # Convert BGR to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Resize to match model input
            image_resized = cv2.resize(image_rgb, self.img_size)
            
            # Normalize pixel values
            image_normalized = image_resized.astype(np.float32) / 255.0
            
            # Add batch dimension
            image_batch = np.expand_dims(image_normalized, axis=0)
            
            print(f"✅ Image processed: {image.shape} -> {image_resized.shape}")
            return image_batch, image_rgb, image_resized
            
        except Exception as e:
            print(f"❌ ERROR PROCESSING IMAGE: {e}")
            raise
    
    def predict_image(self, image_path, confidence_threshold=0.5):
        """Make prediction on a single image"""
        self.threshold = confidence_threshold
        
        # Check if image exists
        if not Path(image_path).exists():
            print(f"❌ IMAGE NOT FOUND: {image_path}")
            return None
        
        try:
            # Preprocess image
            processed_image, original_image, resized_image = self.preprocess_image(image_path)
            
            # Make prediction
            print("🧠 MAKING PREDICTION...")
            prediction = self.model.predict(processed_image, verbose=0)
            confidence_score = float(prediction[0][0])
            
            # Determine class
            if confidence_score >= self.threshold:
                predicted_class = "REAL"
                confidence = confidence_score
            else:
                predicted_class = "FAKE"
                confidence = 1 - confidence_score
            
            print(f"🎯 PREDICTION RESULT:")
            print(f"   Class: {predicted_class}")
            print(f"   Confidence: {confidence:.4f}")
            print(f"   Raw Score: {confidence_score:.4f}")
            
            return {
                'predicted_class': predicted_class,
                'confidence': confidence,
                'raw_score': confidence_score,
                'threshold': self.threshold,
                'is_real': predicted_class == "REAL",
                'image_path': str(image_path)
            }
            
        except Exception as e:
            print(f"❌ PREDICTION ERROR: {e}")
            return None
    
    def analyze_prediction_confidence(self, raw_score):
        """Analyze the confidence level"""
        if raw_score >= 0.8 or raw_score <= 0.2:
            level = "VERY HIGH CONFIDENCE"
            color = "🟢"
        elif raw_score >= 0.7 or raw_score <= 0.3:
            level = "HIGH CONFIDENCE" 
            color = "🟡"
        elif raw_score >= 0.6 or raw_score <= 0.4:
            level = "MEDIUM CONFIDENCE"
            color = "🟠"
        else:
            level = "LOW CONFIDENCE"
            color = "🔴"
        
        return f"{color} {level}"
    
    def visualize_prediction(self, image_path, prediction_result, save_path=None):
        """Create visualization of prediction result"""
        print("🎨 CREATING PREDICTION VISUALIZATION...")
        
        try:
            # Load and preprocess image for display
            original_image = cv2.imread(str(image_path))
            if original_image is None:
                print(f"❌ Could not load image for visualization: {image_path}")
                return
            
            original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
            
            # Create figure
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Plot original image
            ax1.imshow(original_image)
            ax1.set_title('Original Image', fontsize=14, fontweight='bold')
            ax1.axis('off')
            
            # Plot prediction info
            ax2.axis('off')
            
            # Prediction result text
            result_text = f"""🎯 DEEPFACE DETECTION RESULT

📊 PREDICTION: {prediction_result['predicted_class']}
💯 CONFIDENCE: {prediction_result['confidence']:.4f}
📈 RAW SCORE: {prediction_result['raw_score']:.4f}
⚖️ THRESHOLD: {prediction_result['threshold']}

{self.analyze_prediction_confidence(prediction_result['raw_score'])}

📁 IMAGE: {Path(image_path).name}"""
            
            # Color code based on prediction
            if prediction_result['predicted_class'] == "REAL":
                box_color = 'green'
                title = "✅ REAL IMAGE DETECTED"
            else:
                box_color = 'red' 
                title = "🚨 FAKE IMAGE DETECTED"
            
            ax2.text(0.1, 0.9, result_text, transform=ax2.transAxes, fontsize=12,
                    verticalalignment='top', fontfamily='monospace', fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgray', alpha=0.7))
            
            # Add colored bounding box
            fig.patch.set_facecolor('lightgray')
            fig.suptitle(title, fontsize=16, fontweight='bold', color=box_color)
            
            plt.tight_layout()
            
            # Save or show
            if save_path:
                # Ensure directory exists
                Path(save_path).parent.mkdir(parents=True, exist_ok=True)
                plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='lightgray')
                print(f"💾 Visualization saved to: {save_path}")
            else:
                plt.show()
            
            plt.close()
            
        except Exception as e:
            print(f"❌ VISUALIZATION ERROR: {e}")

def test_prediction_system():
    """Test the prediction system with sample images"""
    print("🧪 TESTING PREDICTION SYSTEM...")
    
    # Initialize predictor
    predictor = DeepfakePredictor()
    
    # Test with images from our dataset
    test_images = []
    base_dir = Path(r"D:\A Image Classification\deepfake_detector")
    data_path = base_dir / "data" / "processed"
    
    # Find sample real and fake images
    try:
        real_samples = list((data_path / "test" / "real").glob("*.jpg"))[:2]
        fake_samples = list((data_path / "test" / "fake").glob("*.jpg"))[:2]
        
        test_images.extend(real_samples)
        test_images.extend(fake_samples)
        
    except Exception as e:
        print(f"❌ Error finding test images: {e}")
        return
    
    if not test_images:
        print("❌ No test images found in dataset!")
        print("💡 Make sure you have processed the dataset first")
        return
    
    results = []
    
    for image_path in test_images:
        print(f"\n" + "="*50)
        print(f"🔍 ANALYZING: {image_path.name}")
        print("="*50)
        
        try:
            # Make prediction
            result = predictor.predict_image(image_path)
            if result:
                results.append(result)
                
                # Create visualization
                output_dir = base_dir / "models" / "predictions"
                save_path = output_dir / f"prediction_{Path(image_path).stem}.png"
                predictor.visualize_prediction(image_path, result, save_path)
                
                print(f"✅ Analysis completed for {image_path.name}")
            else:
                print(f"❌ Failed to analyze {image_path.name}")
            
        except Exception as e:
            print(f"❌ Error analyzing {image_path.name}: {e}")
    
    # Print summary
    if results:
        print(f"\n📊 PREDICTION SUMMARY:")
        print(f"   Total images analyzed: {len(results)}")
        
        real_count = sum(1 for r in results if r['predicted_class'] == "REAL")
        fake_count = sum(1 for r in results if r['predicted_class'] == "FAKE")
        
        print(f"   REAL predictions: {real_count}")
        print(f"   FAKE predictions: {fake_count}")
        
        # Save results
        results_data = {
            'test_summary': {
                'total_images': len(results),
                'real_predictions': real_count,
                'fake_predictions': fake_count,
                'average_confidence': np.mean([r['confidence'] for r in results])
            },
            'predictions': results
        }
        
        results_path = output_dir / "prediction_results.json"
        with open(results_path, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        print(f"💾 Results saved to: {results_path}")
        print("🎉 PREDICTION SYSTEM TEST COMPLETED!")
    else:
        print("❌ No successful predictions made!")

def predict_single_image(image_path, threshold=0.5):
    """Convenience function to predict a single image"""
    predictor = DeepfakePredictor()
    
    result = predictor.predict_image(image_path, threshold)
    
    if result:
        # Create visualization
        base_dir = Path(r"D:\A Image Classification\deepfake_detector")
        output_dir = base_dir / "models" / "predictions"
        save_path = output_dir / f"prediction_{Path(image_path).stem}.png"
        predictor.visualize_prediction(image_path, result, save_path)
        
        return result
    else:
        return None

if __name__ == "__main__":
    print("🚀 DEEPFACE PREDICTION SYSTEM")
    print("=" * 50)
    
    # Test the prediction system
    test_prediction_system()
    
    print("\n" + "="*60)
    print("🎯 PREDICTION SYSTEM READY FOR USE!")
    print("="*60)
    print("💡 USAGE EXAMPLES:")
    print("   python scripts/predict_image.py")
    print("   OR")
    print("   from predict_image import predict_single_image")
    print("   result = predict_single_image('path/to/your/image.jpg')")
    print("\n📁 Outputs saved to: models/predictions/")