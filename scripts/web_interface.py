import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import io
import tempfile
import os
from pathlib import Path
import matplotlib.pyplot as plt
import sys
import traceback

# Add the project root to Python path to import our modules
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Set TensorFlow logging level to reduce verbose output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class StreamlitDeepfakeDetector:
    def __init__(self):
        self.model = None
        self.img_size = (128, 128)
        self.model_loaded = False
        self.load_model()
    
    def load_model(self):
        """Load the trained model with comprehensive debugging"""
        model_path = project_root / "models" / "simple_deepfake_model.h5"
        
        st.sidebar.write("🔧 **Model Loading Debug:**")
        st.sidebar.write(f"Looking for model at: `{model_path}`")
        
        try:
            # Check if model file exists
            if not model_path.exists():
                st.sidebar.error(f"❌ Model file not found!")
                st.sidebar.write(f"Expected path: `{model_path}`")
                st.sidebar.write("💡 **Solution:** Run `python train_simple.py` first")
                return False
            
            st.sidebar.write("✅ Model file found!")
            st.sidebar.write("🔄 Loading model into memory...")
            
            # Load the model
            self.model = tf.keras.models.load_model(model_path)
            
            # Test model with a dummy prediction to verify it works
            st.sidebar.write("🧪 Testing model with sample input...")
            dummy_input = np.random.random((1, 128, 128, 3)).astype(np.float32)
            dummy_prediction = self.model.predict(dummy_input, verbose=0)
            
            st.sidebar.write(f"✅ Model test passed! Output shape: {dummy_prediction.shape}")
            st.sidebar.success("🎉 Model loaded successfully!")
            
            self.model_loaded = True
            return True
            
        except Exception as e:
            st.sidebar.error(f"❌ Error loading model!")
            st.sidebar.write(f"**Error details:** `{str(e)}`")
            st.sidebar.write("🔍 **Full traceback:**")
            st.sidebar.code(traceback.format_exc())
            
            # Provide specific solutions based on common errors
            if "h5py" in str(e):
                st.sidebar.write("💡 **Fix:** Try: `pip install h5py`")
            elif "CUDA" in str(e) or "GPU" in str(e):
                st.sidebar.write("💡 **Fix:** GPU issue - model will use CPU")
            elif "shape" in str(e):
                st.sidebar.write("💡 **Fix:** Model architecture mismatch")
            
            return False
    
    def preprocess_image(self, image):
        """Preprocess uploaded image for model prediction with error handling"""
        try:
            st.sidebar.write("🔄 Preprocessing image...")
            
            # Convert PIL Image to numpy array
            if isinstance(image, Image.Image):
                image = np.array(image)
                st.sidebar.write(f"✅ Converted PIL to numpy array - Shape: {image.shape}")
            
            # Ensure image has 3 channels (RGB)
            if len(image.shape) == 2:  # Grayscale
                st.sidebar.write("🔄 Converting grayscale to RGB...")
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif image.shape[2] == 4:  # RGBA
                st.sidebar.write("🔄 Converting RGBA to RGB...")
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
            
            st.sidebar.write(f"✅ Image channels: {image.shape[2]}")
            
            # Convert RGB to BGR (matching training preprocessing)
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            # Resize to match model input
            st.sidebar.write(f"🔄 Resizing to {self.img_size}...")
            image_resized = cv2.resize(image_bgr, self.img_size)
            
            # Normalize pixel values
            image_normalized = image_resized.astype(np.float32) / 255.0
            st.sidebar.write(f"✅ Normalized pixel range: [{image_normalized.min():.3f}, {image_normalized.max():.3f}]")
            
            # Add batch dimension
            image_batch = np.expand_dims(image_normalized, axis=0)
            st.sidebar.write(f"✅ Final input shape: {image_batch.shape}")
            
            return image_batch, image_resized
            
        except Exception as e:
            st.sidebar.error(f"❌ Error preprocessing image!")
            st.sidebar.write(f"**Error:** `{str(e)}`")
            return None, None
    
    def predict(self, image, threshold=0.5):
        """Make prediction on the image with comprehensive debugging"""
        if not self.model_loaded:
            st.error("Model not loaded - cannot make predictions")
            return None
            
        try:
            st.sidebar.write("🎯 Starting prediction...")
            
            processed_image, resized_image = self.preprocess_image(image)
            
            if processed_image is None:
                return None
            
            # Make prediction
            st.sidebar.write("🧠 Running model prediction...")
            prediction = self.model.predict(processed_image, verbose=0)
            confidence_score = float(prediction[0][0])
            
            st.sidebar.write(f"📊 Raw prediction score: {confidence_score:.4f}")
            st.sidebar.write(f"⚖️ Decision threshold: {threshold}")
            
            # Determine class based on threshold
            if confidence_score >= threshold:
                predicted_class = "REAL"
                confidence = confidence_score
                st.sidebar.write(f"✅ Classification: REAL (score ≥ {threshold})")
            else:
                predicted_class = "FAKE"
                confidence = 1 - confidence_score
                st.sidebar.write(f"🚨 Classification: FAKE (score < {threshold})")
            
            st.sidebar.success(f"🎉 Prediction complete! Confidence: {confidence:.1%}")
            
            return {
                'predicted_class': predicted_class,
                'confidence': confidence,
                'raw_score': confidence_score,
                'is_real': predicted_class == "REAL",
                'threshold': threshold
            }
            
        except Exception as e:
            st.sidebar.error(f"❌ Prediction error!")
            st.sidebar.write(f"**Error:** `{str(e)}`")
            st.sidebar.code(traceback.format_exc())
            return None

def create_custom_css():
    """Create custom CSS for better styling"""
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .prediction-box {
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        text-align: center;
        font-size: 1.2rem;
        font-weight: bold;
    }
    .real-prediction {
        background-color: #d4edda;
        border: 3px solid #28a745;
        color: #155724;
    }
    .fake-prediction {
        background-color: #f8d7da;
        border: 3px solid #dc3545;
        color: #721c24;
    }
    .confidence-meter {
        height: 25px;
        background-color: #e9ecef;
        border-radius: 12px;
        margin: 15px 0;
        overflow: hidden;
    }
    .confidence-fill {
        height: 100%;
        border-radius: 12px;
        text-align: center;
        color: white;
        font-weight: bold;
        line-height: 25px;
        transition: width 0.5s ease-in-out;
    }
    .info-box {
        background-color: #e7f3ff;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
    }
    .debug-box {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #6c757d;
        margin: 1rem 0;
        font-family: monospace;
        font-size: 0.8rem;
    }
    </style>
    """, unsafe_allow_html=True)

def display_prediction_result(result):
    """Display the prediction results in a formatted way"""
    if result is None:
        st.error("No prediction result available")
        return
    
    confidence_percent = result['confidence'] * 100
    
    # Prediction box
    if result['is_real']:
        prediction_class = "real-prediction"
        emoji = "✅"
        message = "REAL IMAGE DETECTED"
        fill_color = "#28a745"
    else:
        prediction_class = "fake-prediction"
        emoji = "🚨"
        message = "FAKE IMAGE DETECTED"
        fill_color = "#dc3545"
    
    st.markdown(
        f"""
        <div class="prediction-box {prediction_class}">
            <h2>{emoji} {message}</h2>
        </div>
        """, 
        unsafe_allow_html=True
    )
    
    # Confidence score and meter
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.metric(
            label="Confidence Score", 
            value=f"{confidence_percent:.1f}%"
        )
        
        # Confidence meter
        st.markdown("**Confidence Level:**")
        meter_html = f"""
        <div class="confidence-meter">
            <div class="confidence-fill" style="width: {confidence_percent}%; background-color: {fill_color};">
                {confidence_percent:.1f}%
            </div>
        </div>
        """
        st.markdown(meter_html, unsafe_allow_html=True)
    
    with col2:
        # Model certainty
        if result['confidence'] > 0.8:
            certainty = "Very High"
            certainty_color = "green"
        elif result['confidence'] > 0.7:
            certainty = "High"
            certainty_color = "blue"
        elif result['confidence'] > 0.6:
            certainty = "Medium"
            certainty_color = "orange"
        else:
            certainty = "Low"
            certainty_color = "red"
            
        st.markdown(f"**Certainty:** <span style='color:{certainty_color}; font-weight:bold'>{certainty}</span>", 
                   unsafe_allow_html=True)
    
    # Detailed analysis
    st.subheader("📈 Detailed Analysis")
    
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.metric("Raw Prediction Score", f"{result['raw_score']:.4f}")
        st.metric("Decision Threshold", f"{result['threshold']:.2f}")
    
    with col_b:
        # Interpretation based on distance from threshold
        distance_from_threshold = abs(result['raw_score'] - result['threshold'])
        
        if distance_from_threshold > 0.3:
            interpretation = "Very confident prediction"
            interpretation_color = "green"
        elif distance_from_threshold > 0.2:
            interpretation = "Confident prediction"
            interpretation_color = "blue"
        elif distance_from_threshold > 0.1:
            interpretation = "Moderately confident"
            interpretation_color = "orange"
        else:
            interpretation = "Close to decision boundary"
            interpretation_color = "red"
            
        st.markdown(f"**Interpretation:** <span style='color:{interpretation_color}'>{interpretation}</span>", 
                   unsafe_allow_html=True)
        
        # Show which side of threshold
        if result['is_real']:
            threshold_info = f"{result['raw_score'] - result['threshold']:.3f} above threshold"
        else:
            threshold_info = f"{result['threshold'] - result['raw_score']:.3f} below threshold"
        st.caption(f"Distance: {threshold_info}")

def main():
    # Set page configuration
    st.set_page_config(
        page_title="Deepfake Detection System",
        page_icon="🕵️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Apply custom CSS
    create_custom_css()
    
    # Header
    st.markdown('<h1 class="main-header">🕵️ Deepfake Detection System</h1>', unsafe_allow_html=True)
    
    # Initialize detector with debug info
    st.sidebar.title("🔧 System Status")
    with st.spinner("Loading AI model..."):
        detector = StreamlitDeepfakeDetector()
    
    if not detector.model_loaded:
        st.error("""
        ❌ **Model could not be loaded.** 
        
        **Please ensure:**
        - The model file exists at: `models/simple_deepfake_model.h5`
        - You have trained the model using: `python scripts/train_simple.py`
        - Check the sidebar for detailed error information
        
        **Quick Fix Commands:**
        ```bash
        cd "D:\\A Image Classification\\deepfake_detector"
        python scripts/train_simple.py
        ```
        """)
        
        # Show debug information
        with st.expander("🔍 Technical Debug Information"):
            st.write("**System Information:**")
            st.write(f"- Python version: {sys.version}")
            st.write(f"- TensorFlow version: {tf.__version__}")
            st.write(f"- Working directory: {os.getcwd()}")
            st.write(f"- Project root: {project_root}")
            
            # Check for model file
            model_path = project_root / "models" / "simple_deepfake_model.h5"
            st.write(f"- Model file exists: {model_path.exists()}")
            
            if model_path.exists():
                st.write(f"- Model file size: {model_path.stat().st_size / (1024*1024):.2f} MB")
        
        return
    
    # Sidebar configuration (only shown if model loads successfully)
    st.sidebar.title("⚙️ Configuration")
    
    confidence_threshold = st.sidebar.slider(
        "Confidence Threshold", 
        min_value=0.1, 
        max_value=0.9, 
        value=0.5, 
        step=0.05,
        help="Adjust the threshold for real/fake classification. Higher values make REAL detection stricter."
    )
    
    # Debug mode toggle
    debug_mode = st.sidebar.checkbox("Enable Debug Mode", value=False, 
                                   help="Show detailed processing information")
    
    st.sidebar.markdown("---")
    st.sidebar.info("""
    **📖 How to use:**
    1. Upload an image using the file uploader below
    2. Or take a picture using your camera
    3. Click 'Analyze Image' to get results
    
    **🔍 Interpretation:**
    - **REAL**: Authentic human face
    - **FAKE**: AI-generated or manipulated face
    - **Confidence**: Model's certainty in prediction
    
    **⚖️ Accuracy Note:**
    This model achieves 91% accuracy on test data. 
    Use as an assistive tool, not absolute proof.
    """)
    
    # Model info in sidebar
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔧 Model Information")
    st.sidebar.write(f"**Input Size:** 128×128 pixels")
    st.sidebar.write(f"**Model:** CNN (Convolutional Neural Network)")
    st.sidebar.write(f"**Training Accuracy:** 93.75%")
    st.sidebar.write(f"**Validation Accuracy:** 91.05%")
    
    # Debug information in sidebar if enabled
    if debug_mode:
        st.sidebar.markdown("---")
        st.sidebar.subheader("🐛 Debug Information")
        st.sidebar.write(f"**TensorFlow Version:** {tf.__version__}")
        st.sidebar.write(f"**Model Loaded:** {detector.model_loaded}")
        if detector.model:
            st.sidebar.write(f"**Model Layers:** {len(detector.model.layers)}")
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📤 Upload Image")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose an image file", 
            type=['jpg', 'jpeg', 'png', 'bmp'],
            help="Supported formats: JPG, JPEG, PNG, BMP"
        )
        
        # Camera input
        st.write("**Or use your camera:**")
        camera_file = st.camera_input("Take a picture")
        
        # Use whichever input is available
        image_file = uploaded_file or camera_file
        
        if image_file is not None:
            try:
                # Display uploaded image
                image = Image.open(image_file)
                st.image(image, caption="Uploaded Image", use_column_width=True)
                
                # File info
                file_size = len(image_file.getvalue()) / 1024  # KB
                st.caption(f"File: {image_file.name} | Size: {file_size:.1f} KB | Format: {image.format}")
                
                # Debug info
                if debug_mode:
                    with st.expander("📊 Image Debug Info"):
                        st.write(f"**PIL Image Mode:** {image.mode}")
                        st.write(f"**PIL Image Size:** {image.size}")
                        st.write(f"**NumPy Array Shape:** {np.array(image).shape}")
                
            except Exception as e:
                st.error(f"❌ Error loading image: {str(e)}")
                if debug_mode:
                    st.code(traceback.format_exc())
                image_file = None
    
    with col2:
        st.subheader("📊 Analysis Results")
        
        if image_file is not None:
            # Analyze button
            if st.button("🔍 Analyze Image", type="primary", use_container_width=True):
                # Clear previous debug output
                if debug_mode:
                    st.sidebar.markdown("---")
                    st.sidebar.subheader("🔍 Analysis Debug Log")
                
                with st.spinner("🔄 Analyzing image... This may take a few seconds."):
                    try:
                        # Reload image for prediction
                        image = Image.open(image_file)
                        
                        # Make prediction
                        result = detector.predict(image, confidence_threshold)
                        
                        if result:
                            display_prediction_result(result)
                            
                            # Show debug visualization if enabled
                            if debug_mode:
                                with st.expander("🖼️ Debug Visualization"):
                                    st.write("**Processed Image (resized for model):**")
                                    processed_img, _ = detector.preprocess_image(image)
                                    if processed_img is not None:
                                        # Display the preprocessed image
                                        debug_img = (processed_img[0] * 255).astype(np.uint8)
                                        st.image(debug_img, caption="Model Input (128×128)", use_column_width=False, width=200)
                        else:
                            st.error("❌ Failed to analyze image. Please try another image.")
                            
                    except Exception as e:
                        st.error(f"❌ Analysis error: {str(e)}")
                        if debug_mode:
                            st.code(traceback.format_exc())
            
            # Show placeholder when no analysis yet
            else:
                st.info("👆 Click 'Analyze Image' to see results here")
                st.markdown("""
                <div class="info-box">
                <strong>What to expect:</strong><br>
                - Real/Fake classification<br>
                - Confidence percentage<br>
                - Detailed analysis metrics<br>
                - Interpretation guidance
                </div>
                """, unsafe_allow_html=True)
        
        else:
            st.info("📁 Upload an image or take a photo to begin analysis")
            st.markdown("""
            <div class="info-box">
            <strong>Supported images:</strong><br>
            • Portraits and facial images work best<br>
            • Clear, well-lit images give better results<br>
            • Various image formats supported
            </div>
            """, unsafe_allow_html=True)
    
    # Footer and additional information
    st.markdown("---")
    
    st.markdown("""
    <div class="warning-box">
    <strong>⚠️ Important Disclaimer:</strong><br>
    This deepfake detection system is a proof-of-concept with 91% accuracy. 
    It should be used as an assistive tool for analysis, not as definitive proof of authenticity. 
    Always verify important findings through multiple methods.
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown(
        """
        <div style='text-align: center; color: #666; margin-top: 2rem;'>
            <p>Built with TensorFlow & Streamlit | Deepfake Detection System v1.0</p>
            <p>Model trained on 39,662 facial images | Validation Accuracy: 91.05%</p>
        </div>
        """,
        unsafe_allow_html=True
    )

# Test function to check if everything works
def test_web_interface():
    """Test the web interface components"""
    print("🌐 WEB INTERFACE CODE READY!")
    print("💡 To run the web interface, use: streamlit run scripts/web_interface.py")
    print("💡 Make sure to install streamlit first: pip install streamlit")
    
    # Check dependencies
    try:
        import streamlit
        print("✅ Streamlit is installed!")
    except ImportError:
        print("❌ Streamlit not installed. Run: pip install streamlit")
        return False
    
    # Check if model exists
    model_path = project_root / "models" / "simple_deepfake_model.h5"
    if model_path.exists():
        print("✅ Model file found!")
        print(f"   Location: {model_path}")
        print(f"   Size: {model_path.stat().st_size / (1024*1024):.2f} MB")
    else:
        print("❌ Model file not found. Train the model first.")
        print("   Run: python scripts/train_simple.py")
        return False
    
    return True

if __name__ == "__main__":
    # When running directly (not via streamlit), run tests
    if test_web_interface():
        print("\n🎉 Web interface is ready to run!")
        print("\n🚀 To launch the web app, run this command in your terminal:")
        print("   streamlit run scripts/web_interface.py")
    else:
        print("\n❌ Some issues need to be fixed before running the web interface.")