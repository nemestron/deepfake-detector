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

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Set page config
st.set_page_config(
    page_title="Deepfake Detection System",
    page_icon="🕵️",
    layout="wide",
    initial_sidebar_state="expanded"
)

class StreamlitDeepfakeDetector:
    def __init__(self):
        self.model = None
        self.img_size = (128, 128)
        self.model_loaded = False
        self.load_model()
    
    def create_compatible_model(self):
        """Create a fresh compatible model architecture"""
        try:
            model = tf.keras.Sequential([
                tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
                tf.keras.layers.MaxPooling2D((2, 2)),
                tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
                tf.keras.layers.MaxPooling2D((2, 2)),
                tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
                tf.keras.layers.MaxPooling2D((2, 2)),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(512, activation='relu'),
                tf.keras.layers.Dense(1, activation='sigmoid')
            ])
            
            model.compile(optimizer='adam',
                         loss='binary_crossentropy',
                         metrics=['accuracy'])
            return model
        except Exception as e:
            st.error(f"Error creating model: {e}")
            return None
    
    def load_model(self):
        """Load the trained model with compatibility handling"""
        model_path = project_root / "models" / "simple_deepfake_model.h5"
        
        try:
            # First try to load the existing model
            if model_path.exists():
                st.info("🔄 Loading AI model... This may take a moment.")
                self.model = tf.keras.models.load_model(model_path)
                self.model_loaded = True
                st.success("✅ AI Model loaded successfully!")
                return True
            else:
                st.warning("⚠️ Model file not found. Using demo mode.")
                self.model_loaded = False
                return False
                
        except Exception as e:
            st.warning(f"⚠️ Model compatibility issue: {str(e)[:100]}...")
            st.info("🔄 Switching to demo mode with simulated analysis.")
            self.model_loaded = False
            return False
    
    def preprocess_image(self, image):
        """Preprocess uploaded image"""
        try:
            # Convert to numpy array
            if isinstance(image, Image.Image):
                image = np.array(image)
            
            # Ensure image has 3 channels
            if len(image.shape) == 2:  # Grayscale
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif image.shape[2] == 4:  # RGBA
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
            
            # Convert RGB to BGR (matching training)
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            # Resize
            image_resized = cv2.resize(image_bgr, self.img_size)
            
            # Normalize
            image_normalized = image_resized.astype(np.float32) / 255.0
            
            # Add batch dimension
            image_batch = np.expand_dims(image_normalized, axis=0)
            
            return image_batch, image_resized
            
        except Exception as e:
            st.error(f"Error preprocessing image: {e}")
            return None, None
    
    def predict_demo(self, image):
        """Demo prediction when model isn't available"""
        # Simulate analysis based on image characteristics
        try:
            img_array = np.array(image)
            
            # Simple heuristic based on image stats
            brightness = np.mean(img_array)
            contrast = np.std(img_array)
            
            # Fake images often have different statistical properties
            # This is just a demo - real model would be much more accurate
            if brightness > 127 and contrast > 45:
                raw_score = 0.7  # Likely real
            else:
                raw_score = 0.3  # Likely fake
            
            # Add some randomness to make it interesting
            raw_score += np.random.uniform(-0.2, 0.2)
            raw_score = max(0.1, min(0.9, raw_score))  # Clamp between 0.1-0.9
            
            if raw_score >= 0.5:
                predicted_class = "REAL"
                confidence = raw_score
            else:
                predicted_class = "FAKE"
                confidence = 1 - raw_score
            
            return {
                'predicted_class': predicted_class,
                'confidence': confidence,
                'raw_score': raw_score,
                'is_real': predicted_class == "REAL",
                'demo_mode': True
            }
            
        except Exception as e:
            return {
                'predicted_class': "UNKNOWN",
                'confidence': 0.5,
                'raw_score': 0.5,
                'is_real': False,
                'demo_mode': True
            }
    
    def predict(self, image, threshold=0.5):
        """Make prediction - uses demo if model not available"""
        if self.model_loaded and self.model is not None:
            try:
                processed_image, resized_image = self.preprocess_image(image)
                if processed_image is None:
                    return self.predict_demo(image)
                
                prediction = self.model.predict(processed_image, verbose=0)
                confidence_score = float(prediction[0][0])
                
                if confidence_score >= threshold:
                    predicted_class = "REAL"
                    confidence = confidence_score
                else:
                    predicted_class = "FAKE"
                    confidence = 1 - confidence_score
                
                return {
                    'predicted_class': predicted_class,
                    'confidence': confidence,
                    'raw_score': confidence_score,
                    'is_real': predicted_class == "REAL",
                    'demo_mode': False
                }
                
            except Exception as e:
                st.warning(f"Model prediction failed, using demo: {e}")
                return self.predict_demo(image)
        else:
            return self.predict_demo(image)

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
    .demo-prediction {
        background-color: #fff3cd;
        border: 3px solid #ffc107;
        color: #856404;
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
    </style>
    """, unsafe_allow_html=True)

def display_prediction_result(result, threshold):
    """Display the prediction results"""
    if result is None:
        st.error("❌ Analysis failed. Please try another image.")
        return
    
    confidence_percent = result['confidence'] * 100
    is_demo = result.get('demo_mode', False)
    
    # Prediction box
    if is_demo:
        prediction_class = "demo-prediction"
        emoji = "🔮"
        message = f"DEMO: {result['predicted_class']} (Simulated)"
        fill_color = "#ffc107"
    elif result['is_real']:
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
        
        if is_demo:
            st.caption("🎭 Demo Mode")
    
    # Detailed analysis
    st.subheader("📈 Detailed Analysis")
    
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.metric("Raw Prediction Score", f"{result['raw_score']:.4f}")
        st.metric("Decision Threshold", f"{threshold:.2f}")
    
    with col_b:
        # Interpretation
        distance_from_threshold = abs(result['raw_score'] - threshold)
        
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
        
        if is_demo:
            st.caption("Based on image statistics")

def main():
    # Apply custom CSS
    create_custom_css()
    
    # Header
    st.markdown('<h1 class="main-header">🕵️ Deepfake Detection System</h1>', unsafe_allow_html=True)
    
    # Initialize detector
    detector = StreamlitDeepfakeDetector()
    
    # Sidebar
    st.sidebar.title("⚙️ Configuration")
    confidence_threshold = st.sidebar.slider(
        "Confidence Threshold", 
        min_value=0.1, 
        max_value=0.9, 
        value=0.5, 
        step=0.05,
        help="Adjust the threshold for real/fake classification"
    )
    
    # Model status
    if detector.model_loaded:
        st.sidebar.success("✅ AI Model: ACTIVE")
        st.sidebar.info("Full AI analysis enabled")
    else:
        st.sidebar.warning("🤖 AI Model: DEMO MODE")
        st.sidebar.info("Using simulated analysis")
    
    st.sidebar.markdown("---")
    st.sidebar.info("""
    **📖 How to use:**
    1. Upload an image using the file uploader
    2. Or take a picture using your camera  
    3. Click 'Analyze Image' to get results
    
    **🔍 Interpretation:**
    - **REAL**: Authentic human face
    - **FAKE**: AI-generated or manipulated face
    - **Confidence**: Certainty of prediction
    """)
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📤 Upload Image")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose an image file", 
            type=['jpg', 'jpeg', 'png', 'bmp'],
            help="Upload a facial image for analysis"
        )
        
        # Camera input
        camera_file = st.camera_input("Or take a picture with your camera")
        
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
                
            except Exception as e:
                st.error(f"❌ Error loading image: {str(e)}")
                image_file = None
    
    with col2:
        st.subheader("📊 Analysis Results")
        
        if image_file is not None:
            # Analyze button
            if st.button("🔍 Analyze Image", type="primary", use_container_width=True):
                with st.spinner("🔄 Analyzing image..."):
                    try:
                        # Reload image for prediction
                        image = Image.open(image_file)
                        
                        # Make prediction
                        result = detector.predict(image, confidence_threshold)
                        
                        if result:
                            display_prediction_result(result, confidence_threshold)
                        else:
                            st.error("❌ Failed to analyze image.")
                            
                    except Exception as e:
                        st.error(f"❌ Analysis error: {str(e)}")
            
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
    
    # Footer
    st.markdown("---")
    
    if not detector.model_loaded:
        st.markdown("""
        <div class="warning-box">
        <strong>🔧 Model Compatibility Notice:</strong><br>
        The AI model is currently in demo mode due to compatibility issues. 
        For full AI analysis, we're working on model optimization.
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

if __name__ == "__main__":
    main()