import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import os
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# Set page config
st.set_page_config(
    page_title="Deepfake Detection System",
    page_icon="🕵️",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def load_model():
    """Load the trained model with caching"""
    try:
        model_path = Path("models/simple_deepfake_model.h5")
        if model_path.exists():
            model = tf.keras.models.load_model(model_path)
            return model, True
        else:
            st.warning("🤖 Model file not found. Running in demo mode.")
            return None, False
    except Exception as e:
        st.warning(f"⚠️ Model loading failed: {str(e)[:100]}... Running in demo mode.")
        return None, False

# Initialize model
model, model_loaded = load_model()
IMG_SIZE = (128, 128)

def create_custom_css():
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
    </style>
    """, unsafe_allow_html=True)

def preprocess_image(image):
    """Preprocess uploaded image"""
    try:
        # Convert to numpy array
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # Ensure image has 3 channels
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        
        # Convert RGB to BGR
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Resize and normalize
        image_resized = cv2.resize(image_bgr, IMG_SIZE)
        image_normalized = image_resized.astype(np.float32) / 255.0
        image_batch = np.expand_dims(image_normalized, axis=0)
        
        return image_batch, image_resized
        
    except Exception as e:
        st.error(f"❌ Error preprocessing image: {str(e)}")
        return None, None

def predict_demo(image):
    """Demo prediction when model isn't available"""
    try:
        img_array = np.array(image)
        brightness = np.mean(img_array)
        contrast = np.std(img_array)
        
        # Simple demo logic
        if brightness > 127 and contrast > 45:
            raw_score = 0.7 + np.random.uniform(-0.1, 0.1)
        else:
            raw_score = 0.3 + np.random.uniform(-0.1, 0.1)
        
        raw_score = max(0.1, min(0.9, raw_score))
        
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
        
    except Exception:
        return {
            'predicted_class': "UNKNOWN",
            'confidence': 0.5,
            'raw_score': 0.5,
            'is_real': False,
            'demo_mode': True
        }

def predict_image(image, threshold=0.5):
    """Make prediction - uses AI model if available, otherwise demo"""
    if model_loaded and model is not None:
        try:
            processed_image, _ = preprocess_image(image)
            if processed_image is None:
                return predict_demo(image)
            
            prediction = model.predict(processed_image, verbose=0)
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
            st.warning(f"Model prediction failed: {e}")
            return predict_demo(image)
    else:
        return predict_demo(image)

def main():
    create_custom_css()
    
    # Header
    st.markdown('<h1 class="main-header">🕵️ Deepfake Detection System</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("⚙️ Configuration")
    confidence_threshold = st.sidebar.slider(
        "Confidence Threshold", 0.1, 0.9, 0.5, 0.05,
        help="Adjust the threshold for real/fake classification"
    )
    
    # Model status
    if model_loaded:
        st.sidebar.success("✅ AI Model: ACTIVE")
    else:
        st.sidebar.warning("🤖 AI Model: DEMO MODE")
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📤 Upload Image")
        uploaded_file = st.file_uploader("Choose an image", type=['jpg', 'jpeg', 'png'])
        camera_file = st.camera_input("Or use camera")
        
        image_file = uploaded_file or camera_file
        
        if image_file:
            try:
                image = Image.open(image_file)
                st.image(image, caption="Uploaded Image", use_column_width=True)
            except Exception as e:
                st.error(f"Error loading image: {e}")
    
    with col2:
        st.subheader("📊 Analysis Results")
        
        if image_file:
            if st.button("🔍 Analyze Image", type="primary", use_container_width=True):
                with st.spinner("Analyzing..."):
                    image = Image.open(image_file)
                    result = predict_image(image, confidence_threshold)
                    
                    # Display results
                    confidence_percent = result['confidence'] * 100
                    
                    if result['demo_mode']:
                        prediction_class = "demo-prediction"
                        emoji = "🔮"
                        message = f"DEMO: {result['predicted_class']}"
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
                    
                    st.markdown(f"""
                    <div class="prediction-box {prediction_class}">
                        <h2>{emoji} {message}</h2>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.metric("Confidence Score", f"{confidence_percent:.1f}%")
                    
                    # Confidence meter
                    st.markdown("**Confidence Level:**")
                    st.markdown(f"""
                    <div class="confidence-meter">
                        <div class="confidence-fill" style="width: {confidence_percent}%; background-color: {fill_color};">
                            {confidence_percent:.1f}%
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Detailed info
                    st.subheader("📈 Detailed Analysis")
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.metric("Raw Score", f"{result['raw_score']:.4f}")
                        st.metric("Threshold", f"{confidence_threshold:.2f}")
                    with col_b:
                        if result['confidence'] > 0.8:
                            certainty = "Very High"
                        elif result['confidence'] > 0.7:
                            certainty = "High"
                        elif result['confidence'] > 0.6:
                            certainty = "Medium"
                        else:
                            certainty = "Low"
                        st.metric("Certainty", certainty)
        else:
            st.info("👆 Upload an image or use camera to begin analysis")

if __name__ == "__main__":
    main()