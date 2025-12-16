"""
LC25000 Histopathology Image Classifier
A professional web application for lung and colon cancer tissue classification
"""

import streamlit as st
import numpy as np
from PIL import Image
import io

# Page configuration
st.set_page_config(
    page_title="LC25000 Histopathology Classifier",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional UI
st.markdown("""
<style>
    /* Main container styling */
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #e4e8ec 100%);
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(90deg, #1e3a5f 0%, #2d5a87 100%);
        padding: 2rem 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    .main-header h1 {
        color: white;
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
        font-weight: 700;
    }
    
    .main-header p {
        color: #b8d4e8;
        font-size: 1.1rem;
        margin: 0;
    }
    
    /* Card styling */
    .info-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.08);
        margin-bottom: 1rem;
        border-left: 4px solid #2d5a87;
    }
    
    .result-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
        text-align: center;
        margin-top: 1rem;
    }
    
    .prediction-label {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1e3a5f;
        margin-bottom: 0.5rem;
    }
    
    .confidence-text {
        font-size: 1.2rem;
        color: #5a6c7d;
    }
    
    /* Class cards */
    .class-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        transition: transform 0.2s;
    }
    
    .class-card:hover {
        transform: translateY(-2px);
    }
    
    /* Lung classes */
    .lung-benign { border-top: 4px solid #22c55e; }
    .lung-adeno { border-top: 4px solid #f59e0b; }
    .lung-squamous { border-top: 4px solid #ef4444; }
    
    /* Colon classes */
    .colon-adeno { border-top: 4px solid #8b5cf6; }
    .colon-benign { border-top: 4px solid #06b6d4; }
    
    /* Upload area */
    .upload-section {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        border: 2px dashed #cbd5e1;
        text-align: center;
        margin: 1rem 0;
    }
    
    /* Sidebar styling */
    .sidebar .sidebar-content {
        background: #1e3a5f;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(90deg, #2d5a87 0%, #1e3a5f 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 8px;
        font-weight: 600;
        width: 100%;
        transition: all 0.3s;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(45, 90, 135, 0.4);
    }
    
    /* Progress bar colors */
    .stProgress > div > div {
        background: linear-gradient(90deg, #2d5a87 0%, #1e3a5f 100%);
    }
    
    /* Metric styling */
    .metric-container {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem;
        color: #64748b;
        font-size: 0.9rem;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Responsive adjustments */
    @media (max-width: 768px) {
        .main-header h1 {
            font-size: 1.8rem;
        }
    }
</style>
""", unsafe_allow_html=True)


# Class definitions
CLASS_NAMES = [
    "Colon Adenocarcinoma",
    "Colon Benign Tissue", 
    "Lung Adenocarcinoma",
    "Lung Benign Tissue",
    "Lung Squamous Cell Carcinoma"
]

CLASS_DESCRIPTIONS = {
    "Colon Adenocarcinoma": {
        "description": "A type of cancer that forms in the glandular cells lining the colon.",
        "severity": "Malignant",
        "color": "#8b5cf6",
        "icon": "🔴"
    },
    "Colon Benign Tissue": {
        "description": "Normal, non-cancerous colon tissue with healthy cellular structure.",
        "severity": "Benign",
        "color": "#06b6d4",
        "icon": "🟢"
    },
    "Lung Adenocarcinoma": {
        "description": "The most common type of lung cancer, originating in glandular cells.",
        "severity": "Malignant",
        "color": "#f59e0b",
        "icon": "🔴"
    },
    "Lung Benign Tissue": {
        "description": "Normal, healthy lung tissue without any cancerous cells.",
        "severity": "Benign",
        "color": "#22c55e",
        "icon": "🟢"
    },
    "Lung Squamous Cell Carcinoma": {
        "description": "A type of lung cancer that begins in squamous cells lining the airways.",
        "severity": "Malignant",
        "color": "#ef4444",
        "icon": "🔴"
    }
}


@st.cache_resource
def load_model():
    """Load the Keras model with caching"""
    try:
        import tensorflow as tf
        model = tf.keras.models.load_model('lc2500_model.keras')
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None


def preprocess_image(image, target_size=(768, 768)):
    """Preprocess image for model prediction"""
    # Resize image
    image = image.resize(target_size)
    # Convert to array
    img_array = np.array(image)
    # Ensure RGB
    if len(img_array.shape) == 2:
        img_array = np.stack([img_array] * 3, axis=-1)
    elif img_array.shape[-1] == 4:
        img_array = img_array[:, :, :3]
    # Normalize
    img_array = img_array.astype('float32') / 255.0
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


def predict(model, image):
    """Make prediction on preprocessed image"""
    predictions = model.predict(image, verbose=0)
    predicted_class = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class]
    return CLASS_NAMES[predicted_class], confidence, predictions[0]


def render_header():
    """Render the main header"""
    st.markdown("""
    <div class="main-header">
        <h1>🔬 LC25000 Histopathology Classifier</h1>
        <p>AI-powered lung and colon cancer tissue classification from histopathological images</p>
    </div>
    """, unsafe_allow_html=True)


def render_sidebar():
    """Render the sidebar with information"""
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/microscope.png", width=80)
        st.markdown("### About This Tool")
        st.markdown("""
        This application uses deep learning to classify histopathological images 
        into one of five tissue types related to lung and colon cancer.
        """)
        
        st.markdown("---")
        st.markdown("### 📊 Dataset Information")
        st.markdown("""
        - **Total Images:** 25,000
        - **Image Size:** 768 × 768 px
        - **Format:** JPEG
        - **Classes:** 5
        - **Images per Class:** 5,000
        """)
        
        st.markdown("---")
        st.markdown("### 🏥 Tissue Classes")
        
        for class_name, info in CLASS_DESCRIPTIONS.items():
            severity_color = "#22c55e" if info["severity"] == "Benign" else "#ef4444"
            st.markdown(f"""
            <div style="background: white; padding: 0.8rem; border-radius: 8px; margin-bottom: 0.5rem; border-left: 3px solid {info['color']};">
                <strong>{info['icon']} {class_name}</strong><br>
                <small style="color: {severity_color};">● {info['severity']}</small>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("### ⚠️ Disclaimer")
        st.warning("""
        This tool is for educational and research purposes only. 
        It should not be used as a substitute for professional medical diagnosis.
        """)
        
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; color: #64748b; font-size: 0.8rem;">
            Built with ❤️ using Streamlit<br>
            Powered by TensorFlow/Keras
        </div>
        """, unsafe_allow_html=True)


def render_upload_section():
    """Render the image upload section"""
    st.markdown("### 📤 Upload Histopathology Image")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Choose a histopathology image (JPEG, PNG)",
            type=['jpg', 'jpeg', 'png'],
            help="Upload a 768x768 pixel histopathology image for classification"
        )
    
    with col2:
        st.markdown("""
        <div class="info-card">
            <strong>📋 Requirements</strong>
            <ul style="margin: 0.5rem 0; padding-left: 1.2rem; color: #5a6c7d;">
                <li>Format: JPEG or PNG</li>
                <li>Recommended: 768×768 px</li>
                <li>Type: Histopathology image</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    return uploaded_file


def render_results(image, prediction, confidence, all_predictions):
    """Render the prediction results"""
    class_info = CLASS_DESCRIPTIONS[prediction]
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 🖼️ Uploaded Image")
        st.image(image, use_container_width=True)
    
    with col2:
        st.markdown("### 🎯 Classification Result")
        
        # Main prediction card
        severity_badge = "🟢 Benign" if class_info["severity"] == "Benign" else "🔴 Malignant"
        st.markdown(f"""
        <div class="result-card">
            <div class="prediction-label" style="color: {class_info['color']};">
                {prediction}
            </div>
            <div class="confidence-text">
                Confidence: <strong>{confidence*100:.1f}%</strong>
            </div>
            <div style="margin-top: 1rem; padding: 0.5rem; background: {'#dcfce7' if class_info['severity'] == 'Benign' else '#fee2e2'}; 
                        border-radius: 20px; display: inline-block;">
                {severity_badge}
            </div>
            <p style="margin-top: 1rem; color: #5a6c7d; font-size: 0.95rem;">
                {class_info['description']}
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # Detailed probabilities
    st.markdown("### 📊 Detailed Probability Distribution")
    
    prob_cols = st.columns(5)
    for idx, (class_name, prob) in enumerate(zip(CLASS_NAMES, all_predictions)):
        with prob_cols[idx]:
            info = CLASS_DESCRIPTIONS[class_name]
            is_predicted = class_name == prediction
            border_style = f"3px solid {info['color']}" if is_predicted else "1px solid #e2e8f0"
            bg_color = "#f8fafc" if is_predicted else "white"
            
            st.markdown(f"""
            <div style="background: {bg_color}; padding: 1rem; border-radius: 10px; 
                        text-align: center; border: {border_style}; min-height: 140px;">
                <div style="font-size: 0.75rem; color: #64748b; margin-bottom: 0.5rem;">
                    {class_name.replace(' ', '<br>')}
                </div>
                <div style="font-size: 1.5rem; font-weight: 700; color: {info['color']};">
                    {prob*100:.1f}%
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    # Visual progress bars
    st.markdown("### 📈 Confidence Bars")
    for class_name, prob in sorted(zip(CLASS_NAMES, all_predictions), key=lambda x: x[1], reverse=True):
        info = CLASS_DESCRIPTIONS[class_name]
        col1, col2 = st.columns([3, 1])
        with col1:
            st.progress(float(prob))
        with col2:
            st.markdown(f"**{class_name}**: {prob*100:.1f}%")


def render_sample_images():
    """Render sample images section"""
    st.markdown("### 🔍 About the Tissue Types")
    
    tissue_cols = st.columns(5)
    tissues = [
        ("Lung Benign", "#22c55e", "Normal lung tissue"),
        ("Lung Adenocarcinoma", "#f59e0b", "Glandular lung cancer"),
        ("Lung Squamous", "#ef4444", "Squamous cell cancer"),
        ("Colon Adenocarcinoma", "#8b5cf6", "Glandular colon cancer"),
        ("Colon Benign", "#06b6d4", "Normal colon tissue")
    ]
    
    for col, (name, color, desc) in zip(tissue_cols, tissues):
        with col:
            st.markdown(f"""
            <div style="background: white; padding: 1rem; border-radius: 10px; 
                        text-align: center; border-top: 4px solid {color}; min-height: 100px;">
                <strong style="color: #1e3a5f;">{name}</strong>
                <p style="font-size: 0.8rem; color: #64748b; margin-top: 0.5rem;">{desc}</p>
            </div>
            """, unsafe_allow_html=True)


def main():
    """Main application function"""
    render_header()
    render_sidebar()
    
    # Try to load the model
    model = load_model()
    
    if model is None:
        st.warning("""
        ⚠️ **Model Not Found**
        
        The model file `lc2500_model.keras` was not found. Please ensure:
        1. The model file is in the same directory as this app
        2. The file is named exactly `lc2500_model.keras`
        
        For demo purposes, you can still upload images, but predictions won't be available.
        """)
        model_loaded = False
    else:
        model_loaded = True
        st.success("✅ Model loaded successfully!")
    
    # Upload section
    uploaded_file = render_upload_section()
    
    if uploaded_file is not None:
        try:
            # Load and display image
            image = Image.open(uploaded_file)
            
            if model_loaded:
                with st.spinner("🔄 Analyzing image..."):
                    # Preprocess and predict
                    processed_image = preprocess_image(image)
                    prediction, confidence, all_predictions = predict(model, processed_image)
                
                # Render results
                render_results(image, prediction, confidence, all_predictions)
            else:
                st.image(image, caption="Uploaded Image", use_container_width=True)
                st.info("Model not available for prediction. Please load the model to get classification results.")
                
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")
    else:
        # Show information when no image is uploaded
        render_sample_images()
        
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; padding: 2rem; background: white; border-radius: 15px; margin: 1rem 0;">
            <h3 style="color: #1e3a5f;">👆 Upload an image to get started</h3>
            <p style="color: #64748b;">
                Upload a histopathology image of lung or colon tissue to classify it using our AI model.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div class="footer">
        <p><strong>LC25000 Histopathology Classifier</strong></p>
        <p>Dataset: Lung and Colon Cancer Histopathological Images (LC25000)</p>
        <p>⚠️ For research and educational purposes only. Not intended for clinical diagnosis.</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
