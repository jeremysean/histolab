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

# Custom CSS with Font Awesome icons
st.markdown("""
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.1/css/all.min.css">
<style>
    /* Main container styling */
    .main {
        background: #f8fafc;
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 50%, #0d4f6e 100%);
        padding: 2.5rem 2rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 40px rgba(15, 23, 42, 0.15);
        position: relative;
        overflow: hidden;
    }
    
    .main-header::before {
        content: '';
        position: absolute;
        top: 0;
        right: 0;
        width: 300px;
        height: 100%;
        background: linear-gradient(135deg, rgba(255,255,255,0.1) 0%, transparent 100%);
        border-radius: 0 16px 16px 0;
    }
    
    .header-icon {
        font-size: 2.5rem;
        color: #38bdf8;
        margin-bottom: 0.5rem;
    }
    
    .main-header h1 {
        color: white;
        font-size: 2.2rem;
        margin-bottom: 0.5rem;
        font-weight: 700;
        letter-spacing: -0.5px;
    }
    
    .main-header p {
        color: #94a3b8;
        font-size: 1.05rem;
        margin: 0;
    }
    
    /* Card styling */
    .info-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
        margin-bottom: 1rem;
        border: 1px solid #e2e8f0;
    }
    
    .info-card-header {
        display: flex;
        align-items: center;
        gap: 10px;
        margin-bottom: 12px;
        color: #1e293b;
        font-weight: 600;
    }
    
    .info-card-header i {
        color: #3b82f6;
        font-size: 1.1rem;
    }
    
    /* Result card */
    .result-card {
        background: white;
        padding: 2rem;
        border-radius: 16px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        text-align: center;
        margin-top: 1rem;
        border: 1px solid #e2e8f0;
    }
    
    .prediction-label {
        font-size: 1.6rem;
        font-weight: 700;
        color: #0f172a;
        margin-bottom: 0.5rem;
    }
    
    .confidence-text {
        font-size: 1.1rem;
        color: #64748b;
    }
    
    .confidence-value {
        font-weight: 700;
        color: #0f172a;
    }
    
    /* Severity badges */
    .badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 8px 16px;
        border-radius: 50px;
        font-weight: 600;
        font-size: 0.9rem;
        margin-top: 1rem;
    }
    
    .badge-benign {
        background: #dcfce7;
        color: #166534;
    }
    
    .badge-malignant {
        background: #fee2e2;
        color: #991b1b;
    }
    
    /* Section headers */
    .section-header {
        display: flex;
        align-items: center;
        gap: 10px;
        margin-bottom: 1rem;
        color: #0f172a;
        font-size: 1.25rem;
        font-weight: 600;
    }
    
    .section-header i {
        color: #3b82f6;
    }
    
    /* Tissue type cards */
    .tissue-card {
        background: white;
        padding: 1.25rem;
        border-radius: 12px;
        text-align: center;
        border: 1px solid #e2e8f0;
        transition: all 0.2s ease;
    }
    
    .tissue-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    
    .tissue-card-title {
        font-weight: 600;
        color: #0f172a;
        font-size: 0.9rem;
        margin-bottom: 4px;
    }
    
    .tissue-card-desc {
        color: #64748b;
        font-size: 0.8rem;
    }
    
    /* Probability cards */
    .prob-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        border: 1px solid #e2e8f0;
        min-height: 120px;
    }
    
    .prob-card.active {
        border: 2px solid #3b82f6;
        background: #f8fafc;
    }
    
    .prob-label {
        font-size: 0.75rem;
        color: #64748b;
        margin-bottom: 8px;
        line-height: 1.3;
    }
    
    .prob-value {
        font-size: 1.4rem;
        font-weight: 700;
    }
    
    /* Sidebar styling */
    .sidebar-section {
        background: white;
        padding: 1.25rem;
        border-radius: 12px;
        margin-bottom: 1rem;
        border: 1px solid #e2e8f0;
    }
    
    .sidebar-title {
        display: flex;
        align-items: center;
        gap: 8px;
        font-weight: 600;
        color: #0f172a;
        margin-bottom: 12px;
        font-size: 0.95rem;
    }
    
    .sidebar-title i {
        color: #3b82f6;
        font-size: 0.9rem;
    }
    
    /* Class list in sidebar */
    .class-item {
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 10px 12px;
        background: #f8fafc;
        border-radius: 8px;
        margin-bottom: 8px;
        border-left: 3px solid;
    }
    
    .class-item-text {
        font-size: 0.85rem;
        color: #334155;
        font-weight: 500;
    }
    
    .class-item-badge {
        margin-left: auto;
        font-size: 0.7rem;
        padding: 2px 8px;
        border-radius: 20px;
        font-weight: 600;
    }
    
    /* Progress bar override */
    .stProgress > div > div {
        background: linear-gradient(90deg, #3b82f6 0%, #1d4ed8 100%);
        border-radius: 10px;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 8px;
        font-weight: 600;
        width: 100%;
        transition: all 0.2s;
    }
    
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.4);
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem;
        color: #64748b;
        font-size: 0.85rem;
    }
    
    .footer-icon {
        color: #3b82f6;
        margin: 0 4px;
    }
    
    /* Disclaimer box */
    .disclaimer-box {
        background: #fef3c7;
        border: 1px solid #f59e0b;
        border-radius: 8px;
        padding: 12px;
        font-size: 0.8rem;
        color: #92400e;
    }
    
    .disclaimer-box i {
        color: #f59e0b;
        margin-right: 6px;
    }
    
    /* Stats grid */
    .stats-grid {
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 8px;
    }
    
    .stat-item {
        background: #f8fafc;
        padding: 10px;
        border-radius: 8px;
        text-align: center;
    }
    
    .stat-value {
        font-weight: 700;
        color: #0f172a;
        font-size: 1.1rem;
    }
    
    .stat-label {
        color: #64748b;
        font-size: 0.75rem;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Image container */
    .image-container {
        background: white;
        padding: 1rem;
        border-radius: 12px;
        border: 1px solid #e2e8f0;
    }
    
    /* Empty state */
    .empty-state {
        text-align: center;
        padding: 3rem 2rem;
        background: white;
        border-radius: 16px;
        border: 1px solid #e2e8f0;
        margin: 2rem 0;
    }
    
    .empty-state-icon {
        font-size: 3rem;
        color: #cbd5e1;
        margin-bottom: 1rem;
    }
    
    .empty-state-title {
        font-size: 1.25rem;
        font-weight: 600;
        color: #0f172a;
        margin-bottom: 0.5rem;
    }
    
    .empty-state-text {
        color: #64748b;
        font-size: 0.95rem;
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
    },
    "Colon Benign Tissue": {
        "description": "Normal, non-cancerous colon tissue with healthy cellular structure.",
        "severity": "Benign",
        "color": "#06b6d4",
    },
    "Lung Adenocarcinoma": {
        "description": "The most common type of lung cancer, originating in glandular cells.",
        "severity": "Malignant",
        "color": "#f59e0b",
    },
    "Lung Benign Tissue": {
        "description": "Normal, healthy lung tissue without any cancerous cells.",
        "severity": "Benign",
        "color": "#22c55e",
    },
    "Lung Squamous Cell Carcinoma": {
        "description": "A type of lung cancer that begins in squamous cells lining the airways.",
        "severity": "Malignant",
        "color": "#ef4444",
    }
}


@st.cache_resource
def load_model():
    """Build model architecture and load weights"""
    try:
        import tensorflow as tf
        from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Rescaling, InputLayer, Dropout
        from tensorflow.keras.applications import MobileNetV3Small
        
        base = MobileNetV3Small(
            include_top=False,
            weights='imagenet',
            input_shape=(224, 224, 3),
        )
        base.trainable = True
        
        model = tf.keras.models.Sequential([
            InputLayer(shape=(224, 224, 3)),
            Rescaling(scale=1/127.5, offset=-1),
            base,
            GlobalAveragePooling2D(),
            Dropout(0.2),
            Dense(256, activation='relu'),
            Dense(5, activation="softmax"),
        ])
        
        model.load_weights('lc2500_weights.weights.h5')
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None


def preprocess_image(image, target_size=(224, 224)):
    """Preprocess image for model prediction"""
    image = image.resize(target_size)
    img_array = np.array(image)
    if len(img_array.shape) == 2:
        img_array = np.stack([img_array] * 3, axis=-1)
    elif img_array.shape[-1] == 4:
        img_array = img_array[:, :, :3]
    img_array = img_array.astype('float32')
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
        <div class="header-icon"><i class="fa-solid fa-microscope"></i></div>
        <h1>LC25000 Histopathology Classifier</h1>
        <p>AI-powered lung and colon cancer tissue classification from histopathological images</p>
    </div>
    """, unsafe_allow_html=True)


def render_sidebar():
    """Render the sidebar with information"""
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; padding: 1rem 0;">
            <i class="fa-solid fa-microscope" style="font-size: 2.5rem; color: #3b82f6;"></i>
            <h3 style="margin: 0.5rem 0 0 0; color: #0f172a;">LC25000 Classifier</h3>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        st.markdown("""
        <div class="sidebar-section">
            <div class="sidebar-title">
                <i class="fa-solid fa-circle-info"></i>
                About This Tool
            </div>
            <p style="font-size: 0.85rem; color: #64748b; margin: 0;">
                Deep learning-powered classification of histopathological images 
                into five tissue types for lung and colon cancer detection.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="sidebar-section">
            <div class="sidebar-title">
                <i class="fa-solid fa-database"></i>
                Dataset Information
            </div>
            <div class="stats-grid">
                <div class="stat-item">
                    <div class="stat-value">25K</div>
                    <div class="stat-label">Images</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value">224px</div>
                    <div class="stat-label">Size</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value">5</div>
                    <div class="stat-label">Classes</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value">JPEG</div>
                    <div class="stat-label">Format</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""<div class="sidebar-section"><div class="sidebar-title"><i class="fa-solid fa-layer-group"></i>Tissue Classes</div>""", unsafe_allow_html=True)
        
        for class_name, info in CLASS_DESCRIPTIONS.items():
            badge_bg = "#dcfce7" if info["severity"] == "Benign" else "#fee2e2"
            badge_color = "#166534" if info["severity"] == "Benign" else "#991b1b"
            st.markdown(f"""
            <div class="class-item" style="border-left-color: {info['color']};">
                <span class="class-item-text">{class_name}</span>
                <span class="class-item-badge" style="background: {badge_bg}; color: {badge_color};">{info['severity']}</span>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("""
        <div class="disclaimer-box">
            <i class="fa-solid fa-triangle-exclamation"></i>
            <strong>Disclaimer:</strong> For educational and research purposes only. Not for clinical diagnosis.
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        st.markdown("""
        <div style="text-align: center; color: #64748b; font-size: 0.8rem;">
            <p style="margin: 0;">Built with <i class="fa-solid fa-heart" style="color: #ef4444;"></i> using Streamlit</p>
            <p style="margin: 4px 0 0 0;"><i class="fa-brands fa-python" style="color: #3776ab;"></i> Powered by TensorFlow</p>
        </div>
        """, unsafe_allow_html=True)


def render_upload_section():
    """Render the image upload section"""
    st.markdown("""
    <div class="section-header">
        <i class="fa-solid fa-cloud-arrow-up"></i>
        Upload Histopathology Image
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Choose an image",
            type=['jpg', 'jpeg', 'png'],
            help="Upload a histopathology image for classification",
            label_visibility="collapsed"
        )
    
    with col2:
        st.markdown("""
        <div class="info-card">
            <div class="info-card-header">
                <i class="fa-solid fa-file-image"></i>
                Requirements
            </div>
            <ul style="margin: 0; padding-left: 1.2rem; color: #64748b; font-size: 0.85rem;">
                <li>Format: JPEG or PNG</li>
                <li>Recommended: 224×224 px</li>
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
        st.markdown("""<div class="section-header"><i class="fa-solid fa-image"></i>Uploaded Image</div>""", unsafe_allow_html=True)
        st.image(image, use_container_width=True)
    
    with col2:
        st.markdown("""<div class="section-header"><i class="fa-solid fa-bullseye"></i>Classification Result</div>""", unsafe_allow_html=True)
        
        if class_info["severity"] == "Benign":
            badge_html = '<div class="badge badge-benign"><i class="fa-solid fa-shield-check"></i> Benign</div>'
        else:
            badge_html = '<div class="badge badge-malignant"><i class="fa-solid fa-triangle-exclamation"></i> Malignant</div>'
        
        st.markdown(f"""
        <div class="result-card">
            <div class="prediction-label" style="color: {class_info['color']};">{prediction}</div>
            <div class="confidence-text">Confidence: <span class="confidence-value">{confidence*100:.1f}%</span></div>
            {badge_html}
            <p style="margin-top: 1rem; color: #64748b; font-size: 0.9rem;">{class_info['description']}</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("""<div class="section-header" style="margin-top: 2rem;"><i class="fa-solid fa-chart-bar"></i>Probability Distribution</div>""", unsafe_allow_html=True)
    
    prob_cols = st.columns(5)
    for idx, (class_name, prob) in enumerate(zip(CLASS_NAMES, all_predictions)):
        with prob_cols[idx]:
            info = CLASS_DESCRIPTIONS[class_name]
            is_predicted = class_name == prediction
            card_class = "prob-card active" if is_predicted else "prob-card"
            st.markdown(f"""
            <div class="{card_class}">
                <div class="prob-label">{class_name}</div>
                <div class="prob-value" style="color: {info['color']};">{prob*100:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("""<div class="section-header" style="margin-top: 2rem;"><i class="fa-solid fa-chart-simple"></i>Confidence Breakdown</div>""", unsafe_allow_html=True)
    
    for class_name, prob in sorted(zip(CLASS_NAMES, all_predictions), key=lambda x: x[1], reverse=True):
        info = CLASS_DESCRIPTIONS[class_name]
        col1, col2 = st.columns([3, 1])
        with col1:
            st.progress(float(prob))
        with col2:
            st.markdown(f"<span style='font-size: 0.85rem;'><strong>{class_name}</strong>: {prob*100:.1f}%</span>", unsafe_allow_html=True)


def render_sample_images():
    """Render sample images section"""
    st.markdown("""<div class="section-header"><i class="fa-solid fa-circle-info"></i>About the Tissue Types</div>""", unsafe_allow_html=True)
    
    tissue_cols = st.columns(5)
    tissues = [
        ("Lung Benign", "#22c55e", "Normal lung tissue", "fa-lungs"),
        ("Lung Adenocarcinoma", "#f59e0b", "Glandular lung cancer", "fa-disease"),
        ("Lung Squamous", "#ef4444", "Squamous cell cancer", "fa-virus"),
        ("Colon Adenocarcinoma", "#8b5cf6", "Glandular colon cancer", "fa-bacteria"),
        ("Colon Benign", "#06b6d4", "Normal colon tissue", "fa-shield")
    ]
    
    for col, (name, color, desc, icon) in zip(tissue_cols, tissues):
        with col:
            st.markdown(f"""
            <div class="tissue-card" style="border-top: 4px solid {color};">
                <i class="fa-solid {icon}" style="font-size: 1.5rem; color: {color}; margin-bottom: 8px; display: block;"></i>
                <div class="tissue-card-title">{name}</div>
                <div class="tissue-card-desc">{desc}</div>
            </div>
            """, unsafe_allow_html=True)


def main():
    """Main application function"""
    render_header()
    render_sidebar()
    
    model = load_model()
    
    if model is None:
        st.markdown("""
        <div class="info-card" style="border-left: 4px solid #f59e0b;">
            <div class="info-card-header">
                <i class="fa-solid fa-triangle-exclamation" style="color: #f59e0b;"></i>
                Model Not Found
            </div>
            <p style="margin: 0; color: #64748b; font-size: 0.9rem;">
                The weights file <code>lc2500_weights.weights.h5</code> was not found. Ensure it's in the same directory as this app.
            </p>
        </div>
        """, unsafe_allow_html=True)
        model_loaded = False
    else:
        st.markdown("""
        <div class="info-card" style="border-left: 4px solid #22c55e;">
            <div class="info-card-header">
                <i class="fa-solid fa-circle-check" style="color: #22c55e;"></i>
                Model Loaded Successfully
            </div>
            <p style="margin: 0; color: #64748b; font-size: 0.9rem;">Ready to classify histopathology images.</p>
        </div>
        """, unsafe_allow_html=True)
        model_loaded = True
    
    uploaded_file = render_upload_section()
    
    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file)
            
            if model_loaded:
                with st.spinner("Analyzing image..."):
                    processed_image = preprocess_image(image)
                    prediction, confidence, all_predictions = predict(model, processed_image)
                render_results(image, prediction, confidence, all_predictions)
            else:
                st.image(image, caption="Uploaded Image", use_container_width=True)
                st.info("Model not available. Please load the model for predictions.")
                
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")
    else:
        render_sample_images()
        
        st.markdown("""
        <div class="empty-state">
            <div class="empty-state-icon"><i class="fa-solid fa-cloud-arrow-up"></i></div>
            <div class="empty-state-title">Upload an image to get started</div>
            <div class="empty-state-text">Upload a histopathology image of lung or colon tissue to classify it using our AI model.</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("""
    <div class="footer">
        <p style="margin: 0;"><strong>LC25000 Histopathology Classifier</strong></p>
        <p style="margin: 4px 0;"><i class="fa-solid fa-database footer-icon"></i>Dataset: Lung and Colon Cancer Histopathological Images (LC25000)</p>
        <p style="margin: 4px 0;"><i class="fa-solid fa-triangle-exclamation footer-icon" style="color: #f59e0b;"></i>For research and educational purposes only.</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()