"""
LC25000 Histopathology Image Classifier
Loads model weights from Hugging Face Hub
"""

import streamlit as st
import numpy as np
import os
from PIL import Image

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
    .main { background: #f8fafc; }
    
    .main-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 50%, #0d4f6e 100%);
        padding: 2.5rem 2rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 40px rgba(15, 23, 42, 0.15);
    }
    
    .header-icon { font-size: 2.5rem; color: #38bdf8; margin-bottom: 0.5rem; }
    .main-header h1 { color: white; font-size: 2.2rem; margin-bottom: 0.5rem; font-weight: 700; }
    .main-header p { color: #94a3b8; font-size: 1.05rem; margin: 0; }
    
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
    
    .info-card-header i { color: #3b82f6; font-size: 1.1rem; }
    
    .result-card {
        background: white;
        padding: 2rem;
        border-radius: 16px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        text-align: center;
        margin-top: 1rem;
        border: 1px solid #e2e8f0;
    }
    
    .prediction-label { font-size: 1.6rem; font-weight: 700; color: #0f172a; margin-bottom: 0.5rem; }
    .confidence-text { font-size: 1.1rem; color: #64748b; }
    .confidence-value { font-weight: 700; color: #0f172a; }
    
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
    
    .badge-benign { background: #dcfce7; color: #166534; }
    .badge-malignant { background: #fee2e2; color: #991b1b; }
    
    .section-header {
        display: flex;
        align-items: center;
        gap: 10px;
        margin-bottom: 1rem;
        color: #0f172a;
        font-size: 1.25rem;
        font-weight: 600;
    }
    
    .section-header i { color: #3b82f6; }
    
    .tissue-card {
        background: white;
        padding: 1.25rem;
        border-radius: 12px;
        text-align: center;
        border: 1px solid #e2e8f0;
    }
    
    .tissue-card-title { font-weight: 600; color: #0f172a; font-size: 0.9rem; margin-bottom: 4px; }
    .tissue-card-desc { color: #64748b; font-size: 0.8rem; }
    
    .prob-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        border: 1px solid #e2e8f0;
        min-height: 120px;
    }
    
    .prob-card.active { border: 2px solid #3b82f6; background: #f8fafc; }
    .prob-label { font-size: 0.75rem; color: #64748b; margin-bottom: 8px; line-height: 1.3; }
    .prob-value { font-size: 1.4rem; font-weight: 700; }
    
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
    
    .sidebar-title i { color: #3b82f6; font-size: 0.9rem; }
    
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
    
    .class-item-text { font-size: 0.85rem; color: #334155; font-weight: 500; }
    
    .class-item-badge {
        margin-left: auto;
        font-size: 0.7rem;
        padding: 2px 8px;
        border-radius: 20px;
        font-weight: 600;
    }
    
    .stProgress > div > div {
        background: linear-gradient(90deg, #3b82f6 0%, #1d4ed8 100%);
        border-radius: 10px;
    }
    
    .disclaimer-box {
        background: #fef3c7;
        border: 1px solid #f59e0b;
        border-radius: 8px;
        padding: 12px;
        font-size: 0.8rem;
        color: #92400e;
    }
    
    .disclaimer-box i { color: #f59e0b; margin-right: 6px; }
    
    .stats-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 8px; }
    .stat-item { background: #f8fafc; padding: 10px; border-radius: 8px; text-align: center; }
    .stat-value { font-weight: 700; color: #0f172a; font-size: 1.1rem; }
    .stat-label { color: #64748b; font-size: 0.75rem; }
    
    .footer { text-align: center; padding: 2rem; color: #64748b; font-size: 0.85rem; }
    .footer-icon { color: #3b82f6; margin: 0 4px; }
    
    .empty-state {
        text-align: center;
        padding: 3rem 2rem;
        background: white;
        border-radius: 16px;
        border: 1px solid #e2e8f0;
        margin: 2rem 0;
    }
    
    .empty-state-icon { font-size: 3rem; color: #cbd5e1; margin-bottom: 1rem; }
    .empty-state-title { font-size: 1.25rem; font-weight: 600; color: #0f172a; margin-bottom: 0.5rem; }
    .empty-state-text { color: #64748b; font-size: 0.95rem; }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


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


# =============================================================================
# MODEL ARCHITECTURE
# =============================================================================

def build_model():
    """
    Rebuild the exact model architecture from training.
    This must match the architecture used to create the weights.
    """
    import tensorflow as tf
    from tensorflow.keras import layers, Model, regularizers
    
    # Config (must match training)
    weight_decay = 1e-4
    dropout_rate = 0.4
    num_classes = 5
    image_size = (224, 224)
    
    # ConvBlock function
    def conv_block(x, filters, kernel_size=3, strides=1, use_residual=False, training=False):
        shortcut = x
        
        x = layers.Conv2D(
            filters, kernel_size, strides=strides, padding='same',
            kernel_regularizer=regularizers.l2(weight_decay),
            kernel_initializer='he_normal'
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        
        x = layers.Conv2D(
            filters, kernel_size, padding='same',
            kernel_regularizer=regularizers.l2(weight_decay),
            kernel_initializer='he_normal'
        )(x)
        x = layers.BatchNormalization()(x)
        
        if use_residual:
            shortcut = layers.Conv2D(filters, 1, strides=strides, padding='same')(shortcut)
            shortcut = layers.BatchNormalization()(shortcut)
            x = layers.Add()([x, shortcut])
        
        x = layers.ReLU()(x)
        return x
    
    # SE Block function
    def se_block(x, filters, ratio=16):
        squeeze = layers.GlobalAveragePooling2D()(x)
        excite = layers.Dense(filters // ratio, activation='relu')(squeeze)
        excite = layers.Dense(filters, activation='sigmoid')(excite)
        excite = layers.Reshape((1, 1, filters))(excite)
        return layers.Multiply()([x, excite])
    
    # Build model
    inputs = layers.Input(shape=(*image_size, 3))
    
    # Normalization
    x = layers.Rescaling(1./255)(inputs)
    
    # Note: Data augmentation is NOT included in inference
    # (it was only applied during training)
    
    # Stem
    x = layers.Conv2D(64, 7, strides=2, padding='same', kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D(3, strides=2, padding='same')(x)
    
    # Stage 1 with SE attention
    x = conv_block(x, 64, use_residual=True)
    x = conv_block(x, 64, use_residual=True)
    x = se_block(x, 64)
    x = layers.Dropout(0.2)(x)
    
    # Stage 2 with SE attention
    x = conv_block(x, 128, strides=2, use_residual=True)
    x = conv_block(x, 128, use_residual=True)
    x = conv_block(x, 128, use_residual=True)
    x = se_block(x, 128)
    x = layers.Dropout(0.2)(x)
    
    # Stage 3 with SE attention
    x = conv_block(x, 256, strides=2, use_residual=True)
    x = conv_block(x, 256, use_residual=True)
    x = conv_block(x, 256, use_residual=True)
    x = conv_block(x, 256, use_residual=True)
    x = se_block(x, 256)
    x = layers.Dropout(0.3)(x)
    
    # Stage 4 with SE attention
    x = conv_block(x, 512, strides=2, use_residual=True)
    x = conv_block(x, 512, use_residual=True)
    x = conv_block(x, 512, use_residual=True)
    x = se_block(x, 512)
    x = layers.Dropout(0.3)(x)
    
    # Multi-scale feature aggregation
    gap = layers.GlobalAveragePooling2D()(x)
    gmp = layers.GlobalMaxPooling2D()(x)
    x = layers.Concatenate()([gap, gmp])
    
    # Classifier
    x = layers.Dense(512, kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(256, kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    return Model(inputs, outputs, name='LC25000_SENet_Custom')


@st.cache_resource
def load_model():
    """Download weights from Hugging Face and load model."""
    try:
        import tensorflow as tf
        from huggingface_hub import hf_hub_download
        
        HF_REPO_ID = "jeremysean/histolab"  
        HF_FILENAME = "lc25000_weights.weights.h5" 
                
        # Download weights from Hugging Face
        weights_path = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=HF_FILENAME,
            token=os.environ["HF_TOKEN"]
        )
        
        # Build model architecture
        model = build_model()
        
        # Load weights
        model.load_weights(weights_path)
        
        return model
        
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None


def preprocess_image(image, target_size=(224, 224)):
    """Preprocess image for model prediction."""
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
    """Make prediction on preprocessed image."""
    predictions = model.predict(image, verbose=0)
    predicted_class = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class]
    return CLASS_NAMES[predicted_class], confidence, predictions[0]


def render_header():
    st.markdown("""
    <div class="main-header">
        <div class="header-icon"><i class="fa-solid fa-microscope"></i></div>
        <h1>LC25000 Histopathology Classifier</h1>
        <p>Custom CNN trained from scratch for lung and colon tissue classification</p>
    </div>
    """, unsafe_allow_html=True)


def render_sidebar():
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; padding: 1rem 0;">
            <i class="fa-solid fa-microscope" style="font-size: 2.5rem; color: #3b82f6;"></i>
            <h3 style="margin: 0.5rem 0 0 0; color: #0f172a;">LC25000 Classifier</h3>
            <p style="color: #64748b; font-size: 0.8rem; margin: 0.25rem 0 0 0;">From-Scratch Model</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        st.markdown("""
        <div class="sidebar-section">
            <div class="sidebar-title"><i class="fa-solid fa-microchip"></i>Model Architecture</div>
            <p style="font-size: 0.85rem; color: #64748b; margin: 0;">
                Custom CNN with residual connections and squeeze-excitation attention blocks.
                Trained from scratch on LC25000 dataset.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="sidebar-section">
            <div class="sidebar-title"><i class="fa-solid fa-database"></i>Dataset Information</div>
            <div class="stats-grid">
                <div class="stat-item"><div class="stat-value">25K</div><div class="stat-label">Images</div></div>
                <div class="stat-item"><div class="stat-value">224px</div><div class="stat-label">Size</div></div>
                <div class="stat-item"><div class="stat-value">5</div><div class="stat-label">Classes</div></div>
                <div class="stat-item"><div class="stat-value">96%</div><div class="stat-label">Accuracy</div></div>
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
            <p style="margin: 4px 0 0 0;"><i class="fa-solid fa-face-smile" style="color: #fbbf24;"></i> Model hosted on Hugging Face</p>
        </div>
        """, unsafe_allow_html=True)


def render_upload_section():
    st.markdown("""<div class="section-header"><i class="fa-solid fa-cloud-arrow-up"></i>Upload Histopathology Image</div>""", unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader("Choose an image", type=['jpg', 'jpeg', 'png'], label_visibility="collapsed")
    
    with col2:
        st.markdown("""
        <div class="info-card">
            <div class="info-card-header"><i class="fa-solid fa-file-image"></i>Requirements</div>
            <ul style="margin: 0; padding-left: 1.2rem; color: #64748b; font-size: 0.85rem;">
                <li>Format: JPEG or PNG</li>
                <li>Recommended: 224×224 px</li>
                <li>Type: Histopathology image</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    return uploaded_file


def render_results(image, prediction, confidence, all_predictions):
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
            st.markdown(f"""<div class="{card_class}"><div class="prob-label">{class_name}</div><div class="prob-value" style="color: {info['color']};">{prob*100:.1f}%</div></div>""", unsafe_allow_html=True)
    
    st.markdown("""<div class="section-header" style="margin-top: 2rem;"><i class="fa-solid fa-chart-simple"></i>Confidence Breakdown</div>""", unsafe_allow_html=True)
    
    for class_name, prob in sorted(zip(CLASS_NAMES, all_predictions), key=lambda x: x[1], reverse=True):
        col1, col2 = st.columns([3, 1])
        with col1:
            st.progress(float(prob))
        with col2:
            st.markdown(f"<span style='font-size: 0.85rem;'><strong>{class_name}</strong>: {prob*100:.1f}%</span>", unsafe_allow_html=True)


def render_tissue_info():
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
    render_header()
    render_sidebar()
    
    model = load_model()
    
    if model is None:
        st.markdown("""
        <div class="info-card" style="border-left: 4px solid #f59e0b;">
            <div class="info-card-header"><i class="fa-solid fa-triangle-exclamation" style="color: #f59e0b;"></i>Model Loading Failed</div>
            <p style="margin: 0; color: #64748b; font-size: 0.9rem;">
                Could not load model from Hugging Face. Check your internet connection and repo settings.
            </p>
        </div>
        """, unsafe_allow_html=True)
        model_loaded = False
    else:
        st.markdown("""
        <div class="info-card" style="border-left: 4px solid #22c55e;">
            <div class="info-card-header"><i class="fa-solid fa-circle-check" style="color: #22c55e;"></i>Model Loaded Successfully</div>
            <p style="margin: 0; color: #64748b; font-size: 0.9rem;">Custom CNN loaded from Hugging Face. Ready for classification.</p>
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
                st.info("Model not available. Please check Hugging Face connection.")
                
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")
    else:
        render_tissue_info()
        
        st.markdown("""
        <div class="empty-state">
            <div class="empty-state-icon"><i class="fa-solid fa-cloud-arrow-up"></i></div>
            <div class="empty-state-title">Upload an image to get started</div>
            <div class="empty-state-text">Upload a histopathology image of lung or colon tissue to classify it.</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("""
    <div class="footer">
        <p style="margin: 0;"><strong>LC25000 Histopathology Classifier</strong></p>
        <p style="margin: 4px 0;"><i class="fa-solid fa-database footer-icon"></i>Dataset: Lung and Colon Cancer Histopathological Images</p>
        <p style="margin: 4px 0;"><i class="fa-solid fa-triangle-exclamation footer-icon" style="color: #f59e0b;"></i>For research and educational purposes only.</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()