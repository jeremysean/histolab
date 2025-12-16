"""
LC25000 Histopathology Image Classifier
Loads .keras model from Hugging Face Hub
"""

import streamlit as st
import numpy as np
from PIL import Image

st.set_page_config(
    page_title="LC25000 Histopathology Classifier",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.1/css/all.min.css">
<style>
    .main { background: #f8fafc; }
    .main-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 50%, #0d4f6e 100%);
        padding: 2.5rem 2rem; border-radius: 16px; margin-bottom: 2rem;
        box-shadow: 0 10px 40px rgba(15, 23, 42, 0.15);
    }
    .header-icon { font-size: 2.5rem; color: #38bdf8; margin-bottom: 0.5rem; }
    .main-header h1 { color: white; font-size: 2.2rem; margin-bottom: 0.5rem; font-weight: 700; }
    .main-header p { color: #94a3b8; font-size: 1.05rem; margin: 0; }
    .info-card {
        background: white; padding: 1.5rem; border-radius: 12px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08); margin-bottom: 1rem; border: 1px solid #e2e8f0;
    }
    .info-card-header { display: flex; align-items: center; gap: 10px; margin-bottom: 12px; color: #1e293b; font-weight: 600; }
    .info-card-header i { color: #3b82f6; font-size: 1.1rem; }
    .result-card {
        background: white; padding: 2rem; border-radius: 16px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08); text-align: center; margin-top: 1rem; border: 1px solid #e2e8f0;
    }
    .prediction-label { font-size: 1.6rem; font-weight: 700; color: #0f172a; margin-bottom: 0.5rem; }
    .confidence-text { font-size: 1.1rem; color: #64748b; }
    .confidence-value { font-weight: 700; color: #0f172a; }
    .badge { display: inline-flex; align-items: center; gap: 6px; padding: 8px 16px; border-radius: 50px; font-weight: 600; font-size: 0.9rem; margin-top: 1rem; }
    .badge-benign { background: #dcfce7; color: #166534; }
    .badge-malignant { background: #fee2e2; color: #991b1b; }
    .section-header { display: flex; align-items: center; gap: 10px; margin-bottom: 1rem; color: #0f172a; font-size: 1.25rem; font-weight: 600; }
    .section-header i { color: #3b82f6; }
    .tissue-card { background: white; padding: 1.25rem; border-radius: 12px; text-align: center; border: 1px solid #e2e8f0; }
    .tissue-card-title { font-weight: 600; color: #0f172a; font-size: 0.9rem; margin-bottom: 4px; }
    .tissue-card-desc { color: #64748b; font-size: 0.8rem; }
    .prob-card { background: white; padding: 1rem; border-radius: 10px; text-align: center; border: 1px solid #e2e8f0; min-height: 120px; }
    .prob-card.active { border: 2px solid #3b82f6; background: #f8fafc; }
    .prob-label { font-size: 0.75rem; color: #64748b; margin-bottom: 8px; line-height: 1.3; }
    .prob-value { font-size: 1.4rem; font-weight: 700; }
    .sidebar-section { background: white; padding: 1.25rem; border-radius: 12px; margin-bottom: 1rem; border: 1px solid #e2e8f0; }
    .sidebar-title { display: flex; align-items: center; gap: 8px; font-weight: 600; color: #0f172a; margin-bottom: 12px; font-size: 0.95rem; }
    .sidebar-title i { color: #3b82f6; font-size: 0.9rem; }
    .class-item { display: flex; align-items: center; gap: 10px; padding: 10px 12px; background: #f8fafc; border-radius: 8px; margin-bottom: 8px; border-left: 3px solid; }
    .class-item-text { font-size: 0.85rem; color: #334155; font-weight: 500; }
    .class-item-badge { margin-left: auto; font-size: 0.7rem; padding: 2px 8px; border-radius: 20px; font-weight: 600; }
    .stProgress > div > div { background: linear-gradient(90deg, #3b82f6 0%, #1d4ed8 100%); border-radius: 10px; }
    .disclaimer-box { background: #fef3c7; border: 1px solid #f59e0b; border-radius: 8px; padding: 12px; font-size: 0.8rem; color: #92400e; }
    .disclaimer-box i { color: #f59e0b; margin-right: 6px; }
    .stats-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 8px; }
    .stat-item { background: #f8fafc; padding: 10px; border-radius: 8px; text-align: center; }
    .stat-value { font-weight: 700; color: #0f172a; font-size: 1.1rem; }
    .stat-label { color: #64748b; font-size: 0.75rem; }
    .footer { text-align: center; padding: 2rem; color: #64748b; font-size: 0.85rem; }
    .footer-icon { color: #3b82f6; margin: 0 4px; }
    .empty-state { text-align: center; padding: 3rem 2rem; background: white; border-radius: 16px; border: 1px solid #e2e8f0; margin: 2rem 0; }
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
    "Colon Adenocarcinoma": {"description": "Cancer in glandular cells of colon.", "severity": "Malignant", "color": "#8b5cf6"},
    "Colon Benign Tissue": {"description": "Normal, healthy colon tissue.", "severity": "Benign", "color": "#06b6d4"},
    "Lung Adenocarcinoma": {"description": "Most common type of lung cancer.", "severity": "Malignant", "color": "#f59e0b"},
    "Lung Benign Tissue": {"description": "Normal, healthy lung tissue.", "severity": "Benign", "color": "#22c55e"},
    "Lung Squamous Cell Carcinoma": {"description": "Cancer in squamous cells of airways.", "severity": "Malignant", "color": "#ef4444"},
}


@st.cache_resource
def load_model():
    """Download and load .keras model from Hugging Face."""
    try:
        import tensorflow as tf
        from huggingface_hub import hf_hub_download
        
        # =====================================================
        # EDIT THIS: Your Hugging Face repo info
        # =====================================================
        HF_REPO_ID = "jeremysean/histolab"  # <- Ganti dengan repo kamu
        HF_FILENAME = "lc25000_best.keras"     # <- Nama file model
        # =====================================================
        
        model_path = hf_hub_download(repo_id=HF_REPO_ID, filename=HF_FILENAME)
        model = tf.keras.models.load_model(model_path)
        return model
        
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None


def preprocess_image(image, target_size=(224, 224)):
    image = image.resize(target_size)
    img_array = np.array(image)
    if len(img_array.shape) == 2:
        img_array = np.stack([img_array] * 3, axis=-1)
    elif img_array.shape[-1] == 4:
        img_array = img_array[:, :, :3]
    img_array = img_array.astype('float32')
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


CONFIDENCE_THRESHOLD = 0.90  # 95% threshold

def predict(model, image):
    predictions = model.predict(image, verbose=0)
    predicted_class = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class]
    return CLASS_NAMES[predicted_class], confidence, predictions[0]


def render_header():
    st.markdown("""
    <div class="main-header">
        <div class="header-icon"><i class="fa-solid fa-microscope"></i></div>
        <h1>LC25000 Histopathology Classifier</h1>
        <p>AI-powered lung and colon tissue classification</p>
    </div>
    """, unsafe_allow_html=True)


def render_sidebar():
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
            <div class="sidebar-title"><i class="fa-solid fa-database"></i>Dataset Info</div>
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
            <strong>Disclaimer:</strong> For research purposes only. Not for clinical diagnosis.
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; color: #64748b; font-size: 0.8rem;">
            <p style="margin: 0;">Built with Streamlit</p>
            <p style="margin: 4px 0 0 0;"><i class="fa-solid fa-face-smile" style="color: #fbbf24;"></i> Model on Hugging Face</p>
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
                <li>Format: JPEG/PNG</li>
                <li>Size: 224×224 px</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    return uploaded_file


def render_results(image, prediction, confidence, all_predictions):
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""<div class="section-header"><i class="fa-solid fa-image"></i>Uploaded Image</div>""", unsafe_allow_html=True)
        st.image(image, use_container_width=True)
    
    with col2:
        st.markdown("""<div class="section-header"><i class="fa-solid fa-bullseye"></i>Result</div>""", unsafe_allow_html=True)
        
        # Check confidence threshold
        if confidence < CONFIDENCE_THRESHOLD:
            # Low confidence - not detected
            st.markdown(f"""
            <div class="result-card" style="border: 2px solid #f59e0b;">
                <div style="font-size: 2.5rem; color: #f59e0b; margin-bottom: 1rem;">
                    <i class="fa-solid fa-circle-question"></i>
                </div>
                <div class="prediction-label" style="color: #f59e0b;">Tissue Not Recognized</div>
                <div class="confidence-text">Confidence: <span class="confidence-value">{confidence*100:.1f}%</span></div>
                <div class="badge" style="background: #fef3c7; color: #92400e;">
                    <i class="fa-solid fa-triangle-exclamation"></i> Below {CONFIDENCE_THRESHOLD*100:.0f}% Threshold
                </div>
                <p style="margin-top: 1rem; color: #64748b; font-size: 0.9rem;">
                    The uploaded image could not be confidently classified. Please ensure the image is a valid histopathology sample of lung or colon tissue.
                </p>
            </div>
            """, unsafe_allow_html=True)
        else:
            # High confidence - show prediction
            class_info = CLASS_DESCRIPTIONS[prediction]
            badge = '<div class="badge badge-benign"><i class="fa-solid fa-shield-check"></i> Benign</div>' if class_info["severity"] == "Benign" else '<div class="badge badge-malignant"><i class="fa-solid fa-triangle-exclamation"></i> Malignant</div>'
            st.markdown(f"""
            <div class="result-card">
                <div class="prediction-label" style="color: {class_info['color']};">{prediction}</div>
                <div class="confidence-text">Confidence: <span class="confidence-value">{confidence*100:.1f}%</span></div>
                {badge}
                <p style="margin-top: 1rem; color: #64748b; font-size: 0.9rem;">{class_info['description']}</p>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("""<div class="section-header" style="margin-top: 2rem;"><i class="fa-solid fa-chart-bar"></i>All Probabilities</div>""", unsafe_allow_html=True)
    
    for class_name, prob in sorted(zip(CLASS_NAMES, all_predictions), key=lambda x: x[1], reverse=True):
        col1, col2 = st.columns([3, 1])
        with col1:
            st.progress(float(prob))
        with col2:
            st.markdown(f"**{class_name}**: {prob*100:.1f}%")


def render_tissue_info():
    st.markdown("""<div class="section-header"><i class="fa-solid fa-circle-info"></i>Tissue Types</div>""", unsafe_allow_html=True)
    cols = st.columns(5)
    tissues = [
        ("Lung Benign", "#22c55e", "fa-lungs"),
        ("Lung Adeno", "#f59e0b", "fa-disease"),
        ("Lung Squamous", "#ef4444", "fa-virus"),
        ("Colon Adeno", "#8b5cf6", "fa-bacteria"),
        ("Colon Benign", "#06b6d4", "fa-shield")
    ]
    for col, (name, color, icon) in zip(cols, tissues):
        with col:
            st.markdown(f"""
            <div class="tissue-card" style="border-top: 4px solid {color};">
                <i class="fa-solid {icon}" style="font-size: 1.5rem; color: {color}; margin-bottom: 8px; display: block;"></i>
                <div class="tissue-card-title">{name}</div>
            </div>
            """, unsafe_allow_html=True)


def main():
    render_header()
    render_sidebar()
    
    model = load_model()
    model_loaded = model is not None
    
    if model_loaded:
        st.markdown("""
        <div class="info-card" style="border-left: 4px solid #22c55e;">
            <div class="info-card-header"><i class="fa-solid fa-circle-check" style="color: #22c55e;"></i>Model Loaded</div>
            <p style="margin: 0; color: #64748b; font-size: 0.9rem;">Ready for classification.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="info-card" style="border-left: 4px solid #f59e0b;">
            <div class="info-card-header"><i class="fa-solid fa-triangle-exclamation" style="color: #f59e0b;"></i>Model Failed</div>
            <p style="margin: 0; color: #64748b; font-size: 0.9rem;">Check Hugging Face connection.</p>
        </div>
        """, unsafe_allow_html=True)
    
    uploaded_file = render_upload_section()
    
    if uploaded_file:
        try:
            image = Image.open(uploaded_file)
            if model_loaded:
                with st.spinner("Analyzing..."):
                    processed = preprocess_image(image)
                    prediction, confidence, all_preds = predict(model, processed)
                render_results(image, prediction, confidence, all_preds)
            else:
                st.image(image, caption="Uploaded Image", use_container_width=True)
        except Exception as e:
            st.error(f"Error: {str(e)}")
    else:
        render_tissue_info()
        st.markdown("""
        <div class="empty-state">
            <div class="empty-state-icon"><i class="fa-solid fa-cloud-arrow-up"></i></div>
            <div class="empty-state-title">Upload an image to start</div>
            <div class="empty-state-text">Upload a histopathology image for classification.</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("""
    <div class="footer">
        <p><strong>LC25000 Histopathology Classifier</strong></p>
        <p><i class="fa-solid fa-triangle-exclamation footer-icon" style="color: #f59e0b;"></i>For research only.</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()