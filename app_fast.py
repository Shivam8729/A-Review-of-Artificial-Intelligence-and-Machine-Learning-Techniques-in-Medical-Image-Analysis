import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
from utils import preprocess_brain_image, preprocess_blood_image
from datetime import datetime
import base64
import os

# Suppress TensorFlow warnings for cleaner UI
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

st.set_page_config(page_title='Medical AI Analyzer - Fast', layout='wide')

# CSS (same as original for consistency)
st.markdown("""
<style>
/* [Same CSS as original - glassmorphism, animations ] */
@keyframes fadeIn { from { opacity: 0; transform: translateY(20px); } to { opacity: 1; transform: translateY(0); } }
@keyframes slideIn { from { opacity: 0; transform: translateX(-20px); } to { opacity: 1; transform: translateX(0); } }
@keyframes pulse { 0% { transform: scale(1); } 50% { transform: scale(1.05); } 100% { transform: scale(1); } }

.medical-report { 
    background: rgba(255,255,255,0.15); 
    backdrop-filter: blur(20px); 
    border: 1px solid rgba(255,255,255,0.2); 
    border-radius: 20px; 
    padding: 30px; 
    box-shadow: 0 8px 32px rgba(0,0,0,0.1); 
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
    animation: fadeIn 0.8s ease-out;
}
.header { 
    text-align: center; 
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
    color: white; 
    border-radius: 15px; 
    padding: 20px; 
    margin-bottom: 25px; 
    animation: slideIn 0.6s ease-out;
}
.patient-info, .results-table { 
    background: rgba(255,255,255,0.95); 
    color: #000 !important;
    border-radius: 12px; 
    overflow: hidden; 
    box-shadow: 0 4px 15px rgba(0,0,0,0.08);
}
.patient-info td, .results-table td { color: #000 !important; }
.patient-info th, .results-table th { color: #fff !important; }
.patient-info table, .results-table { border-collapse: collapse; width: 100%; margin-top: 10px; }
.patient-info th, .patient-info td, .results-table th, .results-table td { border: 1px solid rgba(0,0,0,0.1); padding: 12px; text-align: left; }
.patient-info th, .results-table th { background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); color: white; font-weight: bold; }
.disclaimer { font-size: 0.85em; color: #333 !important; margin-top: 25px; background: rgba(255,255,255,0.8); padding: 15px; border-radius: 10px; }
.stButton>button { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important; border-radius: 25px !important; border: none !important; color: white !important; padding: 12px 30px !important; font-weight: bold !important; transition: all 0.3s ease !important; box-shadow: 0 4px 15px rgba(0,0,0,0.2) !important; }
.stButton>button:hover { transform: scale(1.05) !important; box-shadow: 0 6px 20px rgba(0,0,0,0.3) !important; animation: pulse 0.5s !important; }
.stTextInput>label div, .stNumberInput>label div, .stSelectbox>label div { font-weight: bold; color: #333; }
.stTextInput>div>div>input, .stNumberInput>div>div>input { border-radius: 12px !important; border: 2px solid rgba(102,126,234,0.3) !important; background: rgba(255,255,255,0.8) !important; backdrop-filter: blur(10px) !important; transition: all 0.3s ease !important; }
.stTextInput>div>div>input:focus, .stNumberInput>div>div>input:focus { border-color: #667eea !important; box-shadow: 0 0 0 3px rgba(102,126,234,0.2) !important; transform: scale(1.02); }
</style>
""", unsafe_allow_html=True)

# ============= LAZY MODEL LOADERS =============
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def _resolve_model_path(preferred_name: str, fallback_name: str) -> str:
    """Return preferred file path if present, otherwise fallback path.

    Searches relative to both:
    - project root (cwd)
    - medical_ai_app folder (this script dir)
    """
    candidates = [
        # Prefer fixed/non-fixed that are stored in the same folder as this script
        os.path.join(BASE_DIR, preferred_name),
        os.path.join(BASE_DIR, fallback_name),
        # Also consider that the artifacts might live in the project root
        os.path.join(os.getcwd(), preferred_name),
        os.path.join(os.getcwd(), fallback_name),
        # Absolute/implicit candidates
        preferred_name,
        fallback_name,
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    # Return the most likely preferred path for a better error message
    return os.path.join(BASE_DIR, preferred_name)


@st.cache_resource
def load_brain_model():
    """Lazy load ONLY brain model - pre-compiled"""
    st.info("⚙️ Loading optimized Brain Tumor model (first time only)...")

    brain_model_path = _resolve_model_path(
        preferred_name="brain_model_fixed.h5",  # if you add it later
        fallback_name="brain_model.h5",
    )

    try:
        model = tf.keras.models.load_model(brain_model_path)
        model.compile(
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'],
            optimizer='adam'
        )
    except Exception as e:
        st.error(f"❌ Failed to load brain model from: {brain_model_path}")
        raise e

    st.success("✅ Brain model ready!")
    return model, ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']


@st.cache_resource
def load_blood_model():
    """Lazy load ONLY blood model - pre-compiled"""
    st.info("⚙️ Loading optimized Blood Cell model (first time only)...")

    blood_model_path = _resolve_model_path(
        preferred_name="blood_model_fixed.h5",
        fallback_name="blood_model.h5",
    )
    # If fixed labels/models exist only in project root, also try root-specific filenames.
    # (Keeps backward compatibility with different training scripts.)

    try:
        model = tf.keras.models.load_model(blood_model_path)
        model.compile(
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'],
            optimizer='adam'
        )
    except Exception as e:
        st.error(f"❌ Failed to load blood model from: {blood_model_path}")
        raise e

    st.success("✅ Blood model ready!")
    # If you later store labels in .npy, you can load from that here.
    return model, ['EOSINOPHIL', 'LYMPHOCYTE', 'MONOCYTE', 'NEUTROPHIL', 'BASOPHIL']

# ============= CACHED PREPROCESS =============
@st.cache_data
def cached_preprocess_brain(uploaded_file):
    return preprocess_brain_image(uploaded_file)

@st.cache_data
def cached_preprocess_blood(uploaded_file):
    return preprocess_blood_image(uploaded_file)

# Session state (expanded)
if 'patient_info' not in st.session_state:
    st.session_state.patient_info = None
if 'scan_type' not in st.session_state:
    st.session_state.scan_type = 'Brain Tumor MRI'
if 'image' not in st.session_state:
    st.session_state.image = None
if 'probs' not in st.session_state:
    st.session_state.probs = None
if 'models_loaded' not in st.session_state:
    st.session_state.models_loaded = {'brain': False, 'blood': False}

lang = st.selectbox('भाषा / Language', ['English', 'हिंदी'])

translations = {
    'en': {
        'title': '🩺 AI Medical Diagnostic System - Optimized',
        'patient_header': 'Patient Information', 
        'name': 'Full Name',
        'age': 'Age',
        'gender': 'Gender', 
        'next': 'Next →',
        'scan_type': 'Select Scan Type',
        'brain': '🧠 Brain Tumor MRI (224x224)',
        'blood': '🩸 Blood Cell Analysis (128x128)',
        'upload': 'Upload Medical Image (JPG/PNG)',
        'analyze': '🔬 Analyze & Generate Report',
        'report_header': 'AI Assisted Diagnostic Report',
        'diagnosis': 'Primary Diagnosis',
        'confidence': 'Confidence Scores', 
        'disclaimer': '* AI-assisted only. Consult your doctor.',
        'download': '📥 Download Report'
    },
    'hi': {
        'title': '🩺 AI चिकित्सा निदान प्रणाली - तेज',
        'patient_header': 'रोगी जानकारी',
        'name': 'पूरा नाम',
        'age': 'आयु',
        'gender': 'लिंग',
        'next': 'आगे →',
        'scan_type': 'स्कैन प्रकार चुनें', 
        'brain': '🧠 मस्तिष्क ट्यूमर MRI',
        'blood': '🩸 रक्त कोशिका विश्लेषण',
        'upload': 'मेडिकल इमेज अपलोड करें',
        'analyze': 'विश्लेषण करें',
        'report_header': 'AI सहायता प्राप्त रिपोर्ट',
        'diagnosis': 'मुख्य निदान',
        'confidence': 'आत्मविश्वास स्कोर',
        'disclaimer': '* AI सहायता। डॉक्टर से संपर्क करें।',
        'download': 'रिपोर्ट डाउनलोड'
    }
}
t = translations['en' if lang == 'English' else 'hi']

st.title(t['title'])

# ================== STEP 1: PATIENT INFO ==================
if st.session_state.patient_info is None:
    st.header(t['patient_header'])
    with st.form('patient_form'):
        col1, col2, col3 = st.columns(3)
        with col1:
            name = st.text_input(t['name'])
        with col2:
            age = st.number_input(t['age'], min_value=1, max_value=120)
        with col3:
            gender = st.selectbox(t['gender'], ['Male', 'Female', 'Other'])
        if st.form_submit_button(t['next']):
            st.session_state.patient_info = {'name': name, 'age': age, 'gender': gender}
            st.rerun()
else:
    st.success(f"👤 {st.session_state.patient_info['name']} | Age: {st.session_state.patient_info['age']} | {st.session_state.patient_info['gender']}")

    # ================== STEP 2: SCAN TYPE ==================
    st.header(t['scan_type'])
    scan_type = st.radio(t['scan_type'], [t['brain'], t['blood']], horizontal=True, key='scan_radio')
    
    if scan_type != st.session_state.scan_type:
        st.session_state.scan_type = scan_type
        st.session_state.probs = None  # Reset results
        st.rerun()

    # ================== STEP 3: IMAGE UPLOAD ==================
    uploaded_file = st.file_uploader(t['upload'], type=['jpg', 'jpeg', 'png'], key='uploader')
    
    if uploaded_file is not None:
        st.session_state.image = Image.open(uploaded_file)
        st.image(st.session_state.image, caption='Uploaded Scan', width=400)
        
        # Analyze button with disable logic
        if st.button(t['analyze'], type='primary', disabled=st.session_state.probs is not None):
            with st.spinner(f'🔬 AI Analysis in progress...\nModel loading if first time (10-20s)'):
                if 'Brain' in scan_type:
                    if not st.session_state.models_loaded['brain']:
                        brain_model, brain_classes = load_brain_model()
                        st.session_state.models_loaded['brain'] = True
                    else:
                        brain_model, brain_classes = load_brain_model()
                    
                    img = cached_preprocess_brain(uploaded_file)
                    probs = brain_model.predict(img, verbose=0)[0]
                    
                else:  # Blood
                    if not st.session_state.models_loaded['blood']:
                        blood_model, blood_classes = load_blood_model()
                        st.session_state.models_loaded['blood'] = True
                    else:
                        blood_model, blood_classes = load_blood_model()
                    
                    img = cached_preprocess_blood(uploaded_file)
                    probs = blood_model.predict(img, verbose=0)[0]
                
                st.session_state.probs = probs
                st.session_state.pred_name = brain_classes[np.argmax(probs)] if 'Brain' in scan_type else blood_classes[np.argmax(probs)]
                st.session_state.classes = brain_classes if 'Brain' in scan_type else blood_classes
                st.rerun()
    
    # ================== STEP 4: RESULTS ==================
    if st.session_state.probs is not None:
        st.header(t['report_header'])
        
        col1, col2 = st.columns([1, 2])
        with col1:
            st.markdown(f"""
            <div class="medical-report">
                <div class="header">
                    <h3>📋 Report Generated</h3>
                    <p><strong>Time:</strong> {datetime.now().strftime('%H:%M:%S')}</p>
                </div>
                <div class="patient-info">
                    <table>
                        <tr><th>{t['name']}</th><td>{st.session_state.patient_info['name']}</td></tr>
                        <tr><th>{t['age']}</th><td>{st.session_state.patient_info['age']}</td></tr>  
                        <tr><th>{t['gender']}</th><td>{st.session_state.patient_info['gender']}</td></tr>
                        <tr><th>Scan</th><td>{st.session_state.scan_type}</td></tr>
                    </table>
                </div>
                <h4>🎯 {t['diagnosis']}: **{st.session_state.pred_name}**</h4>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown('<img src="data:image/png;base64,' + base64.b64encode(np.array(st.session_state.image)).decode() + '" width="400">', unsafe_allow_html=True)
        
        st.subheader(t['confidence'])
        st.markdown("""
        <table class="results-table">
            <tr><th>Diagnosis</th><th>Probability</th></tr>
        """, unsafe_allow_html=True)
        
        for cls, prob in zip(st.session_state.classes, st.session_state.probs):
            st.markdown(f'<tr><td>{cls}</td><td><strong>{prob:.1%}</strong></td></tr>', unsafe_allow_html=True)
        
        st.markdown("""
            </table>
            <div class="disclaimer">""" + t['disclaimer'] + """</div>
            </div>

        """, unsafe_allow_html=True)
        
        st.button("🔄 New Analysis", on_click=lambda: [setattr(st.session_state, k, None if k in ['probs', 'image'] else v) for k, v in st.session_state.items()], type='secondary')
