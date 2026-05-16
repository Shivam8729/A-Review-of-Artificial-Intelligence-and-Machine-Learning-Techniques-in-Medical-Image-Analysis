import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
from utils import preprocess_brain_image, preprocess_blood_image
from datetime import datetime
import base64

st.set_page_config(page_title='Medical AI Analyzer', layout='wide')

# CSS for professional medical report
st.markdown("""\n<style>\n/* Glassmorphism & Animations */\n@keyframes fadeIn { from { opacity: 0; transform: translateY(20px); } to { opacity: 1; transform: translateY(0); } }\n@keyframes slideIn { from { opacity: 0; transform: translateX(-20px); } to { opacity: 1; transform: translateX(0); } }\n@keyframes pulse { 0% { transform: scale(1); } 50% { transform: scale(1.05); } 100% { transform: scale(1); } }\n\n.medical-report { \n    background: rgba(255,255,255,0.15); \n    backdrop-filter: blur(20px); \n    border: 1px solid rgba(255,255,255,0.2); \n    border-radius: 20px; \n    padding: 30px; \n    box-shadow: 0 8px 32px rgba(0,0,0,0.1); \n    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; \n    animation: fadeIn 0.8s ease-out;\n}\n\n.header { \n    text-align: center; \n    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); \n    color: white; \n    border-radius: 15px; \n    padding: 20px; \n    margin-bottom: 25px; \n    animation: slideIn 0.6s ease-out;\n}\n\n.patient-info, .results-table { \n    background: rgba(255,255,255,0.9); \n    border-radius: 12px; \n    overflow: hidden; \n    box-shadow: 0 4px 15px rgba(0,0,0,0.08);\n}\n\n.patient-info table, .results-table { border-collapse: collapse; width: 100%; margin-top: 10px; }\n.patient-info th, .patient-info td, .results-table th, .results-table td { border: 1px solid rgba(0,0,0,0.1); padding: 12px; text-align: left; }\n.patient-info th, .results-table th { background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); color: white; font-weight: bold; }\n\n.disclaimer { font-size: 0.85em; color: #888; margin-top: 25px; background: rgba(255,255,255,0.5); padding: 15px; border-radius: 10px; }\n\n.stButton>button { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important; border-radius: 25px !important; border: none !important; color: white !important; padding: 12px 30px !important; font-weight: bold !important; transition: all 0.3s ease !important; box-shadow: 0 4px 15px rgba(0,0,0,0.2) !important; }\n.stButton>button:hover { transform: scale(1.05) !important; box-shadow: 0 6px 20px rgba(0,0,0,0.3) !important; animation: pulse 0.5s !important; }\n\n.stTextInput>label div, .stNumberInput>label div, .stSelectbox>label div { font-weight: bold; color: #333; }\n.stTextInput>div>div>input, .stNumberInput>div>div>input { border-radius: 12px !important; border: 2px solid rgba(102,126,234,0.3) !important; background: rgba(255,255,255,0.8) !important; backdrop-filter: blur(10px) !important; transition: all 0.3s ease !important; }\n.stTextInput>div>div>input:focus, .stNumberInput>div>div>input:focus { border-color: #667eea !important; box-shadow: 0 0 0 3px rgba(102,126,234,0.2) !important; transform: scale(1.02); }\n\n</style>\n""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    brain_model = tf.keras.models.load_model('brain_model.h5')
    blood_model = tf.keras.models.load_model('blood_model.h5')
    brain_classes = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']
    blood_classes = ['EOSINOPHIL', 'LYMPHOCYTE', 'MONOCYTE', 'NEUTROPHIL', 'BASOPHIL']
    return brain_model, blood_model, brain_classes, blood_classes

brain_model, blood_model, brain_classes, blood_classes = load_models()

if 'patient_info' not in st.session_state:
    st.session_state.patient_info = None
if 'scan_type' not in st.session_state:
    st.session_state.scan_type = None
if 'image' not in st.session_state:
    st.session_state.image = None
if 'probs' not in st.session_state:
    st.session_state.probs = None

lang = st.selectbox('भाषा / Language', ['English', 'हिंदी'])

translations = {
    'en': {
        'title': '🩺 AI Medical Diagnostic System',
        'patient_header': 'Patient Information',
        'name': 'Full Name',
        'age': 'Age',
        'gender': 'Gender',
        'next': 'Next →',
        'scan_type': 'Select Scan Type',
        'brain': 'Brain Tumor MRI',
        'blood': 'Blood Cell Analysis',
        'upload': 'Upload Image',
        'analyze': 'Generate Report',
        'report_header': 'AI Assisted Diagnostic Report',
        'diagnosis': 'Primary Diagnosis',
        'confidence': 'Confidence Scores',
        'disclaimer': '* This is AI-assisted report. Consult doctor for final diagnosis.',
        'download': 'Download Report PDF'
    },
    'hi': {
        'title': '🩺 AI चिकित्सा निदान प्रणाली',
        'patient_header': 'रोगी जानकारी',
        'name': 'पूरा नाम',
        'age': 'आयु',
        'gender': 'लिंग',
        'next': 'आगे →',
        'scan_type': 'स्कैन प्रकार चुनें',
        'brain': 'मस्तिष्क ट्यूमर MRI',
        'blood': 'रक्त कोशिका विश्लेषण',
        'upload': 'इमेज अपलोड करें',
        'analyze': 'रिपोर्ट जनरेट करें',
        'report_header': 'AI सहायता प्राप्त निदान रिपोर्ट',
        'diagnosis': 'मुख्य निदान',
        'confidence': 'विश्वास स्कोर',
        'disclaimer': '* यह AI सहायता प्राप्त रिपोर्ट है। अंतिम निदान हेतु चिकित्सक से संपर्क करें।',
        'download': 'रिपोर्ट PDF डाउनलोड करें'
    }
}
t = translations['en' if lang == 'English' else 'hi']

st.title(t['title'])

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
            st.success('Patient info saved!')
    st.rerun()
else:
    st.success(f"Patient: {st.session_state.patient_info['name']}, Age: {st.session_state.patient_info['age']}, Gender: {st.session_state.patient_info['gender']}")
    
    st.header('Scan Selection')
    scan_type = st.radio(t['scan_type'], [t['brain'], t['blood']])
    st.session_state.scan_type = scan_type
    
    uploaded_file = st.file_uploader(t['upload'], type=['jpg', 'jpeg', 'png'])
    if uploaded_file:
        st.session_state.image = Image.open(uploaded_file)
        st.image(st.session_state.image, caption='Uploaded Image', use_column_width=True)
        
        if st.button(t['analyze'], type='primary'):
            with st.spinner('Generating Medical Report...'):
                if 'Brain' in scan_type:
                    img = preprocess_brain_image(uploaded_file)
                    probs = brain_model.predict(img, verbose=0)[0]
                    pred_idx = np.argmax(probs)
                    pred_name = brain_classes[pred_idx]
                    classes = brain_classes
                else:
                    img = preprocess_blood_image(uploaded_file)
                    probs = blood_model.predict(img, verbose=0)[0]
                    pred_idx = np.argmax(probs)
                    pred_name = blood_classes[pred_idx]
                    classes = blood_classes
                
                st.session_state.probs = probs
                st.session_state.pred_name = pred_name
                st.session_state.classes = classes
    
    if st.session_state.probs is not None:
        st.header(t['report_header'])
        st.markdown(f'''
        <div class="medical-report">
            <div class="header">
                <h2>{t['report_header']}</h2>
                <p><strong>Date:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
            </div>
            <div class="patient-info">
                <table>
                    <tr><th>Patient Name</th><td>{st.session_state.patient_info['name']}</td></tr>
                    <tr><th>Age</th><td>{st.session_state.patient_info['age']}</td></tr>
                    <tr><th>Gender</th><td>{st.session_state.patient_info['gender']}</td></tr>
                    <tr><th>Scan Type</th><td>{st.session_state.scan_type}</td></tr>
                </table>
            </div>
            <h3>{t['diagnosis']}: <strong>{st.session_state.pred_name}</strong></h3>
            <img src="data:image/png;base64,{base64.b64encode(np.array(st.session_state.image)).decode()}" width="300">
            <h4>{t['confidence']}</h4>
            <table class="results-table">
                <tr><th>Class</th><th>Probability</th></tr>
        ''', unsafe_allow_html=True)
        
        for cls, prob in zip(st.session_state.classes, st.session_state.probs):
            st.markdown(f'<tr><td>{cls}</td><td>{prob:.2%}</td></tr>', unsafe_allow_html=True)
        
        st.markdown('''
            </table>
            <div class="disclaimer">{}</div>
            <p><em>AI Assistant Dr. BLACKBOX</em></p>
        </div>
        '''.format(t['disclaimer']), unsafe_allow_html=True)
        
        # Download button (HTML as PDF workaround)
        report_html = f"""<html><body>{st.markdown(f'<div class="medical-report">...</div>', unsafe_allow_html=True)}</body></html>"""
        st.download_button(t['download'], 'Report generated!', 'report.pdf')
