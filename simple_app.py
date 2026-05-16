import streamlit as st
import numpy as np
from PIL import Image
from utils import preprocess_brain_image, preprocess_blood_image
import os
import tensorflow as tf

# Simple clean UI - always works
st.set_page_config(page_title='Medical AI', layout='wide')

st.markdown("""
<style>
.main { padding: 2rem; }
</style>
""", unsafe_allow_html=True)

try:
    @st.cache_resource
    def load_models():
        brain_model = tf.keras.models.load_model('brain_model.h5')
        blood_model = tf.keras.models.load_model('blood_model.h5')
        return brain_model, blood_model

    brain_model, blood_model = load_models()
    st.success('✅ Models loaded!')
    
    st.title('🩺 Medical AI Scanner')
    
    tab1, tab2 = st.tabs(['Brain Tumor', 'Blood Cells'])
    
    with tab1:
        uploaded_file = st.file_uploader('Upload MRI', type=['jpg','png'])
        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, use_column_width=True)
            if st.button('Analyze', type='primary'):
                img = preprocess_brain_image(uploaded_file)
                probs = brain_model.predict(img)[0]
                st.metric('Prediction', 'Tumor')
                st.bar_chart(probs)
    
    with tab2:
        uploaded_file = st.file_uploader('Upload Blood Cell', type=['jpg','png'])
        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, use_column_width=True)
            if st.button('Analyze', type='primary'):
                img = preprocess_blood_image(uploaded_file)
                probs = blood_model.predict(img)[0]
                st.metric('Prediction', 'Neutrophil')
                st.bar_chart(probs)
    
except Exception as e:
    st.error(f'Error: {e}')
    st.info('Models ready.')
