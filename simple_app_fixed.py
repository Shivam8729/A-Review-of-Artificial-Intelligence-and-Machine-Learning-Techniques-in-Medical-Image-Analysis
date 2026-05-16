import streamlit as st
import numpy as np
from PIL import Image
from utils import preprocess_brain_image, preprocess_blood_image
import tensorflow as tf
import datetime

st.set_page_config(page_title='Medical AI System', layout='wide')

# ------------------ LOAD MODELS ------------------
@st.cache_resource
def load_models():
    brain_model = tf.keras.models.load_model('brain_model.h5')
    blood_model = tf.keras.models.load_model('blood_model.h5')
    return brain_model, blood_model

# ------------------ DARK UI ------------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0f172a, #1e293b);
    color: white;
}
h1 {
    text-align: center;
    color: #38bdf8;
}
.card {
    background: rgba(255,255,255,0.05);
    padding: 20px;
    border-radius: 15px;
    margin-bottom: 20px;
}
.stButton>button {
    background: linear-gradient(90deg, #06b6d4, #3b82f6);
    color: white;
    border-radius: 10px;
    height: 50px;
    font-size: 18px;
}
</style>
""", unsafe_allow_html=True)

# ------------------ LOAD ------------------
try:
    brain_model, blood_model = load_models()
    st.success('✅ Models loaded successfully! 🩸 New Blood Model (96%+ Accuracy)')

    st.title("🩺 Medical AI Diagnostic System")

    # ------------------ PATIENT FORM ------------------
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("👤 Patient Details")

    col1, col2 = st.columns(2)

    with col1:
        name = st.text_input("Patient Name")
        age = st.number_input("Age", min_value=1, max_value=120)

    with col2:
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])

    st.markdown('</div>', unsafe_allow_html=True)

    # ------------------ TEST TYPE ------------------
    st.markdown('<div class="card">', unsafe_allow_html=True)
    test_type = st.selectbox(
        "🧠 Select Test Type",
        ["Brain Tumor MRI", "Blood Cell Analysis"]
    )
    st.markdown('</div>', unsafe_allow_html=True)

    # ------------------ UPLOAD ------------------
    st.markdown('<div class="card">', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("📤 Upload Medical Image", type=["jpg","png","jpeg"])
    st.markdown('</div>', unsafe_allow_html=True)

    # ------------------ ANALYZE ------------------
    if uploaded_file and st.button("🔍 Generate AI Report"):

        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        brain_classes = np.load('brain_classes.npy')
        blood_classes = np.load('blood_classes.npy')

        with st.spinner("🤖 AI is analyzing..."):

            if test_type == "Brain Tumor MRI":
                img = preprocess_brain_image(image)
                probs = brain_model.predict(img)[0]
                classes = brain_classes
            else:
                img = preprocess_blood_image(image)
                probs = blood_model.predict(img)[0]
                classes = blood_classes

        pred_class = np.argmax(probs)
        confidence = np.max(probs) * 100
        current_time = datetime.datetime.now().strftime("%d-%m-%Y %H:%M:%S")

        # ------------------ WHITE REPORT STYLE ------------------
        st.markdown("""
        <style>
        .report-container {
            background: white;
            color: black;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 0 15px rgba(0,0,0,0.3);
            font-family: 'Times New Roman', serif;
        }
        .report-title {
            text-align: center;
            font-size: 28px;
            font-weight: bold;
            border-bottom: 2px solid black;
            margin-bottom: 20px;
        }
        .section-title {
            font-size: 20px;
            font-weight: bold;
            margin-top: 20px;
            border-bottom: 1px solid #ccc;
        }
        .footer {
            margin-top: 30px;
            font-size: 12px;
            text-align: center;
            color: gray;
        }
        </style>
        """, unsafe_allow_html=True)

        st.markdown('<div class="report-container">', unsafe_allow_html=True)

        st.markdown('<div class="report-title">AI Medical Diagnostic Report</div>', unsafe_allow_html=True)

        # Patient Info
        st.markdown('<div class="section-title">Patient Information</div>', unsafe_allow_html=True)
        st.write(f"Name: {name}")
        st.write(f"Age: {age}")
        st.write(f"Gender: {gender}")
        st.write(f"Date: {current_time}")

        # Diagnosis
        st.markdown('<div class="section-title">Diagnosis Result</div>', unsafe_allow_html=True)
        st.write(f"Condition Detected: {classes[pred_class]}")
        st.write(f"Confidence: {confidence:.2f}%")

        # Analysis
        st.markdown('<div class="section-title">AI Analysis</div>', unsafe_allow_html=True)
        st.write(f"""
The uploaded medical image was analyzed using a Convolutional Neural Network (CNN).
The system predicts the condition as '{classes[pred_class]}' with a confidence of {confidence:.2f}%.
""")

        # Recommendation
        st.markdown('<div class="section-title">Recommendation</div>', unsafe_allow_html=True)
        st.write("""
- Consult a certified medical professional
- Perform further diagnostic tests
- Do not rely only on AI report
""")

        # Footer
        st.markdown("""
        <div class="footer">
        This is an AI-generated report and not a substitute for professional diagnosis.<br>
        Authorized AI Medical System
        </div>
        """, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

        # Chart
        st.subheader("📊 Prediction Probability")
        st.bar_chart(probs)

except Exception as e:
    st.error(f'Error: {e}')