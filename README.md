# Medical AI App 🩺

## Setup
1. Open terminal in project dir.
2. `cd medical_ai_app`
3. `python -m venv venv`
4. `venv\\Scripts\\activate`
5. `pip install -r requirements.txt`

## Train Models (First Time)
```
python train_models.py
```
- Trains CNNs on datasets.
- Saves `brain_model.h5`, `blood_model.h5`, class mappings.
- May take 30-60 mins depending on hardware.

## Run UI
```
streamlit run "medical_ai_app/app_fast.py"
```
- Opens http://localhost:8501
- Tabs for Brain Tumor (4 classes) & Blood Cells (5 classes).
- Animated UI with confidence report cards.

## Datasets
- Assumes paths in code match your dir.
- Brain: 4 classes, high accuracy.
- Blood: Filtered to 5 single classes.

Enjoy! 🚀
