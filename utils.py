import os
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ------------------ UNIVERSAL IMAGE LOADER ------------------
def load_image(input_data):
    """
    Handles:
    - file path
    - uploaded file
    - PIL image
    """
    if isinstance(input_data, Image.Image):
        img = input_data
    else:
        img = Image.open(input_data)

    if img.mode != 'RGB':
        img = img.convert('RGB')

    return img


# ------------------ PREPROCESS FUNCTIONS ------------------
def preprocess_brain_image(input_data, target_size=(224, 224)):
    img = load_image(input_data)
    img = img.resize(target_size)
    img = np.array(img) / 255.0
    return np.expand_dims(img, axis=0)


def preprocess_blood_image(input_data, target_size=(128, 128)):
    img = load_image(input_data)
    img = img.resize(target_size)
    img = np.array(img) / 255.0
    return np.expand_dims(img, axis=0)


# ------------------ LOAD BLOOD DATASET ------------------
def load_blood_data(base_dir):
    labels_path = os.path.join(base_dir, 'labels.csv')
    images_dir = os.path.join(base_dir, 'JPEGImages')
    
    df = pd.read_csv(labels_path)

    valid_rows = []
    for idx in range(len(df)):
        category = str(df.iloc[idx]['Category']).strip()
        
        # Ignore multi-label
        if category and ',' not in category:
            image_num = int(df.iloc[idx]['Image'])
            image_name = f'BloodImage_{image_num:05d}.jpg'
            valid_rows.append({'image': image_name, 'category': category})

    df_valid = pd.DataFrame(valid_rows)

    # Label encoding
    class_to_idx = {cls: i for i, cls in enumerate(sorted(df_valid['category'].unique()))}
    df_valid['label'] = df_valid['category'].map(class_to_idx)

    images = []
    labels = []

    for _, row in df_valid.iterrows():
        img_path = os.path.join(images_dir, row['image'])

        if os.path.exists(img_path):
            try:
                img = load_image(img_path)
                img = img.resize((128, 128))

                images.append(np.array(img) / 255.0)
                labels.append(row['label'])

            except Exception as e:
                print(f'Skip {img_path}: {e}')

    return np.array(images), np.array(labels), class_to_idx


# ------------------ DATA AUGMENTATION ------------------
def get_datagen():
    return ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        zoom_range=0.2
    )