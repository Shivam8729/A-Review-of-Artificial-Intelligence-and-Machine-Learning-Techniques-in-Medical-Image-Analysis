import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from PIL import Image
import glob
from sklearn.model_selection import train_test_split
from utils import get_datagen, load_blood_data

# Base dirs (adjust if needed)
BRAIN_BASE = r'c:/Users/rahul/OneDrive/Desktop/Review of Ai and ML techniques in medical image Analysis/Brain Tumor MRI Dataset'
BLOOD_BASE = r'c:/Users/rahul/OneDrive/Desktop/Review of Ai and ML techniques in medical image Analysis/Blood Cell Images/dataset-master/dataset-master'

print('Loading Brain Tumor data...')
classes_brain = ['glioma', 'meningioma', 'notumor', 'pituitary']
class_to_idx_brain = {cls: i for i, cls in enumerate(classes_brain)}

X_brain, y_brain = [], []
for split in ['Training', 'Testing']:
    for cls in classes_brain:
        path = os.path.join(BRAIN_BASE, split, cls, '*')
        for img_path in glob.glob(path):
            try:
                img = Image.open(img_path)
                if img.mode == 'L':
                    img = img.convert('RGB')
                img = img.convert('RGB')
                img = img.resize((224, 224))
                X_brain.append(np.array(img) / 255.0)
                y_brain.append(class_to_idx_brain[cls])
            except Exception as e:
                print(f'Skip {img_path}: {e}')

X_brain = np.array(X_brain)
y_brain = np.eye(len(classes_brain))[y_brain]  # One-hot
X_train_b, X_val_b, y_train_b, y_val_b = train_test_split(X_brain, y_brain, test_size=0.2)

# Blood data
print('Loading Blood Cell data...')
X_blood, y_blood, class_to_idx_blood = load_blood_data(BLOOD_BASE)
idx_to_class_blood = {v: k for k, v in class_to_idx_blood.items()}
y_blood_cat = np.eye(len(class_to_idx_blood))[y_blood]
X_train_bl, X_val_bl, y_train_bl, y_val_bl = train_test_split(X_blood, y_blood_cat, test_size=0.2)

print(f'Brain data: {X_train_b.shape}, Blood data: {X_train_bl.shape}')

# Brain CNN (ResNet50 base)
base_model_b = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model_b.trainable = False
brain_model = Sequential([
    base_model_b,
    GlobalAveragePooling2D(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(4, activation='softmax')
])
brain_model.compile(optimizer=Adam(1e-3), loss='categorical_crossentropy', metrics=['accuracy'])
cb = [EarlyStopping(patience=5), ModelCheckpoint('brain_model.h5', save_best_only=True)]
brain_model.fit(X_train_b, y_train_b, epochs=20, batch_size=32, validation_data=(X_val_b, y_val_b), callbacks=cb)
print('Brain model saved as brain_model.h5')

# Blood CNN
blood_model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Conv2D(256, (3,3), activation='relu'),
    GlobalAveragePooling2D(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(len(class_to_idx_blood), activation='softmax')
])
blood_model.compile(optimizer=Adam(1e-3), loss='categorical_crossentropy', metrics=['accuracy'])
cb = [EarlyStopping(patience=5), ModelCheckpoint('blood_model.h5', save_best_only=True)]
blood_model.fit(X_train_bl, y_train_bl, epochs=30, batch_size=32, validation_data=(X_val_bl, y_val_bl), callbacks=cb, class_weight={i:1/len(class_to_idx_blood) for i in range(len(class_to_idx_blood))})
print('Blood model saved as blood_model.h5')

# Save class mappings
np.save('brain_classes.npy', classes_brain)
np.save('blood_classes.npy', list(idx_to_class_blood.values()))

print('Training complete!')
