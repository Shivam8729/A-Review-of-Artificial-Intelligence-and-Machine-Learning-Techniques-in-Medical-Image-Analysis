import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize

# ------------------ PATH FIX (IMPORTANT 🔥) ------------------

BASE_DIR = os.path.dirname(__file__)        # medical_ai_app
ROOT_DIR = os.path.dirname(BASE_DIR)        # main project folder

# Models (inside medical_ai_app)
BRAIN_MODEL = os.path.join(BASE_DIR, "brain_model.h5")
BLOOD_MODEL = os.path.join(BASE_DIR, "blood_model.h5")

# Datasets (outside folder - based on your screenshot)
BRAIN_TEST = os.path.join(ROOT_DIR, "Brain Tumor MRI Dataset")
BLOOD_TEST = os.path.join(ROOT_DIR, "Blood Cell Images")

# Output folder
FIGURES_DIR = os.path.join(ROOT_DIR, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

# ------------------ LOAD DATA ------------------
def load_data(path, img_size):
    datagen = ImageDataGenerator(rescale=1./255)

    data = datagen.flow_from_directory(
        path,
        target_size=img_size,
        batch_size=32,
        class_mode='categorical',
        shuffle=False
    )
    return data

# ------------------ CONFUSION MATRIX ------------------
def plot_confusion_matrix(model, data, title, save_name):
    y_true = data.classes
    y_pred = np.argmax(model.predict(data), axis=1)

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    plt.savefig(os.path.join(FIGURES_DIR, f"{save_name}.png"))
    plt.savefig(os.path.join(FIGURES_DIR, f"{save_name}.pdf"))
    plt.close()

    print(f"✅ Saved: {save_name}.png/pdf")

# ------------------ ROC CURVE ------------------
def multi_class_roc(model, data, save_name):
    print("📊 Generating ROC curve...")

    y_true = data.classes
    y_pred_prob = model.predict(data)

    n_classes = y_pred_prob.shape[1]
    y_true_bin = label_binarize(y_true, classes=list(range(n_classes)))

    plt.figure()

    for i in range(n_classes):

        # Skip if class not present
        if np.sum(y_true_bin[:, i]) == 0:
            print(f"⚠️ Skipping class {i} (no samples)")
            continue

        try:
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred_prob[:, i])
            roc_auc = auc(fpr, tpr)

            plt.plot(fpr, tpr, lw=2, label=f'Class {i} (AUC={roc_auc:.2f})')

        except Exception as e:
            print(f"⚠️ Error in class {i}: {e}")

    plt.plot([0,1],[0,1],'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()

    plt.savefig(os.path.join(FIGURES_DIR, f"{save_name}.png"))
    plt.savefig(os.path.join(FIGURES_DIR, f"{save_name}.pdf"))
    plt.close()

    print(f"✅ Saved: {save_name}.png/pdf")

# ------------------ ACCURACY BAR ------------------
def accuracy_bar(brain_acc, blood_acc):
    labels = ['Brain MRI', 'Blood Cells']
    values = [brain_acc, blood_acc]

    plt.figure()
    plt.bar(labels, values)
    plt.title("Model Accuracy Comparison")

    plt.savefig(os.path.join(FIGURES_DIR, "accuracy_bar.png"))
    plt.savefig(os.path.join(FIGURES_DIR, "accuracy_bar.pdf"))
    plt.close()

    print("✅ Saved: accuracy_bar.png/pdf")

# ------------------ MAIN ------------------
def evaluate():

    print("🚀 Generating research paper-ready figures...")

    # Load models
    brain_model = load_model(BRAIN_MODEL)
    blood_model = load_model(BLOOD_MODEL)

    # Load data
    brain_data = load_data(BRAIN_TEST, (224,224))
    blood_data = load_data(BLOOD_TEST, (128,128))

    # ------------------ ACCURACY ------------------
    brain_pred = np.argmax(brain_model.predict(brain_data), axis=1)
    brain_true = brain_data.classes
    brain_acc = np.mean(brain_pred == brain_true)

    blood_pred = np.argmax(blood_model.predict(blood_data), axis=1)
    blood_true = blood_data.classes
    blood_acc = np.mean(blood_pred == blood_true)

    print(f"Brain Accuracy: {brain_acc}")
    print(f"Blood Accuracy: {blood_acc}")

    # ------------------ CONFUSION MATRICES ------------------
    plot_confusion_matrix(brain_model, brain_data, "Brain Confusion Matrix", "brain_confusion")
    plot_confusion_matrix(blood_model, blood_data, "Blood Confusion Matrix", "blood_confusion")

    # ------------------ ROC ------------------
    multi_class_roc(brain_model, brain_data, "brain_roc")
    multi_class_roc(blood_model, blood_data, "blood_roc")

    # ------------------ ACCURACY GRAPH ------------------
    accuracy_bar(brain_acc, blood_acc)

    print("🎉 ALL FIGURES GENERATED SUCCESSFULLY!")

# ------------------ RUN ------------------
if __name__ == "__main__":
    evaluate()