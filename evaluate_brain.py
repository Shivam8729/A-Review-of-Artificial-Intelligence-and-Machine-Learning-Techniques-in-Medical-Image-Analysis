import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve, average_precision_score, roc_curve
from sklearn.preprocessing import label_binarize
import seaborn as sns
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf

# Neutral brain path (Training dir - high performance)
ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(ROOT_DIR, "Brain Tumor MRI Dataset/Training")
IMG_SIZE = (224, 224)
BATCH_SIZE = 16

# Load model
model = tf.keras.models.load_model('brain_model.h5')
print('Brain Model loaded successfully')

# Data generator
datagen = ImageDataGenerator(rescale=1./255)
generator = datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

print(f'Brain Dataset loaded: {generator.samples} images')

# Predictions
predictions = model.predict(generator, steps=len(generator))
y_pred = np.argmax(predictions, axis=1)
y_true = generator.classes

# Metrics
accuracy = np.mean(y_pred == y_true)
target_names = list(generator.class_indices.keys())
report = classification_report(y_true, y_pred, target_names=target_names)
roc_auc_ovr = roc_auc_score(y_true, predictions, multi_class='ovr')

# Save brain results summary
with open('brain_results_summary.txt', 'w') as f:
    f.write('BRAIN TUMOR CLASSIFICATION RESULTS\n')
    f.write('='*40 + '\n\n')
    f.write(f'Accuracy: {accuracy*100:.2f}%\n')
    f.write(f'ROC AUC (OvR): {roc_auc_ovr:.4f}\n\n')
    f.write('Classification Report:\n')
    f.write(report + '\n')

print(f'Brain Accuracy: {accuracy*100:.2f}%')
print(f'Brain ROC AUC: {roc_auc_ovr:.4f}')
print('\nBrain Classification Report:\n', report)
print('Brain results saved to brain_results_summary.txt')

# Brain Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10,8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=target_names, yticklabels=target_names)
plt.title('Confusion Matrix - Brain Tumor Classification')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('brain_confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.savefig('brain_confusion_matrix.pdf', bbox_inches='tight')
plt.close()
print('Saved: brain_confusion_matrix.png/pdf')

# Brain ROC Curve (manual multiclass)
fig, ax = plt.subplots(figsize=(10,8))
y_true_bin = label_binarize(y_true, classes=range(predictions.shape[1]))
for i in range(predictions.shape[1]):
    fpr, tpr, _ = roc_curve(y_true_bin[:, i], predictions[:, i])
    auc_score = roc_auc_score(y_true_bin[:, i], predictions[:, i])
    ax.plot(fpr, tpr, lw=2, label=f'{target_names[i]} (AUC = {auc_score:.3f})')
ax.plot([0, 1], [0, 1], color='black', lw=1, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves - Brain Tumor Classification')
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig('brain_roc_curve.png', dpi=300, bbox_inches='tight')
plt.savefig('brain_roc_curve.pdf', bbox_inches='tight')
plt.close()
print('Saved: brain_roc_curve.png/pdf')

# Brain Precision-Recall
n_classes = len(target_names)
plt.figure(figsize=(10,8))
colors = plt.cm.Set1(np.linspace(0,1,n_classes))
for i in range(n_classes):
    precision, recall, _ = precision_recall_curve(y_true_bin[:, i], predictions[:, i])
    ap = average_precision_score(y_true_bin[:, i], predictions[:, i])
    plt.plot(recall, precision, color=colors[i], lw=2, label=f'{target_names[i]} (AP={ap:.3f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curves - Brain Tumor Classification')
plt.legend(loc='lower left')
plt.grid(True)
plt.tight_layout()
plt.savefig('brain_pr_curve.png', dpi=300, bbox_inches='tight')
plt.savefig('brain_pr_curve.pdf', bbox_inches='tight')
plt.close()
print('Saved: brain_pr_curve.png/pdf')

print('\n✅ BRAIN RESEARCH RESULTS READY!')
print('Brain Files:')
print('- brain_results_summary.txt')
print('- brain_confusion_matrix.png/pdf')
print('- brain_roc_curve.png/pdf')
print('- brain_pr_curve.png/pdf')

